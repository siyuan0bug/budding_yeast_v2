# PEAKS、HGGS、ERACS、Low-Rank的回归实现

2026-11-18故障修复：HGGS跨轮target覆盖、Lev非有限得分/重复ID、Physics-PRE事件过度屏蔽已定向修复并真实输入重放；ERACS增加未触发警告及机制汇总，阈值不改。历史补跑25完成/5失败状态不变，未启动新正式补跑。见[修复说明](BASELINE_REPAIRS_2026-11-18.md)。

2026-11-17 20:47更新：新增GPU小探针、并发入口验证后，seed42补跑已在tmux启动；进度和路径见[本批说明](SEED42_BASELINE_EXTENSIONS_2026-11-17.md)。第8节“未启动正式GPU实验”描述的是实现完成时的历史测试状态。

本次用户授权实现四类方法，允许生成新输入，要求原生物规则、初值及ODE只复用不改。实现位于`utils/sampling/baseline/`，不import/exec参考仓库；选择、流程、训练、评测分别组织。本文件说明可执行范围及适配边界，不将小规模测试当作正式性能结果。

## 1. 实现入口与原文

|论文|项目入口|工作流|明确的任务适配|
|---|---|---|---|
|[Gurbuz等，PEAKS，ICML2025](https://proceedings.mlr.press/v267/gurbuz25a.html)|`dynamic --method peaks_regression`|连续接收合成样本、分位数接受、更新、最终收尾训练|回归的selected-reference kernel influence替代原分类class-logit prototype，名称为PEAKS-Regression-Reference|
|[Rao等，HGGS，AAAI2025](https://ojs.aaai.org/index.php/AAAI/article/view/34920)|`generate --method hggs_projected`|GF边界排名＋pilot/global残差保留，再MGS生成/ODE/训练|39维轨迹表征替代振荡频率；既有两池中的随机coarse bank；MGS坐标投影到新合法提案，分轮冷启动|
|[Cheng等，ERACS，NeurIPS2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/92a035beb61e68f7a688467e5103a17c-Abstract-Conference.html)|`dynamic --method eracs_regression`|训练中检查SNR、低于阈值触发非负带权OMP、替换resident subset|使用FNO实际Relative L2输出头梯度及随机投影、默认Batch-OMP分组。不是固定五轮只追加|
|[Cohen-Addad等，Low-Rank，ICML2026](https://proceedings.mlr.press/v306/cohen-addad26a.html)|`run --method low_rank`|初始pilot、Algorithm2低秩敏感度、有放回带权抽样、最终训练|梯度嵌入上的能量loss延拓；不声称保持原Relative L2的低秩理论保证|

裸注册名`peaks/hggs`仍显式指向对应工作流，不用同名占位器掩盖协议差异；`ogas`裸名同样指向此前已经实现的generation入口。当前四项回归适配验证对象是FNO，不声称所有模型已经适配。

## 2. 共用回归梯度

`baseline/regression.py`对当前训练损失求导：

`L_i = mean_c ||prediction_i,c - target_i,c|| / sqrt(||target_i,c||² + eps²)`。

沿时间求范数，epsilon读取实际criterion；没有ratio截断。FNO末层是共享时间的线性Conv1d：

`g_i,W = sum_t (dL_i/dprediction_i,:,t) outer hidden_i,:,t`，`g_i,b = sum_t dL_i/dprediction_i,:,t`。

这与每个样本的autograd输出头梯度逐项对账；不是把39维输出当39个类别，不使用softmax或one-hot。默认投影到128维用于动态方法，Low-Rank使用协议的sketch维度；投影会改变几何与梯度近似，需声明。输入/初值保留物理精度，模型输入仍float32。

## 3. PEAKS：保留流式IDS，明确回归评分

原文Eq.1/2从一次参数更新对预测的影响出发；原分类近似Eq.11使用真实类别对应的输出权重作为prototype。连续轨迹标签没有唯一的“真实类别输出权重”，因此不能把Eq.11直接照搬。

本实现使用已选训练参考的平均输出头梯度`g_ref`，对候选计算：

`score_i = dot(g_ref, g_i)`。

未投影时，它等于`output_error_derivative_ref^T K(ref,i) output_error_derivative_i`，是一次SGD步对已选参考loss的首阶改善量（差一个公共学习率）；测试用有限差分验证。这是**参考式回归适配**，不是原PEAKS分类class-logit代理，也不声称精确预测AdamW整网更新的效果。

保留的完整流程：

1. 先仅用共用初始集训练f0（默认`bootstrap_epochs=5`），然后才从固定训练侧source流中抽候选，按实际访问数揭示真轨迹；没有预先用real/test打分。
2. 将当前score加入cache，按与作者`percentileofscore(kind=rank)`一致的百分位规则接受高分样本；保留负分及tie语义。
3. 每`refresh_steps`个训练更新清空cache，重新计算selected-training参考；有限cache容量为资源适配。
4. 每次实际训练批次包含`delta`个新接受样本和其余旧样本；在Lightning预取之后替换真实batch，避免新样本延迟/漏入。
5. 同一模型/optimizer继续训练；样本达到目标后收尾训练，不重复冷启动。

默认delta8、选择比例10%、refresh500更新、参考64条、候选读取64条。达到扫描上限明确报错，不暗中随机补齐。`random_stream`提供同样连续训练和new+old批次的随机对照。

## 4. ERACS：SNR触发的动态替换

按论文附录，计算当前coreset的梯度均值和协方差**谱范数**：

`SNR = ||mean(g_i)|| / sqrt(||Cov(g_i)||_spectral)`。

已有sample weight时，以实际加权样本梯度计算。分母只加数值下限。采用Algorithm1的`SNR < threshold`触发，和理论“低SNR表示代表性退化”一致；原文实验段另有“exceeds”措辞，未据此把触发方向反转。

触发后，只读取训练侧池的监督轨迹，计算回归梯度，使用非负带ridge OMP拟合全训练侧均值梯度。默认按100条样本组成group匹配；`omp_group_size=1`可选逐样本版，但90k/10k规模代价很高。group成员随机顺序固定；最后一组截取时重新计算其均值再求解，保证账本与实际权重对应。

对group系数`a_j`、组大小`n_j`、当前总resident数量`m`，成员权重为`m*a_j/n_j`。因此共享训练器的`mean_i(weight_i * L_i)`精确对应OMP选定的组系数，而非再按batch权重和归一化。若匹配已到驻点但预算未满，继续填充的组和零权重数量均显式记录；零梯度时保留零估计器，不偷偷改成uniform。

默认每5epoch检查、threshold=.1；这些是项目基准起点，未声称作者已在酵母验证。ERACS起始resident达到最终容量（共用初始2000＋训练侧随机扩充至10000），这与五轮从2000逐步追加的协议不同；初始归一化仍只拟合共同2000。必须先与`random_dynamic`的连续训练随机替换比较，不能混称完全相同训练预算。随机对照在每个检查周期替换，不复制ERACS的SNR触发序列；检查/替换次数及实际标签成本另报。

## 5. Low-Rank：基方向loss有定义，无需伪造ODE样本

选择嵌入为每条训练侧轨迹的FNO输出头梯度（可投影）。定义选择用loss延拓：`ell(v)=||v||²`。它对任何梯度基向量都成立，因此正交SVD基上`ell(v_j)=1`；不需要把基向量当成酵母参数，也没有向ODE查询假标签。原文合成实验使用平方范数代价，另一实验使用逐样本梯度的范数proxy；当前“梯度特征＋平方能量”结合这两种构件，属于明确的任务适配，不是任一原实验的逐项复现，也不能声称与FNO的原Relative loss完全相同。实验表述为Low-Rank-GradientEnergy。

按论文Algorithm2，`x= sum_j alpha_j v_j + r(x)`：

`sigma(x) = (gamma+1) sum_j alpha_j² ell(v_j) + gamma*k*xi + lambda*||r(x)||²`。

默认rank32、gamma5、lambda1、xi1e-6。保留basis-loss项、投影残差项和常数项，不仅是一个裸SVD分数。按sigma比例独立有放回抽取m个draw slots，权重`1/(m*p_i)`；重复样本合并后换算为共享训练器对唯一行取mean时的正确线性权重。

所以10000个draw slots可能少于10000条唯一输入，符合该weighted sampling版本；报告draw slots、重复次数、唯一训练数及全部访问真值数。若另做无放回版本，必须独立命名并说明权重变化，不能沿用这里的无偏估计器。

## 6. HGGS：GF和MGS均有，而非只有residual_topk

GF先从训练侧source随机提取默认5×初始预算的coarse bank（通常10000），共享初始2000包含在内：

- 根据原Eq.4计算K近邻的平均`||output_j-output_i||² / ||input_j-input_i||²`，分块查询KNN，不保留N²矩阵。
- 取梯度最高的边界部分；用它试训pilot，再从其余coarse点按真实预测残差保留全局部分。
- 这里output是用初始训练尺度归一化、随机投影后的39变量轨迹，而非原文的振荡频率；input是标准化参数＋实际初值。默认最终初始2000由边界1000＋全局1000组成，因此**不保证与其他方法随机初始2000相同**。
- 原文LHS coarse初始化被本项目既有source范式替代；全部coarse标签访问和pilot训练成本都报告，不能当作只访问最终2000。

MGS在每个采样轮开始时，用当前模型重算已查询训练点残差；按三分量GMM划分low/middle/high。退化时采用作者源码同样支持的分位数分层，明确记日志。原Eq.5的逐坐标box采样保留：

`target = alpha * (parent_a-parent_b) + parent_b`，各维alpha独立U(0,1)。

交叉用high/high两个不同父样本，变异用high/middle；默认交叉/变异各50%，跨小批次累计精确配额。target只是算法期望坐标，最终投影到**新抽取的合法提案原样对象**；参数、初值、合法mutation和ODE完全不插值、不修改。故名称为`hggs_projected`，不是原无约束连续box的原样复现。

当前流程仍使用已有冷启动轮次服务，并显式记录此训练日程适配。采样过程中checkpoint保留本轮残差分层和RNG；恢复后不会把刚补采的标签擅自用于重划原轮分层。失败/严格数值重算/补采继续用现有canonical服务。新增轨迹仅存一次。

## 7. 工作流、存储和运行

当前固定池五轮冷启动、逐轮refit、normalized validation-best不变。PEAKS/ERACS则独立连续训练，初始训练统计冻结，以免持续训练中模型坐标突变；默认200epoch，PEAKS前120epoch允许接收新样本，之后收尾。正式超参数仍需与对应随机对照固定并验证，不据real表现调参。

```text
utils/sampling/baseline/
  regression.py, peaks.py, eracs.py, low_rank.py, hggs.py
utils/sampling/dynamic/
  dataset.py       有界resident原始轨迹RAM缓存，重用共享变换
  callback.py      流式接收/训练batch或SNR触发重选
  runner.py       共用fit_round/评测/日志/恢复服务
  cli.py
utils/sampling/generation/hggs_initial.py  GF初始过滤/试训
```

只检查入口，不开始训练：

```bash
bash run_sampling_budding.sh dynamic --protocol /home/users/zsy/mydata/budding_yeast/benchmarks/budding_sampling/seed42_a866aef5d988 --pool mutation --method peaks_regression --dry-run
bash run_sampling_budding.sh dynamic --protocol /home/users/zsy/mydata/budding_yeast/benchmarks/budding_sampling/seed42_a866aef5d988 --pool lhs --method eracs_regression --dry-run
bash run_sampling_budding.sh generate --protocol /home/users/zsy/mydata/budding_yeast/benchmarks/budding_sampling/seed42_a866aef5d988 --pool mutation --method hggs_projected --dry-run
```

Low-Rank使用原`run --method low_rank`入口（或显式batch方法列表），当前没有把新方法加入默认正式批次。PEAKS/ERACS用dynamic入口，`--config`提供DynamicConfig JSON；HGGS用`--hggs-config`，生成LHS仍要求原已审核的`--lhs-config`。

```text
experiments/budding_dynamic/<protocol>_dynamic<hash>/<pool>/<method>/seed_42/
  config.json, train.log, performance.csv, loss.png
  stream/model.ckpt, history.csv, loss.png, dataset.json, best_exposure.json
  stream/eval/                    公共评测JSON/CSV/real图
  eval/                           最终公共报告引用/汇总
  _state/                         恢复状态、last、标签访问及动态ID清单
```

dynamic仅保存source引用、ID、必要权重/RNG和checkpoint，**不重复写轨迹**。HGGS沿用独立`budding_generation/<protocol>_gen<hash>/...`，原coarse/pilot引用source，新增ODE轨迹分块仅一次。Low-Rank沿用固定池结果布局。源代码身份变更不会覆盖旧结果；不要删一致性检查后在旧目录硬续跑。

PEAKS、ERACS、Low-Rank可以访问训练侧候选真值，实际访问可能达到整个训练侧池；这不是只查询最终10000标签的AL。禁止访问同/跨来源holdout或real来选样。ERACS动态替换的历史训练曝光可能超过最终驻留量；所有这些费用均单独记录。

## 8. 验证及资源边界

验证包括真实Relative梯度与autograd/有限差分、PEAKS cache/tie/刷新、谱协方差SNR、OMP加权目标、Low-Rank全敏感度公式与无偏权重、HGGS GF/MGS和RNG；以及小FNO的选择→训练→评测、两池动态运行、评测及训练周期恢复、HGGS真实canonical ODE查询、旧方法回归。

完成记录：联合新增/既有方法48项通过（170.59秒），原固定池/兼容接口6项通过（44.02秒）；最后补齐PEAKS初始训练阶段并增加两种随机流程对照后，17项专项再次通过（60.37秒，包含此前重叠用例，不把它们相加当独立总数）。没有把小模型结果用于选择real最优超参数。82个冻结生物及原始生成文件SHA变化数0，35个相关模块编译、三个真实shell入口dry-run、bash语法与diff检查通过；参考仓库运行时import检查通过。

日志和汇总位于`scratch/probes/baseline_implementation_20261117/`：`additional_and_existing_regression.log`、`additional_legacy_regression.log`、`additional_bootstrap_final.log`、`regression_baseline_verification.json`。新增用例为`tests/test_additional_regression_baselines.py`。

默认测试CPU单线程、CUDA不可见，未启动正式GPU实验。resident缓存有界、读轨迹/候选评分受公共资源槽限制，KNN/梯度均分块；没有新建90k²矩阵、没有递归进程启动。仍不能拿旧固定池16并发的资源结论直接套用新动态或生成方法，正式前需单项规模资源核查。
