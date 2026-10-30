# AL候选池配额设计与官方训练预算核对（2026-10-28）

> **2026-10-28 新配额已接入**：Budding AL采用模块集合等额、集合内合法模式容量感知均衡；新百万池为9815/126225/863960条L1/L2/L3，默认文件改为`budding_pool_1000000_seed42_balanced.npz`，初始集使用独立`balanced`路径。静态与Fission不变，本轮未启动训练。实现、有限容量与分布说明见 [两级均衡候选池](BALANCED_AL_POOL_2026-10-28.md)。

已采纳用户选择：**模块集合之间等额，集合内再均衡合法突变模式**；保留当前mutation类型、参数赋值和强度分布。非GAL环境已按用户明确回复改为80%/15%/5%，有GAL仍固定mdt150。

已实施AL环境分配；用户后续暂定采用40/40/40/100 epoch、warmup4/4/4/8，现已写入Budding批脚本的全部默认模型/策略行。已交付当前目录的组合枚举和容量感知逐集合配额。**新配额已接入生产，百万balanced池已生成；尚未生成新初始集或启动训练，Fission脚本未改。** 下文设计配额已由实际产物核对，覆盖结果见BALANCED_AL_POOL_COVERAGE_2026-10-28.md。

## 1. 选定的“公平”口径与实际配额

只有模块集合的等额，和所有“模块＋模式”组合全局等额，是两种不同目标。用户已选择前者；二级再按当前集合内合法模式均衡。包含已有合法复合模式，不新增/删除任何mutation类型。

根据catalog和TOUCH精确枚举：22个基因/功能模块（不含MDT）、70个基本模式单元、加入现有表达＋稳定性复合后80种单模块allele。排除跨模块参数冲突后：

| L | 合法模块集合数 | 含模式/复合的组合数 | 模块等额未限容量 | 模块等额容量截限后 |
|---|---|---|---|---|
| 1 | 22 | 80 | 12415 | 9815 |
| 2 | 230 | 2990 | 129797 | 126225 |
| 3 | 1520 | 69427 | 857788 | 863960 |

百万池的容量感知设计配额因此为 **9815 / 126225 / 863960**，约 **0.9815% / 12.6225% / 86.3960%**。这取代之前按固定5/40/55比例设计的思路，Budding大池不再使用POOL_RATIOS决定层配额；该常量仍用于初始集选点和Fission原规则。

未限容量时，1772个合法模块集合平均约564条；有限组合饱和后，多数集合分到571或572条，彼此相差至多1。配额使用固定seed的余数分配，不以real预测误差指定某个模块背景优先。

**二级分配：** 每个模块集合的配额，再按其合法allele组合等额分配；顺序无关的GAL+db与db+GAL只计一个组合。旧25%复合分支是随机混合规则；新配额已将合法复合模式作为显式单元分配，因此其最终占比可能变化。这是模式选择概率的设计变化，强度CDF仍为mutant-aware，mutation类型本身不变。

并不是所有模式单元都能填同样多的唯一输入。统一采用 `q_s=min(C_s,lambda)` 的容量感知均衡：达到容量上限的单元饱和，其余单元重新均分余额，并保留缺口/饱和记录。对模块集合一级已做这个计算；模式二级已按实际接受计数维护容量、跨名称物理等价和去重结果，不能直接把72,497个名字各生成同样多就宣称完全均匀。

## 2. 有限容量不是抽样故障

对没有连续强度的模块集合，穷举所有合法模式、MC的2–30整数拷贝和三个环境，再按实际参数＋初值去重、排除real碰撞，得到精确容量。当前共有25个有限容量的模块集合；其他集合至少有一个连续强度选项。

| 模块集合 | L | 合法模式组合数 | 扣除real后的不同输入容量 | 配额 |
|---|---|---|---|---|
| BCK2 | 1 | 2 | 88 | 88 |
| BCK2;BUB2 | 2 | 2 | 90 | 90 |
| BCK2;BUB2;CDH1 | 3 | 4 | 180 | 180 |
| BCK2;BUB2;MAD2 | 3 | 2 | 90 | 90 |
| BCK2;BUB2;SBF_MBF | 3 | 2 | 90 | 90 |
| BCK2;CDH1 | 2 | 4 | 180 | 180 |
| BCK2;CDH1;MAD2 | 3 | 4 | 180 | 180 |
| BCK2;CDH1;SBF_MBF | 3 | 4 | 180 | 180 |
| BCK2;MAD2 | 2 | 2 | 90 | 90 |
| BCK2;MAD2;SBF_MBF | 3 | 2 | 90 | 90 |
| BCK2;SBF_MBF | 2 | 2 | 90 | 90 |
| BUB2 | 1 | 1 | 2 | 2 |
| BUB2;CDH1 | 2 | 2 | 6 | 6 |
| BUB2;CDH1;MAD2 | 3 | 2 | 6 | 6 |
| BUB2;CDH1;SBF_MBF | 3 | 2 | 6 | 6 |
| BUB2;MAD2 | 2 | 1 | 2 | 2 |
| BUB2;MAD2;SBF_MBF | 3 | 1 | 3 | 3 |
| BUB2;SBF_MBF | 2 | 1 | 3 | 3 |
| CDH1 | 1 | 2 | 4 | 4 |
| CDH1;MAD2 | 2 | 2 | 6 | 6 |
| CDH1;MAD2;SBF_MBF | 3 | 2 | 6 | 6 |
| CDH1;SBF_MBF | 2 | 2 | 6 | 6 |
| MAD2 | 1 | 1 | 2 | 2 |
| MAD2;SBF_MBF | 2 | 1 | 3 | 3 |
| SBF_MBF | 1 | 1 | 3 | 3 |

例如BCK2的L1最大88条，正好解释旧百万池单BCK2模块只有88条的现象；SBF_MBF的L1只有3条、CDH1只有4条、MAD2/BUB2各2条。即使要求“公平”，也不应重复这些输入以填表面配额，或偷偷修改已固定的mutation定义。初始集仍需单独保证各L层足够候选，不能机械把大池比例套入初始训练比例；本轮没有修改初始集规则。

当前“合法”按共享TOUCH冲突政策定义，SBF_MBF＋CLN2没有可保留模式组合；所以L2不是231而是230个模块集合，L3不是1540而是1520个。此次不改变这个冲突定义。APC-A等跨名称等效控制仍需生产去重；枚举表不声称每个符号组合对应互不相交的物理参数域。

## 3. 环境比例：已实施80/15/5

用户最初写90/15/5，总和110%；经回复确认采用 **80/15/5**。

修改 `data/systems/budding/samplers/sobol.py` 中sample_pure_random_candidate的非GAL分支：

- u<0.80：mdt90。
- 0.80≤u<0.95：mdt150。
- 0.95≤u：mdt160。
- 有GAL：固定mdt150。

作用于AL预生成池与调用相同budding_pool配方的运行时兜底池；没有改静态Gibbs-Sobol、其他静态策略、Fission或data根目录历史代码。比例是非GAL环境提议分配，有限容量、real排除、全池去重后未必在每个小单元里精确达到80/15/5。已有NPZ不会被自动改写。

边界0.80/0.95、1000个等距概率坐标得到800/150/50，以及GAL强制150均已回归验证。

## 4. Sobol与mutant-aware分别做什么

mutant-aware规定强度的目标分布F；Sobol提供[0,1]中的低差异坐标u；最终使用 **alpha=F^{-1}(u)**。

例如GAL-CLN2在0.06–0.30间对数均匀：

`alpha = exp(log(0.06) + u*(log(0.30)-log(0.06)))`。

u=0.5对应sqrt(0.06×0.30)≈0.1342，不是线性区间中点0.18。MC用离散2–30、权重1/N的累积分布决定拷贝数；Sobol并没有把它改成等概率整数或线性均匀强度。

当前多worker实现每256候选块独立派生seed、扰乱Sobol，再按候选的活动突变取坐标。它提升的是概率坐标覆盖，不能保证每个稀有基因背景都有足量样本；不同背景混用候选流，也不保证每个背景内部单独具有Sobol覆盖。新配额实现已给每个模块/模式单元独立可复现的概率坐标子流，再通过同一CDF映射，保留mutation的强度定义。

### 4.1 用户追问：先分配组合，再在组合内部生成强度

新方案先枚举合法模块集合并分配配额，再将每个集合的配额分给其合法模式组合（包含现有复合模式），最后对每个模式组合生成Sobol概率坐标，并逐维按原mutant-aware逆CDF映射成强度。例如先固定“CLN3[KO]+BCK2[KO]+CLN2[MC]”，再抽取CLN2拷贝数；不是先随机抽一个所有基因通用的强度，再让Sobol决定它属于什么模式。

这意味着离散模板及计划条数尽量事先确定；连续强度、离散MC拷贝数和适用环境仍需采样/分配。固定KO等没有强度自由度，必须枚举其有限输入容量。最终候选还要通过real碰撞、全局去重等检查；救援若改变模式，必须按实际模式重新记账并在原单元补采，不能把计划条数视为最终已接受条数。候选顺序可按seed打乱，Sobol也可按seed扰乱，因此预分配不等于取消所有随机性。

Sobol只负责概率坐标覆盖，均衡模块/模式由配额负责；顺序是“模块集合→模式组合→该组合内概率坐标→原分布逆CDF”。每个组合内部做联合坐标覆盖，不能保证每个投影在有限条数下严格均匀，也不意味着MC的2–30拷贝等概率。当前Budding AL生产器已使用以上流程，旧随机原语不再决定正式池。

## 5. 恢复模块是否“回到了mutant-aware分布”

应区分两种概率：

- **强度分布：是。** 当前mutant-aware与其Sobol逆CDF对恢复的PPX-GAL、PPX-ts/db及Mcm1-ts等使用对应MA规则。PPX-GAL为LogUniform(0.2,0.6)，PPX-ts/Mcm1-ts使用0.2×Beta(0.5,2)，PPX-db为Uniform(0,0.01)。
- **选择哪种模式：由采样器另行决定。** 原MA默认class权重与AL的模块内均匀不是一个选择器。旧AL初选曾为PPX四种各25%、Mcm1两种各50%；新池按完整合法模式组合分配，不能继续引用该初选概率。静态class下PPX约29.79/29.79/19.15/21.28，Mcm1为KO60.87%/ts39.13%。

原始mutant_aware表中SBF_MBF还包含CA；当前用户确定的类型集合只保留KO，故当前选择KO为100%。此次不以“恢复分布”为由将CA加回来。新二级配额也不强行恢复旧class/S5比例，因为用户已选择集合内模式均衡。

## 6. AL4PDE本地官方代码的实际训练设置

以下是当前服务器副本的源码/配置结论，不把默认文件与实验override混成一份配置。

| 配置/流程 | 训练预算与模型设置 |
|---|---|
| config/main.yaml + run_active_learning.py | num_epochs500、num_al_iter5；每个AL轮次使用相同500epoch，不是20/100；每轮重新初始化模型/优化器/调度器 |
| config/experiment/burgers.yaml、ks.yaml等1D实验 | 通常500epoch/轮、batch512；arena_orig采用UNet1D基座，hidden16、ch_mults[1,2,2,4]、Adam lr1e-3 |
| prob_model/model_wrapper/fno.yaml默认 | FNO width64、modes20、Adam1e-3；base batch64，具体实验可覆盖 |
| 2dcfd_fno / 3dcfd_fno | FNO width32、modes沿用20、batch16/10、500epoch、4轮 |
| 2dcfd_sinenet | 400epoch、batch32、4轮 |
| sub_exp/topk / power | 同任务同模型训练设置，ensemble改为2；Random通常1个模型 |
| main_time.yaml + run_al_time.py | **评测模型500epoch、选样模型30epoch**；是两个模型，不是前几轮30、最后500 |
| experiment/burgers_time.yaml | 评测UNet batch512；选样FNO ensemble2、width20、batch1024，预测批32768 |

AL4PDE默认cosine调度warmup0、eta_min1e-5；部分模型配置用StepLR(step80,gamma0.5)。官方不是所有模型机械共用相同width/batch，也没有对每种AL策略独立任意更改主模型容量；同一任务/模型下主要切换选样器及其所需ensemble。

另外pool_random/top_k/power/max_dist等配置的pool_size通常100000。当前项目100万全池打分的工作量不能忽略；官方数值只是参考，不代表必须把当前池缩小，也不能把BAIT此前的长运行窗口全部归咎于epoch。

源码位置：`al4pde/config/main.yaml`、`main_time.yaml`、`config/experiment/*.yaml`、`scripts/run_active_learning.py`、`run_al_time.py`、`al4pde/models/wrapper.py`、`al4pde/prob_models/ensemble.py`。

## 7. Tau_BNO本地官方代码的实际设置

Tau_BNO是静态训练基准，没有当前项目的AL逐轮训练协议。

- args_parse.py命令行默认epoch50、batch20、lr8e-4、min_lr2e-7。
- **实际批脚本覆盖为epoch1000、batch1**；大多数模型lr8e-4，GINO1e-3，NN/KAN2e-4；种子53/25/81。
- train_scripts.sh进一步指定min_lr2e-6、weight_decay0、aux_loss_lambda0。
- train_fno.py统一使用AdamW和CosineAnnealingLR(T_max=epoch)，没有该入口的warmup。

| 模型配置 | 容量 |
|---|---|
| FNO | hidden64、modes16、4层 |
| TauBNO | FNO通道64、modes16、FNO2层及GNN2层等组合结构 |
| DeepONet | width256、branch/trunk各3层 |
| Transformer | d_model256、4层、4头、FF384、dropout0.1 |

因此“正常必须20/100”不是这两个仓库的默认。官方更值得参考的是：同模型跨方法保持训练协议一致、区分选样与评测成本、按架构定义容量，而不是把不同模型的width都写成相同数值。其数据规模、输入维度与本项目不同，epoch和batch不能直接当成等价计算预算。

## 8. 本项目更快、可对齐的训练方案（用户暂定，Budding launcher已更新）

之前轻模型从batch512提高到1024时，epoch同步翻倍到100/200，是为了近似保留优化器更新次数。现在若优先节省实验时间，可以明确降低训练预算，而非继续维持旧步数。

**用户暂定采用40/40/40/100 epoch、warmup4/4/4/8，已更新run_al_budding.sh的16条默认实验行及注释中的可选biom行。** 20/20/20/100、warmup2/2/2/8仅保留为未来快速筛选备选，没有作为当前默认。 所有AL策略对同一模型应用相同方案，保持冷启动与每轮累计训练归一化。只缩epoch不应单独把某个策略的ensemble成员数从2变1。

推荐保留已选模型容量/峰值LR以避免同时引入容量变化：FNO192/m24/n4/b1024/lr1e-3；DeepONet256/n3/b1024/lr1e-3；UNet64/n4/b1024/lr1e-3；Transformer192/n4/b512/lr5e-4。跨模型batch不完全相同应记录更新次数；同FNO家族可统一通道语义及规模，同模型跨AL必须相同。若后续将Transformer也统一batch1024，应先确认其显存可行性，本轮没有做GPU训练探针。

当前DataLoader的drop_last=True，所以对1万/2万/4万/7万训练量、batch1024，每轮每epoch更新数为9/19/39/68。预算对比：

| 训练epoch | batch | 主模型更新总数 | 相对原协议 |
|---|---|---|---|
| (100, 100, 100, 200) | 1024 | 20300 | 1 |
| (40, 40, 40, 100) | 1024 | 9480 | 0.466995 |
| (20, 20, 20, 100) | 1024 | 8140 | 0.400985 |

40/100方案主模型更新数约为原46.7%，20/100约40.1%；总墙钟不会等比例下降，ODE、候选打分、评测和保存仍有成本。此表不含额外ensemble训练的更新数。

从已有Random首轮synthetic验证曲线看，20epoch时仍在改进：

| 模型 | 旧20epoch时val | 旧40epoch时val | 旧100epoch时val |
|---|---|---|---|
| deeponet | 0.332385 | 0.274397 | 0.246238 |
| fno | 0.332011 | 0.231604 | 0.176878 |
| unet | 0.47036 | 0.357527 | 0.25827 |

这些是旧100epoch余弦调度下的截面，不能当成重训20epoch短调度的精度结果。它们仅支持不要预先保证20epoch已充分收敛，建议正式对比保留40epoch作为折中；选择epoch应看synthetic验证集，不能用real测试排名逐策略调参。

## 9. 交付与状态

- 已实施：AL pool的非GAL环境80/15/5；GAL仍mdt150。
- 已计算：1772个合法模块集合、72497个含模式组合；容量感知的逐模块集合配额及25个有限容量集合的精确容量。
- 已验证：环境边界、1000点计数、GAL强制环境、组合去冲突/复合规范化、容量分配；相关回归见regression.log。
- 已生产实现：按模块集合/模式配额生成候选、按单元分配Sobol子流、容量/别名/real救援后的补采与报告，以及产物版本和旧缓存身份检查；新百万池已完成。
- 已实施用户暂定训练预算：run_al_budding.sh全部默认实验40/40/40/100 epoch、warmup4/4/4/8；模型容量、batch和峰值LR不变，Fission不变。未重启原批实验。

完整输出：`/home/users/zsy/mydata/budding_yeast/analysis/al_pool_redesign_2026-10-28`；`capacity_aware_module_quotas.csv`是选定口径的每集合设计配额，`legal_allele_combinations.csv`是完整枚举，`quota_options.csv`保留两种未限容量口径的对照。`plan_budding_pool_quotas.py`只计算计划，不生成正式候选。

后续训练配置验证：`bash -n run_al_budding.sh`通过；`DRY_RUN=1`确认16条实验均使用40/100 epoch；`tests/test_al_batch_logging.py`全部16项通过，覆盖实际训练命令中的epoch/warmup传递、模型配置、队列错误处理和W&B离线行为。该验证未执行模型训练或正式数据生成。
