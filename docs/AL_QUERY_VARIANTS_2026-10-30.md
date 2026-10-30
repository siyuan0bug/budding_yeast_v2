> **最新运行状态**：用户已要求全14项FNO切换full长程，正式批次已启动。默认为AL_SUITE=research、TRAINING_PROFILE=full；screen仍可显式选择。见[后台启动记录](AL_FULL_LAUNCH_2026-10-30.md)。

# 固定训练协议下的主动学习策略枚举

本轮接入七个可单独运行的研究变体，保留六个baseline和首版ours作对照。10月30日根据loss变化率，后续加入默认screen初筛与显式full长预算复核，见[新训练设置](AL_TRAINING_BUDGET_2026-10-30.md)。没有启动正式训练，没有修改候选池、初始集、mutation、ODE、损失或原始生成器。工作开始于10月29日，完成于10月30日。

## 1. 统一协议与入口

- FNO：width192、4 blocks、modes24、batch512、峰值LR1e-3。
- 当前默认full为100/100/100/200 epoch、warmup5/5/5/10；显式screen为50/50/50/100、warmup4/4/4/8。两档余弦下限均1e-5，同档位方法和辅助ensemble预算一致。
- 训练量10k→20k→40k→70k，按新增有效、唯一轨迹计预算；失败继续走现有同策略补采流程。
- 共用原 `gene_ranges` 百万pool、同一初始10k训练/1250验证/1250测试/126 real数据。每轮只在累计合成训练集重新拟合归一化。
- ensemble方法使用相同的2个独立成员；最终轮只训练评测主模型，保留现有做法。
- 新研究组对所有方法统一采用 `seed + 10007*round` 的主模型冷启动seed，避免Power等选样操作消耗全局随机数后改变下一轮初始化。每轮保存 `training_seed.json`；辅助成员沿用独立seed规则。它不保证GPU运算逐位确定，也不消除不同训练集的梯度差异。
- 直接运行脚本默认是research十四项独立实验；`AL_SUITE=original`仅保留原七项。新组的结果路径附加 `research_v1`，避免覆盖已有结果。旧入口默认保留legacy RNG规则，新组固定每轮seed，因此已有旧结果只作为历史参考，研究组内的baseline会按相同seed协议重跑。

先检查计划：

```bash
cd budding_yeast_v2
AL_SUITE=research DRY_RUN=1 bash run_al_budding.sh
```

用户自行启动正式训练：

```bash
cd budding_yeast_v2
AL_SUITE=research SKIP_PREPARE=1 MAX_JOBS_PER_GPU=2 bash run_al_budding.sh
```

只跑新变体（仍建议在完整比较时采用研究组相同seed协议下的baseline）：

```bash
AL_SUITE=research SKIP_PREPARE=1 \
STRATEGIES="ours_response_marginal ours_response_feedback ours_interaction ours_invariance ours_event ours_error ours_coverage" \
bash run_al_budding.sh
```

GPU仍为0/1/2/3，各卡最多同时2项。14项当前分配为4/4/4/2；失败记录并继续后续任务，沿用现有调度规则。W&B强制offline。

结果位置：

`/home/users/zsy/mydata/budding_yeast/experiments/al_budding/fno/hybrid_mse_rel_<strategy>/w192_n4_m24_b512_e100-200_lr1e-3_lmin1e-5_wm5-10_al4_refit_gene_ranges_research_v1_full/seed_42/`

## 2. 七种新变体究竟有什么区别

这些是候选算法，不是已验证有效的新方法；本轮没有新增精度指标。

|策略名|实现机制|查询后反馈|
|---|---|---|
|`ours_response_marginal`|修正首版静态计分；单点与响应预算分开；响应方向递减收益|无，用于隔离批次目标修正|
|`ours_response_feedback`|同上，再用历史真实响应误差调整响应边的优先级|有|
|`ours_interaction`|比较MC剂量响应在有/无一个额外KO背景下的差异|无，独立检验四点查询|
|`ours_invariance`|查询违反已核实TEM1条件不变性的预测|无，检验发现共同偏差的能力|
|`ours_event`|按成员对观察窗口内分裂/时刻/关键峰值的分歧选样|无|
|`ours_error`|用先前查询前的预测误差，学习预测未查询输入的误差|有|
|`ours_coverage`|根据反馈分数，在模块集合、模式背景和强度区域逐级分配预算|有|

另外六个baseline：Random、Uncertainty、Power、MaxDist、LCMD、BAIT。首版 `ours_response` 的数学计分和调用方式保留。

### 边际响应计分的具体实现

按唯一新查询数分配约10%单点、80%响应、10%随机探索；预算小时采用确定的取整规则。已知训练端点和本批已选端点免费，不能再次贡献单点奖励。响应边只拿响应分歧奖励：

`utility(action | selected) = R(action) * novelty / [(1+CDF_distance) * new_endpoint_count]`

共享端点入选后，动态更新相邻动作的新端点数；不是仅更新cost而保留首版端点U奖励。已覆盖响应方向用16维固定CountSketch和正半定累积矩阵近似：

`novelty(v) = 1 / (1 + v^T C_background v)`

所有成员/通道/32个采样时刻都贡献CountSketch，不是只读取16个坐标。各背景独立累积方向；它是递减冗余的近似，不等于精确互信息。保留既有近邻及远端对照，远距离通过原CDF权重降低优先级，不禁止。

如果有效响应边不足、方差为零或剩余预算无法容纳整个动作，剩余额度明确标为 `point_fallback`，再保留随机探索；不会拆开四元组后仍声称查询了完整相互作用。

### 四点相互作用与不变性的边界

相互作用比较 `(d-c)-(b-a)` 的成员分歧，四个端点均已存在于pool或合成训练集。首版限于离散 `MC/GAL-MC` 剂量控制，以及增加/移除一个额外KO；其他相同输入精确匹配，不做连续强度的近邻凑组。重叠参数控制、事件开关变化或缺少端点时排除该块。它没有实现任意背景差异、任意连续突变的近似四点匹配。

TEM1不变性仅在Tem1T=0、TEM1初值=0、其他参数/初值/事件一致且只改变ka15pp时成立。分数来自成员平均预测轨迹的变化，因此成员一致预测了错误变化也能触发。当前ODE和事件源码带指纹检查；源码变化后必须重新审查规则，否则该策略报错停止，而不是继续套用旧证明。不能把有限精度求解伪影解释成真实响应。

不变性分数为物理log1p轨迹差，而其他响应分数为当前归一化轨迹的成员方差；每个策略内部排序，不跨策略比较这两个分数的数值大小。

### 事件策略

每个成员基于完整预测时间轴计算质量跌落>0.1的分裂候选，10步内重复去抖，提取观察窗口六分类、是否分裂、首次分裂时间及CLB2/ORI/SPN/ESP1幅值特征。所有特征映射到[0,1]后平均成员方差。无分裂时有单独标志，首次分裂时间置为窗口终点作为编码；不能把它解释成真实分裂时间。

事件来自预测轨迹。真实label只在查询后核验；没有调用real参考阈值上包络或旧ours的合成速率门控。观察窗口的分组不自动意味着真实极限环、分岔或生物学因果证明。

## 3. 反馈如何避免泄漏和跨轮尺度变化

1. 只使用此前已接受的合成查询结果，以及明确定义的合成训练锚点；不读取合成验证/测试或real真值来打分。
2. ODE调用前，记录入选端点的成员平均物理log1p预测，最多32个均匀采样时刻。采用现有物理解码器，包括其数值保护；不是重新定义训练归一化。
3. ODE返回后、数据追加及下轮重拟合前，按稳定candidate ID对齐成功样本。失败端点不产生反馈，配对不完整则不计算响应误差；已接受点仍可提供单点误差。不会用新一轮已经训练过这些样本的残差冒充未见样本误差。
4. 反馈目标在物理 `log1p(max(Y,0))` 空间计算MSE，不随每轮Z-score统计改变。原始负值轨迹继续按现有规则保存，该反馈变换与训练的log1p预处理一致，不额外拒绝样本。
5. 误差预测器使用物理参数和初值的固定逐维Z-score输入。统计只在首次剩余pool输入上拟合一次；它与主模型每轮train-only归一化独立。NPZ保存该输入统计以复现。
6. 默认ExtraTrees为32树、深度12、叶节点至少8样本、最多2线程；只在先前查询反馈上拟合。少于32条时显式回退不确定性。随机保留20%历史反馈报告校准误差，随后在全部历史反馈重拟合；该反馈保留部分仍有采集选择偏差，不是全pool无偏误差评估。
7. `ours_response_feedback` 用预测响应误差的百分位，将原响应方差乘以[1,2]的因子。`ours_error` 直接按预测单点误差排序。首轮没有历史反馈时分别退化到边际响应/单点不确定性，并保留10%随机探索。
8. `ours_coverage` 的三级分组为模块集合、完整模式组合+环境、每个可变控制的四分位CDF强度区域。分数为预测误差（首轮为U）除以三级已有样本计数乘积的平方根；每入选一个点实时更新计数。保留10%随机探索，不要求最终训练每组强行等额。

这些策略自身有明确的采集设置，均写入实现/审计说明，尚未经过超参优化；“训练超参保持一致”不等于不同采集函数没有内部设计参数。

## 4. 输出增加的审计文件

原有round checkpoint、完整数据快照、loss图、real预测图和反归一化metrics不变。

每个新变体在 `acquisitions/after_round_XX/` 增加：

- `query_design.json`：策略、动作、所有端点、未知端点计费、接受状态、回退、耗时及反馈拟合情况。
- 有反馈的三种策略另保存 `prequery_feedback.npz`：candidate IDs、固定输入特征及统计、查询前预测、实际轨迹的log1p采样值、单点/响应误差和响应观测数。
- `response_feedback.json`：成功完整配对的实测响应误差；负端点ID明确编码为 `-1-合成训练行号`，与real mutant编号无关。

补采时反馈仍按稳定ID对应；审计汇总各补采批次，已失败输入不被当作成功轨迹。文件使用临时文件后替换，避免半截NPZ/JSON。

## 5. 真实百万pool可用性检查

输入几何全量审计已完成：排除初始12,500个合成样本后，剩余987,500个候选；加入10,000个合成训练锚点。没有读取预测目标、重跑ODE或生成新样本。

|对照类型|合法动作数|涉及的剩余候选数|涉及的已知训练锚点数|
|---|---:|---:|---:|
|同背景响应|3,709,878|961,789|9,824|
|TEM1条件不变性|23,720|8,392|75|
|精确MC＋KO四点相互作用|123,198|38,248|381|

因此新策略有真实可查询对照；但不变性专用候选总数不足每轮全部预算，策略会如实回退，不能宣称70k全是该类查询。后续池缩小后动作数量也会变化。

产物：`/home/users/zsy/mydata/budding_yeast/analysis/al_query_variants_2026-10-29/`，包括源文件SHA256、稳定candidate ID四元组/边和计数。指纹：pool `9d7011138b58bd71b7a0438e0e210128bcd70a138ff518f9ae531e4ca6850d00`，initial `d7814986e1f94d2b6e5d9eda29a341d6461379e6991f29160b86281dd93a002a`。

复现：

```bash
PYTHONPATH=.. /home/users/zsy/.conda/envs/neuralop/bin/python scripts/analysis/audit_al_query_geometry.py \
  --pool /home/users/zsy/mydata/budding_yeast/budding_pool_1000000_seed42_gene_ranges.npz \
  --initial /home/users/zsy/mydata/budding_yeast/budding_initial_train10000_seed42_gene_ranges.npz \
  --output /home/users/zsy/mydata/budding_yeast/analysis/al_query_variants_2026-10-29
```

实现位于 `utils/active_learning/strategies/{experimental,query_design,query_feedback}.py`；注册与仿真仍由统一registry/engine处理。此前idea与收敛证据见[收敛与研究方案](AL_CONVERGENCE_AND_NEXT_IDEAS_2026-10-29.md)。

## 6. 本轮验证结果

最终135项回归全部通过（30.84秒）。包括新增20项定向测试、首版ours、registry/SelectionPlan、已有数据追加及snapshot、ensemble训练和RNG保护、dtype、离线W&B和批任务调度。检查特别覆盖：所有七个变体选择及审计落盘；零响应不能复用端点奖励；四元组/训练锚点的唯一计费；预查询预测不被查询后模型改变；失败/重排后的稳定ID反馈；已知合成锚点反馈；无成功样本不创建反馈；冷启动和拟合后的误差预测器。

运行了 `bash -n run_al_budding.sh`、`git diff --check`、实际 `train.py --help` 和14项/只选7项的dry-run。现有测试中有CPU小模型/ODE接口回归，不是新的正式FNO实验；本轮没有报告模型精度收益。14条warning来自已有测试中的Tensor拷贝建议、CPU测试未用GPU、loader worker建议及W&B弃用提示，没有失败项。

全量测试命令：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.. \
  /home/users/zsy/.conda/envs/neuralop/bin/python -m pytest \
  tests/test_al_query_variants.py tests/test_al_response.py tests/test_al_package.py \
  tests/test_al_experiment_workflow.py tests/test_al_batch_logging.py \
  tests/test_al_ensemble_uncertainty.py tests/test_al_pool_dtype.py \
  tests/test_response_analysis_tools.py -q
```

先比较统一协议下的结果，再对有收益的机制做消融和多seed复核。使用real badcase作事后诊断，不能将其真值、编号或专属邻域写入采集函数。
