# Response V2：明确动作预算及同路径随机对照

这是针对旧原型实际响应覆盖不足的修正，保留旧方法名/结果；不修改ODE、mutation模式、合法强度分布、mdt或初值处理。代码在`utils/sampling/ours/response_v2/`，路径编译在`utils/sampling/generation/path_bank.py`，随机选点核在`utils/sampling/baseline/path_random.py`。

## 两个方法

- `ours_response_v2_generate`：用同一背景、同一模块干预方向上至少两个不同已查询输入拟合局部响应，再按当前模型预测与该估计的完整时间Relative差异选样。
- `path_random_generate`：相同路径生成、支撑建设、探索预算和路径内批次限制；在可选响应点中均匀随机选点。不是新论文baseline，是本课题的消融对照。

生成和选择分别使用独立可恢复随机流。二者使用相同主seed的共同初始训练集；初始候选流相同，后续候选因各自已查询历史而变化，不能声称全程候选逐点相同。

## 查询动作

1. **explore**：累计按20%分配新背景查询，整数余数跨批次结转，不与响应分数排序。
2. **support**：单锚点路径显式补充真实标签，不能赋予伪造误差分数。已有足够响应支撑时通常每批至多补一条新路径；冷启动允许更多。
3. **response/path_random**：必须有至少两个不同的已查询路径输入。分数用公共归一化及公共RelativeL2Loss，完整时间轴、逐变量、eps0.01、无截断；是未知真值的局部估计误差，不是认证误差界。
4. **capacity_fallback**：合法容量不足时明确记录补位，不能冒充响应收益。小规模验证将此比例作为检查项。

未知背景没有数值误差分数（保存为null），所以不存在旧版固定1混排。每次ODE返回后立即把有效标签加入路径索引，无需等下一轮模型训练；ODE失败不形成支撑，释放其动作预算并沿用公共生成器补足有效样本数。

## 同背景合法性

路径键包含模块/模式/环境、非干预参数及固定初值/事件开关。新物理输入只由现有`sample_scaling_factor`和`apply_gene_mutation`重抽、再调用原初值准备；不插值物理参数，不构造四角。重叠参数足迹不独立编辑。固定原子不能被伪造为连续强度；路径规划以确定性的原生规则探测识别可变轴，固定模式仍可通过原生全局探索出现。

优先重用已有多点路径，并在同一候选bank上生成一组路径内点。避免每个候选都来自不同背景、无法形成真实响应支撑。两方法的路径规划不使用模型误差权重，便于控制比较。

## 查询前任务诊断

保存原始归一化模型预测及该次查询的归一化统计快照。真值返回后记录：

- normalized_relative_l2：与训练相同的公共公式/eps/完整时间轴，使用查询前那一轮统计。
- physical_relative_l2：共享公式在原始物理空间的诊断。
- 动作、路径ID、当轮模型阶段、估计分数（有支撑才定义）。

归一化每轮训练仍按累计训练数据重新拟合，验证best仍为归一化合成验证Relative。反馈不使用更新后的模型冒充事前预测，不跨轮直接比较归一化loss的绝对值。当前任务诊断仅接受既定rel_l2/eps0.01协议；其他loss/eps明确拒绝，避免静默不一致。

## 验证和产物

`tests/test_response_v2.py`覆盖未知分数、即时支撑、动作预算、生成流对照、任务loss一致性、统计快照、失败预算退款、恢复后重建路径/选点一致，以及真实FNO+ODE两轮流程。

训练侧小试入口`PYTHONPATH=.. python scripts/analysis/probe_response_v2.py --help`，复用共同初始2000的既有FNO，按公共生成流程各查询128条，不重新训练、不评测real或holdout标签。仅用于检查分数有无信息和机制覆盖，不能当作新训练模型的正式终评。完整报告在外部`resource_profiles/response_v2_probe_20261122/report.json`。

验证后长跑入口：

```bash
bash run_sampling_budding.sh extensions --suite response_v2 \
  --seed 42 --seeds 42 123 1024 --gpus 0 1 2 3 \
  --directory /home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/response_v2_multiseed \
  --initial-jobs 6 --max-jobs 6 --jobs-per-gpu 2 \
  --io-concurrency 2 --mechanism-concurrency 4 --cache-chunks 128 --dry-run
```

该suite只含两个方法×三个seed，共6项，当前仅Budding mutation。完整训练沿用2000→4000→6000→8000→10000、100/100/100/100/200 epochs。预测图仅末轮生成，指标/loss每轮保留，W&B离线。结果位于统一`fno/mutation/<方法>/<profile>/seed_<主seed>/`，旧原型结果不覆盖。

小试分数相关性或困难样本识别成功，不保证最终泛化收益；需要与路径随机对照和既有baseline完成多seed比较。
