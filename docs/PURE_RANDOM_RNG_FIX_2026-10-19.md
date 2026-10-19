# Pure-random 单随机流修复与 corr 风险说明（2026-10-19）

## 1. 本轮范围

用户授权修复 pure-random 主生成器调用 mutant_aware 时的随机数链路；corr 重要性先解释。
没有运行正式训练、HPO、real-mutant 模型重评或正式数据再生成，没有覆盖旧 NPZ、权重或指标。
重复样本切分、real 过滤/救援规则、ODE/事件、图软注意力和归一化公式均未改变。

## 2. 修复机制与兼容性

`generate_pure_random_dataset.py` 创建的 `rng=np.random.RandomState(seed)` 现显式传到：

- `sample_scaling_factor(..., rng=rng)`：幅度采样，包括 Uniform、Normal、Beta、离散 MC。
- `apply_gene_mutation(..., rng=rng)`：CDC14-ts、APC-A、TAB6-1、MDT、ORI/BUD/SPN 扰动等内部随机分支。
- `_resolve_conflicts(..., rng=rng)` → `_sample_mode(..., rng=rng)`：参数冲突时重抽模式。
- `rescue_collision_candidate(..., rng=rng)`：连续模式切换/碳源漂移及嵌套幅度、等位基因采样。
- 主函数本身的基因、初始模式、复合模式、KO 区间和碳源抽样继续使用同一个 rng。

没有在 helper 内反复重置 seed，也没有通过设置全局 seed 掩盖局部/全局混用。
新 API 兼容 `RandomState` 和 `Generator`；主入口仍使用 RandomState，不额外换随机数算法。
辅助函数省略 rng 时保留历史全局行为，仅为兼容未迁移调用方。
其他生成器（包括独立的 1M/Sobol 候选池）没有因此自动完成 RNG 迁移，不作全仓库复现保证。

另一个隐藏来源是冲突检查遍历 Python set：同一 seed 在不同 `PYTHONHASHSEED` 下可能先处理不同冲突。
显式 RNG 路径固定为 `sorted(touch_set)`；省略 RNG 的旧路径不变。
原幅度区间、模式权重、复合概率、70/30 救援规则与冲突重抽/丢弃规则均保留。
多个冲突同时出现时，显式路径采用确定的键顺序，这替代了旧的环境相关顺序；不保证重现任意旧进程的样本序列。

新导出的 NPZ 额外记录 `sampling_seed` 和 `sampling_rng_contract=pure_random_single_stream_v1`，
已有 data/params/param_names/mutant_names/labels/pattern_labels 的含义不变。
相同 seed 的新/旧代码会生成不同候选（随机流消费顺序已改变），不能把本次修改当作旧 NPZ 的精确重建。
未来生成请使用新输出文件名保留历史版本。

## 3. 验证

新增 `tests/test_pure_random_rng.py`：

- 两种 NumPy RNG、硬 KO/区间 KO：固定局部 seed，改变/消耗全局状态后候选完全相同；换局部 seed 后不同。
- 遍历全部 84 个 gene/mode 单元及额外随机扰动分支；禁止全局 np.random 抽样，确保没有漏接。
- 冲突重抽的四种已有 MODE_FREQ 配置、救援的连续/碳源分支均传递同一个对象。
- 三种 PYTHONHASHSEED 独立进程，分别验证两种 KO 设定的候选/冲突结果哈希相同。
- 使用人工 WT 基准条目和即时成功的假 executor，不积分真实 mutant，验证生成器传参、救援、
  顺序打包、各数组和 RNG 元数据的可复现性。

专项测试 **18 passed，26.55 秒**。另以修改前源码函数为参考，逐一比较 84 个 gene/mode、
四种模式权重设定、40 次冲突重抽/救援：省略 rng 的结果和最终全局随机状态均与修改前一致。
源码比较确认 ma 的行为改动仅在上述五个采样函数，ODE/事件和生物参数常量未改。

这里保证的是候选参数链路和人工固定成功条件下的打包结果。
实际 ODE 导出仍可能受到求解失败、墙钟超时、依赖版本和硬件的影响；不能把 seed 修复等同于任意环境下 NPZ 字节完全一致。
更不意味着已有模型的 MAE/MSE 已改善。

合并此前运行/图/IC 回归后的结果：**403 passed，15 个预期的历史迁移警告，54.29 秒**。

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q -p no:cacheprovider \
  tests/test_pure_random_rng.py tests/test_runtime_contract_repairs.py \
  tests/test_relation_module_repair.py tests/test_graph_model_migration.py \
  tests/test_yeast_graph.py tests/test_ic_wt_contracts.py \
  tests/test_mechanistic_bifurcate_pino.py
```

## 4. corr 为什么重要，但不等于坏例根因

当前代码的两个事实都保留：目标恒定的通道 Pearson 未定义但返回 0；三维输入按变量将所有
mutant×时间点拼接计算 pooled corr，而不是先逐 mutant 计算再平均。

本轮只读同一历史运行：

`experiments/mutant_aware_pure_random_100000_210m_500s/biom_refiner_plus/hybrid_mse_rel_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/`

- config.json 的 norm_type=log1p。
- eval/global_metrics_denorm.json：real count=126，all_39_vars pooled corr=0.9722762703895569。
- eval/per_mutant_metrics.csv：126 行，Correlation 算术平均=0.846338657159654，中位数=0.8905211985111237。

这不是新的预测结果，也不说明所有差异都来自坏例：恒定通道约定、样本间浓度差异和动力学差异均可能参与。
但它实证说明不能把 pooled 0.9723 写成“126 条曲线的平均 corr 是 0.9723”。

对当前默认路径的影响：

- HybridMSERelLoss.forward 实际优化 MSE + lambda_rel*RelL2，不包含 corr；不能只根据其文档措辞推断。
- 自动选 checkpoint 的 val_loss_synth 来自归一化 MSE，不使用 corr。
- tune.py 返回 real 物理 MAE，corr 只记录/展示，不作为当前 Optuna 目标。
- 因此当前 corr 口径不会直接制造上述训练梯度/选权/HPO 错误；它是评估解释和坏例诊断的重要风险，
  而不是把 MAE/MSE 预测变差的充分原因。

后续可在保留历史 pooled 指标的同时，显式新增逐 mutant 汇总、动态通道有效相关系数及有效数量、
恒定通道的 MAE/MSE；定义近恒定阈值和无有效通道时的处理后再实施。
本轮只解释，没有改 metrics.py、指标文件或历史榜单数值。
