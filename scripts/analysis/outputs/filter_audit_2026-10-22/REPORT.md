# Budding 轨迹过滤审计（2026-10-22）

对象：`/data/zsy/budding_yeast/mutant_aware_100000_210m_500s.npz`。对全部 100126 × 39 × 500 原始物理轨迹重新流式扫描，包含 100000 合成与 126 real；并重放当前参数门控。仅分析已有样本，不修改求解器、生成器、AL 或源文件。

这是历史保存后的幸存样本，含旧模块和旧阶数，不能据此恢复生成时丢弃或超时的数量，也不能直接代表新候选池分布。P1–P5b 沿用用户提供审计表中的观察窗分类，不作分岔或稳态证明。

## 合成样本统计

- MASS 最大值：6.08210131387；MASS>10：0 条。
- 按当前 AL real-envelope 规则拒绝：11460 条。
- 按当前 nominal-flux 参数门控拒绝：1150 条。
- 两种规则的并集：12253 条。
- 最小原始值：-2.76605679579e-06。

| 轨迹组 | 总数 | 浓度门控拒绝 | 参数门控拒绝 | 两者并集 |
|---|---:|---:|---:|---:|
| P1 | 6711 | 236 | 164 | 377 |
| P2 | 29052 | 2141 | 316 | 2291 |
| P3 | 18389 | 3354 | 122 | 3447 |
| P4 | 6849 | 774 | 88 | 846 |
| P5a | 30341 | 3893 | 383 | 4177 |
| P5b | 8658 | 1062 | 77 | 1115 |

原始负值的逐阈值数量：

{"negative_below_minus_1e4": 0, "negative_below_0": 20332, "negative_below_1e-12": 17017, "negative_below_1e-08": 7988, "negative_below_1e-06": 14, "negative_below_0.0001": 0, "negative_below_0.001": 0, "negative_below_0.01": 0}

## 解释与建议

数值完整性、有限值、完成观察窗和经复算确认的严重负值应作为共同有效性检查。高幅值、观察窗内停滞与分裂失败本身不等于无效轨迹；应保留对应标签并记录幅值标记。MASS=10 和两倍 real 最大浓度是研究域先验，不是数学奇点或普适物理边界。该数据没有触发 MASS 上限，无法估计未保存候选的损失。

门控参考：AL4PDE 将采样输入分布与采集策略分开；failure-averse AL 单独建模可行域。建议各策略共享同一有效性和域配置，避免将不同可行域效果误归因给采集算法。负浓度应在任何 log/clipping 前审计，小负值与绝对容差、事件扰动比较；较大或持续负值先加严容差重算，再决定拒绝。

## 参考

- AL4PDE: https://arxiv.org/html/2408.01536v2#S4.SS1
- Failure-averse Active Learning: https://arxiv.org/abs/2110.14443
- SUNDIALS negative values: https://sundials.readthedocs.io/en/v7.2.1/cvode/Usage/index.html#advice-on-controlling-unphysical-negative-values
- SciPy solve_ivp tolerances: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

复现入口：`scripts/analysis/dataset_diagnosis/audit_trajectory_filters.py --dataset <NPZ> --features <原 samples.csv.gz> --output-dir <新目录>`。逐样本 `samples.csv.gz`、逐变量 `variables.csv`、`extrema.npz` 和 `summary.json` 同目录保存。
