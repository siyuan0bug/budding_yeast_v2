# 新版数据生成：文件与调用速查

当前生产入口：`python data/generate.py`。31 个旧生成/LHS 文件原地原样留存，
生产导入不依赖它们；显式新旧对照审计仍可以导入旧版。旧 NPZ 无需转换。

## 文件名

使用 `--data_dir DIR` 自动命名：`<recipe>_<N>_seed<seed>_modular.npz`。
`N` 是请求的合成样本数，文件名不含 real 数量；求解失败/过滤后实际数可能较少。
使用 `--output FILE.npz` 可自行指定文件名，默认拒绝覆盖。

| 旧生成文件 | 新 recipe | 默认 N / seed |
| --- | --- | --- |
| generate_pure_random_dataset.py | budding_random | 100000 / 42 |
| 同上，KO interval | budding_random_ko_interval | 100000 / 42 |
| mass_random_sampler.py | budding_mass_random | 100000 / 42 |
| mass_random_mutant_dist.py | budding_mass_random_dist | 100000 / 42 |
| mass_dataset_generator.py | budding_mass | 100000 / 42 |
| mutant_aware.py | budding_mutant_aware | 100000 / 42 |
| gibbs_boltzmann_sampler.py | budding_gibbs | 100000 / 42 |
| gibbs_sobol_sampler.py | budding_gibbs_sobol | 100000 / 42 |
| generate_kdpp_gated_dataset.py | budding_kdpp | 100000 / 42 |
| generate_kdpp_curv_gated_dataset.py | budding_kdpp_curvature | 100000 / 42 |
| generate_motif_topology_dataset.py | budding_motif | 100000 / 42 |
| bams_dataset_generator.py | budding_bams | 100000 / 42 |
| mass_wt_blind_sampler.py | budding_wt_blind | 100000 / 42 |
| generate_1m_pure_random_pool.py | budding_pool | 1000000 / 42 |
| generate_distribution_ablation_twins.py | budding_twins | 100000 / 42 |
| mirror_pure_random_ko_interval.py | budding_ko_mirror | 继承源文件行数 / 42 |
| generate_bams_ablation_datasets.py | budding_bams_ablation | 50000 / 20260915 |
| fission_random_sampler.py | fission_random | 1000 / 20000 |

特殊输出：Twins 第二份在主文件名后加 `_mass`；BAMS 消融额外加 `_hard`、`_hypo`。
KO mirror 必须传 `--source`，自动文件名为 `<源文件stem>_ko_interval_seed<seed>_modular.npz`；
它修改源文件对应行，不使用 `--samples` 重新抽取 N 行。BAMS 消融也必须传 `--source`。

```bash
python data/generate.py --list
python data/generate.py --recipe budding_kdpp --data_dir /path/to/data --workers 4
python data/generate.py --recipe fission_random --samples 1000 --data_dir /path/to/data
```

## 文件内容与旧版的关系

轨迹文件保留 `data, params, param_names, mutant_names, labels, pattern_labels`。
Budding 为 `(样本数,39,500)` / `(样本数,141)`，时间 0–210 分钟；
Fission 为 `(样本数,16,500)` / `(样本数,58)`，时间 0–300 分钟。
默认 real 在前，Budding 126 个、Fission 20 个（含 WT）；`--no-real` 不包含 real。
普通生成入口若 real 求解不完整会报错，避免静默导出残缺测试集。
合成样本求解失败、质量过滤后不一定补足 N；实际数量看 `is_real` 和 `generation_report`。

新增 `y_init,is_real,time_grid,system_id,sample_id,pair_id,overrides_json` 等溯源字段。
参数池只有参数/初值/元数据和 `p_mean,p_std`，没有伪造的轨迹和标签，不能直接训练。
不能宣称所有新旧文件逐样本一致：MASS 生物初值/real 按要求统一到 MA；
碰撞统一救援、状态重置、BAMS 工作进程随机数、配对失败处理等也有明确差异。
历史重复调用状态不会被复刻；本轮没有生成十万/百万规模文件进行逐行比对。

## 消费接口

公共生物接口：`budding_yeast_v2.data.systems.budding.api`。
`data/__init__.py`、训练/评测、AL、HANO、物理参数及生产分析脚本已切换到新模块。
`yeast_graph.py` 默认读取新 ODE/事件源码；`yeast_relations.py` 的 226 条非自环边不变。
DataModule 旧的五项加载返回接口不变，优先使用 `is_real/time_grid`，继续读取旧格式。
Fission 的 DataModule、独立16维图、通用模型训练/评测及可微物理残差已接入，详见
[Fission扩展](FISSION_EXTENSION.md)。Budding专用生物架构与AL策略仍需独立移植。
批处理标准数据路径已改为新文件名；历史特殊分布实验路径保留供显式选择。
更换生成器不改写现有 NPZ，也不允许旧 checkpoint 静默套用新数据的归一化统计量。
