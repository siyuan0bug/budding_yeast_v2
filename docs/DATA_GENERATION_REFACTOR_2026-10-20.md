# 数据集生成代码整理：使用和维护说明

本次增加模块化生成入口 `data/generate.py`，同时完整保留原有脚本。旧命令继续执行旧代码；新命令执行拆分后的实现。没有覆盖正式 NPZ，也没有更改训练、归一化或数据切分代码。

> **2026-10-21 最终目录落地**：新框架统一在 `data/systems/`。
> 公共配置/协议为 `contracts.py`，注册和查配置为 `registry.py`，生成与CLI解析为 `generation.py`，
> 校验导出为 `dataset_io.py`；Budding/Fission 各自的 `recipes.py`、`adapter.py`、`samplers/` 管系统专属规则。
> `data/generate.py` 只保留启动转发。原 `data/sampling/`、`data/recipes/`、`data/generation/` 目录已移除，
> 旧Python导入由 `systems/compat.py` 惰性映射到同一个真实模块，兼容Sampler、状态赋值与旧pickle路径。
> 31个原始生成器原地保留；归档哈希清单现位于 `systems/archived_generators.json`。
> 同时修复Fission增量去重、混合格式seed汇总及Fission采样审计接口，命令和默认配置保持不变。

后续按要求将两个 real 定义模块统一命名为 `real_mutants.py`。超时、配对、派生数据校验等最终复查修正见 [最终复查说明](DATA_GENERATION_FINAL_REVIEW_2026-10-20.md)。

## 文件分别负责什么

一次生成按以下顺序进行：选配置 → 抽取基因和突变模式 → 修改参数 → 碰撞救援 → ODE 仿真 → 表型判定 → 保存 NPZ。

| 位置 | 内容与修改时机 |
| --- | --- |
| `data/generate.py` | 稳定命令入口，只转发到共享生成模块 |
| `systems/contracts.py` | SystemSpec、候选、结果、运行配置、Recipe及adapter协议 |
| `systems/registry.py` | 唯一系统注册、recipe查找、默认配置解析和sampler创建 |
| `systems/generation.py` | CLI参数解析、配额、候选救援调度、并行仿真、结果收集 |
| `systems/dataset_io.py` | NPZ校验、元数据与原子导出 |
| `systems/compat.py` | 旧框架导入兼容，不复制实现或模块状态 |
| `systems/{budding,fission}/recipes.py` | 各系统的数据集组合、配额、默认样本数与seed |
| `systems/{budding,fission}/adapter.py` | 各系统的sampler组装、上下文、碰撞及结果过滤 |
| `systems/{budding,fission}/samplers/` | 各自的选模块算法与救援；Budding含Random/Gibbs/Sobol/k-DPP/Motif/BAMS等 |
| `systems/{budding,fission}/states.py`、`parameters.py`、`ode.py` | 变量、基础参数、方程与事件 |
| `systems/{budding,fission}/simulation.py`、`phenotypes.py` | 各自的初值处理、仿真及原生标签 |
| `systems/{budding,fission}/real_mutants.py` | 各系统的真实基准定义 |
| `systems/budding/mutations/`、`distributions/` | MASS与mutant_aware两套合成突变赋值和幅度分布 |
| `systems/budding/derived.py` | Twins、KO mirror和BAMS配对派生数据 |

例如，只修改 MC 的幅度分布，就修改对应的 `distributions/*.py`；不需要在每个 k-DPP、Motif 或 Random 文件中重复修改。Budding 和 Fission 分别维护自己的生物模型和采样算法，共用候选/结果结构、并行调度和导出代码。Fission 当前接入其原有 Random 算法；没有把 Budding 专用图算法直接套到 Fission 上。

`data/systems/generation.py` 不再包含 `system == 'budding' / 'fission'` 字符串分支：系统上下文、碰撞去重、救援、仿真、结果过滤、real 数量校验和派生数据入口都由对应 `adapter` 提供。新增系统时只需实现协议并在 `systems/registry.py` 注册，不需要修改共享 runner。

## real mutant 和两套突变实现

Budding 的基础参数、变量初值、ODE、事件和 real mutant 全部取自 `mutant_aware.py`。新代码唯一的 real 定义位于 `systems/budding/real_mutants.py`，不再使用旧 MASS 的 real 初值。两套 `mutations/` 和 `distributions/` 只决定**合成候选**如何变异；选择 `mass` 不会切换 real 基准或基础方程。

Fission 的 real 定义独立放在 `systems/fission/real_mutants.py`，保留原脚本的 20 条规则，包括 WT。名字不同但参数相同的实验规则仍分别保留；只对合成候选进行碰撞处理。

LHS 与 `real_mutants_only.py` 原样留存，未迁移到新配置。

## 旧脚本与新配置对应关系

| 旧脚本 | `--recipe` | 默认合成突变 / 幅度分布 |
| --- | --- | --- |
| `generate_pure_random_dataset.py` | `budding_random` / `budding_random_ko_interval` | mutant_aware / mutant_aware |
| `mass_random_sampler.py` | `budding_mass_random` | mass / mass |
| `mass_random_mutant_dist.py` | `budding_mass_random_dist` | mass / mutant_aware |
| `mass_dataset_generator.py` | `budding_mass` | mass / mass |
| `mutant_aware.py` | `budding_mutant_aware` | mutant_aware / mutant_aware |
| `gibbs_boltzmann_sampler.py` | `budding_gibbs` | mutant_aware / mutant_aware |
| `gibbs_sobol_sampler.py` | `budding_gibbs_sobol` | mutant_aware / mutant_aware |
| `generate_kdpp_gated_dataset.py` | `budding_kdpp` | mutant_aware / mutant_aware |
| `generate_kdpp_curv_gated_dataset.py` | `budding_kdpp_curvature` | mutant_aware / mutant_aware |
| `generate_motif_topology_dataset.py` | `budding_motif` | mutant_aware / mutant_aware |
| `bams_dataset_generator.py` | `budding_bams` | mutant_aware / mutant_aware |
| `mass_wt_blind_sampler.py` | `budding_wt_blind` | mass / mass；保留专用 QSS、逆解、46 边图和 ledger |
| `generate_1m_pure_random_pool.py` | `budding_pool` | mutant_aware / mutant_aware；不仿真 |
| `generate_distribution_ablation_twins.py` | `budding_twins` | 两套实现配对比较 |
| `mirror_pure_random_ko_interval.py` | `budding_ko_mirror` | 保留源数据，只重算 KO 行 |
| `generate_bams_ablation_datasets.py` | `budding_bams_ablation` | 保留共享基底，增加配对诊断样本 |
| `fission_random_sampler.py` | `fission_random` | Fission 自己的突变和分布 |

旧 MASS 的定向 motif 与新版 MASS 的结构选择分别保留在 `systems/budding/samplers/mass_motifs.py`、`systems/budding/samplers/mass.py`。WT-Blind 的专用强度策略保留，`--wt-fold-dist tier2` 为默认值，也可选择旧有的 `flat` 策略；其模式频率默认 `flat`，可用 `s5`。

## 如何运行

从 `budding_yeast_v2/` 目录执行。推荐使用项目的 `neuralop` 环境。

```bash
python data/generate.py --list

# 只查看组合和运行参数，不生成数据
python data/generate.py --recipe budding_kdpp --show-config

# 少量合成样本检查；不包含 real
python data/generate.py --recipe budding_kdpp --samples 100 \
  --seed 42 --workers 4 --no-real --output /tmp/budding_kdpp_modular.npz

# 同一算法切换合成突变与幅度分布
python data/generate.py --recipe budding_kdpp --samples 100 \
  --mutation-profile mass --distribution-profile mass \
  --seed 42 --workers 4 --output /tmp/budding_kdpp_mass_modular.npz

python data/generate.py --recipe fission_random --samples 100 \
  --seed 20000 --workers 4 --output /tmp/fission_modular.npz

python data/generate.py --recipe budding_pool --samples 1000000 \
  --seed 42 --output /tmp/budding_pool_modular.npz

python data/generate.py --recipe budding_twins --samples 100 --seed 42 \
  --output /tmp/twins_mutant_aware.npz --paired-output /tmp/twins_mass.npz

python data/generate.py --recipe budding_ko_mirror \
  --source /tmp/budding_kdpp_modular.npz --output /tmp/budding_kdpp_mirror.npz
```

各recipe继承各自默认值：普通Budding通常为100000/seed42，Fission为1000/seed20000，参数池为1000000/seed42；完整列表用 `--list` 查看。`--samples` 是请求的合成候选数，real 默认额外添加，`--no-real` 可关闭。数值失败、质量过滤和曲率门控可能使实际导出的样本数减少；NPZ 中记录请求数与实际数。

默认拒绝覆盖已有输出，只有显式 `--overwrite` 才允许覆盖。派生数据始终禁止覆盖源文件。Twins 使用两侧共同的成功行，保证配对；BAMS 消融同时输出 control、`_hard.npz` 和 `_hypo.npz`。KO mirror 若重算失败则报错，不保存“新参数配旧轨迹”。

## 保持一致的范围与明确差异

1. **旧代码原样保留**：迁移前 `data/` 下 44 个 Python 文件的 SHA256 保存在 `data/systems/legacy_sources.json`，测试逐文件校验。旧入口没有改成转发，因此仍可独立作为历史对照。
2. **基础生物模型严格对照**：Budding 按 mutant_aware；Fission 按 fission_random_sampler。参数、变量顺序、求解公式、事件与标签算法从原代码拆出；不运行旧脚本来冒充模块化实现。
3. **按用户要求统一的行为**：新流程使用 MA 的 Budding real 初值，合成候选碰撞后统一尝试换突变形式或碳源救援。原来直接丢弃碰撞的分支不再沿用。因此这些分支之后的完整样本序列不保证等同旧文件。救援上限默认 30 次，耗尽时报错，不输出碰撞候选。
4. **随机数边界**：普通算法保留原有抽样方式和调用次序，运行前设种子，完成后恢复调用方的全局 RNG。BAMS 工作者内的二分探测使用 seed 与 sample_id 派生的随机流，避免依赖进程完成顺序；不能据此保证与旧进程随机状态下的整批 BAMS 输出逐字节相同。Fission 每次运行重新构造 L1 网格，修复旧实现同进程重复调用时从已洗牌网格开始的问题。
5. **输出可核查**：保留主要轨迹/参数/名字/标签字段，新增系统、版本、参数初值、real 标志、层级、配对编号、配置、覆盖字典及生成报告。NPZ 文件的字段集合与压缩字节不保证和旧文件一致。参数池只保存参数和初值，不制造假轨迹或假标签；其中均值/标准差仅描述候选池，训练仍使用 train-only 统计量。
6. **失败处理**：只导出完整、有限且维度正确的轨迹。Fission 显式检查积分完成到 300 min；病态仿真有 worker 内期限与并行父进程停滞保护。历史失败样本的导出行为不作为等价目标。

Budding 输出维度仍为 `N × 39 × 500`、参数 141 维、210 min；Fission 为 `N × 16 × 500`、参数 58 维、300 min，保留原质量单位换算。Fission 初值来自预热，不用 Budding 的固定初值逻辑。

## 本次收口验证（2026-10-21）

393 项独立回归测试通过，额外覆盖增量索引前后边界、全局状态别名、旧Sampler/pickle导入、混合指标格式和CLI清单。最终核验确认 31 个原始生成脚本、18 套生成配置均未改变；381 项突变规则审计和 105 项采样审计通过。Budding/Fission 均完成小规模双进程生成及数据读取验证。验证记录位于 `scratch/probes/systems_consolidation_2026-10-21/verification.json`。
突变/分布审计381/381，采样/RNG审计105/105通过。Budding与Fission均实际双进程生成小文件并成功校验。
全部18个recipe字段/默认值对照收口前快照一致；不改生物方程、原标签、突变/分布或救援强度策略。
去重修复恢复原allclose判据，对曾漏判的重复候选会触发救援，因此未来新生成序列可能与漏洞版本不同；
不改写既有数据或实验。证据保存在 `scratch/probes/systems_consolidation_2026-10-21/`。

## 前期验证及证据

后续已补充采样算法、两套突变分布和功能模块的独立核验，并修正迁移遗漏，见 [采样一致性报告](SAMPLER_EQUIVALENCE_AUDIT_2026-10-20.md)。

验证环境：`/home/users/zsy/.conda/envs/neuralop/bin/python`，NumPy 2.2.6、SciPy 1.15.3。浮点逐元素一致的结论限定于该环境和被检查的样本。

- 完整 126 个 Budding real mutant：旧 MA 与新模拟器的结果逐元素一致，最大绝对差为 **0**。审计脚本 `scratch/probes/audit_modular_real_parity.py`，结果 `scratch/probes/modular_real_parity.json`。
- 完整 20 个 Fission real mutant（含 WT）：旧脚本与新模拟器的轨迹、参数和表型逐项一致，轨迹最大绝对差为 **0**。审计脚本 `scratch/probes/audit_modular_fission_parity.py`，结果 `scratch/probes/modular_fission_parity.json`。
- `tests/test_modular_datasets.py`：初次 51 项通过；采样一致性补充后 **65 项通过**，覆盖两套突变赋值和分布、固定 seed 候选序列、救援、图核、WT-Blind、Fission 预热/轨迹、NPZ 和派生配对流程。
- 旧回归覆盖 pure-random RNG、运行契约、IC/WT、图构建和关系图迁移，**399 项通过**。
- 实际执行新 CLI，双进程生成 4 条合成轨迹并成功读取；输出仅位于 `scratch/probes/`。

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/users/zsy/.conda/envs/neuralop/bin/python -m pytest -q tests/test_modular_datasets.py
```

测试和临时审计按仓库要求位于被忽略的 `tests/`、`scratch/`。没有重新生成正式十万/百万规模数据，也没有执行训练；不能把小规模逐项等价检查表述为所有算法、所有随机种子的整批 NPZ 完全相同。
