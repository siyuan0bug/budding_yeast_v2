# AGENTS.md — 出芽酵母细胞周期神经算子项目 AI Agent 工作指南

> **适用对象**：面向所有介入本项目的 AI Agent（如 Claude, Gemini, GPT 等）与核心研发人员。
> **核心使命**：本指南确立了代码库的**绝对工程铁律**、**全类型代码存放规范**、**系统架构认知字典**与**防坑避雷指南**。任何 Agent 在执行增删改查任务前，必须严格遵守本文档所立之规范。

---

## 目录
- [一、AI Agent 开发铁律与全类型代码归宿规范 (MANDATORY)](#一ai-agent-开发铁律与全类型代码归宿规范-mandatory)
  - [1.1 绝对工程铁律：严禁在仓库根目录乱放临时脚本](#11-绝对工程铁律严禁在仓库根目录乱放临时脚本)
  - [1.2 全类型代码存放字典 (Where Code Belongs)](#12-全类型代码存放字典-where-code-belongs)
- [二、科学背景与预测任务定义](#二科学背景与预测任务定义)
  - [2.1 输入输出张量契约](#21-输入输出张量契约)
  - [2.2 核心挑战：Sim-to-Real 泛化与测试基准](#22-核心挑战sim-to-real-泛化与测试基准)
- [三、运行环境与执行指令 (Environment & CLI)](#三运行环境与执行指令-environment--cli)
  - [3.1 Python 解释器与关键依赖](#31-python-解释器与关键依赖)
  - [3.2 必须设置的 PYTHONPATH](#32-必须设置的-pythonpath)
  - [3.3 核心生产入口与流水线](#33-核心生产入口与流水线)
- [四、模型架构体系 (`models/`)](#四模型架构体系-models)
  - [4.1 统一 Forward 接口契约](#41-统一-forward-接口契约)
  - [4.2 标准基线模型体系 (`models/baselines/`)](#42-40-基线模型体系-modelsbaselines)
  - [4.3 30+ 原创生物算子体系 (`models/ours/`)](#43-30-原创生物算子体系-modelsours)
- [五、损失函数与物理约束体系 (`utils/losses.py`)](#五损失函数与物理约束体系-utilslossespy)
- [六、主动学习 (Active Learning) 体系 (`utils/al_callback.py`)](#六主动学习-active-learning-体系-utilsal_callbackpy)
  - [6.1 1M 免预仿真纯参数空间候选池架构](#61-1m-免预仿真纯参数空间候选池架构)
  - [6.2 当前在线策略与官方对齐状态](#62-当前在线策略与官方对齐状态)
- [七、数据生成、归一化与生物网络图谱 (`data/`)](#七数据生成归一化与生物网络图谱-data)
  - [7.1 归一化规范与最新实测证据](#71-归一化规范与最新实测证据)
  - [7.2 数据集生成引擎清单](#72-数据集生成引擎清单)
  - [7.3 多关系生物因果拓扑图](#73-多关系生物因果拓扑图)
- [八、关键避坑指南与科学细节 (Gotchas & Best Practices)](#八关键避坑指南与科学细节-gotchas--best-practices)
- [九、权威专著文档导航 (`docs/`)](#九权威专著文档导航-docs)

---

## 一、AI Agent 开发铁律与全类型代码归宿规范 (MANDATORY)

### 1.1 绝对工程铁律：严禁在仓库根目录乱放临时脚本
> [!CAUTION]
> **绝对禁止 AI Agent 在仓库根目录下直接创建任何临时探针、测试或调试 `.py` 文件！**（如 `_cap_probe_tmp.py`, `inspect_xxx.py`, `temp_test.py` 等）。
> 仓库根目录只允许保留核心生产入口（`train.py`, `eval.py`, `tune.py`）、批量运行脚本（`run_pipeline.sh`, `run_al_budding.sh`, `run_tune.sh`）以及权威文档（`README.md`, `AGENTS.md`）。

### 1.2 全类型代码存放字典 (Where Code Belongs)

当 AI Agent 根据用户需求编写不同类型的代码时，**必须严格存放在对应专属目录**：

| 代码类型 | 必须存放的目标目录 | 示例 / 说明 | 注册与合规要求 |
|:---|:---|:---|:---|
| **神经网络模型** | `models/baselines/` 或 `models/ours/` | 如新基线 `mamba_no.py`，新原创 `bio_refiner_v3.py` | 必须在 [models/yeast_lit_module.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/models/yeast_lit_module.py) 的 `MODEL_REGISTRY` 中注册，并提供标准 `forward_ic_time` 接口。 |
| **损失函数 / 物理约束** | `utils/losses.py` | 如新损失 `AdaptivePhaseLoss` | 必须在 [utils/losses.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py) 的 `LOSS_REGISTRY` 中注册，支持课程学习权重调度。 |
| **主动学习策略 / 物理残差** | `utils/active_learning/` / `utils/ode_residual_engine.py` | 如新 AL4PDE 策略实现 | 策略在 `utils/active_learning/strategies/` 实现并注册到 `registry.py`；统一返回 SelectionPlan，仿真/入库由 engine 驱动。`al_callback.py` 只保留兼容导入。 |
| **数据生成器 / 空间采样器** | `data/` | 如 `gibbs_sobol_sampler.py`, `bams_dataset_generator.py` | 保持模块化独立运行，支持 `--seed` 与 `--data_dir` 参数。 |
| **持久生产级分析 / 评测工具** | `scripts/analysis/` 或 `scripts/eval/` | 如 `analyze_experiments.py`, `eval_ensemble_benchmark.py` | 必须使用 `argparse` 提供标准命令行参数，长期保留供科研人员复用。 |
| **自动化单元测试 / 回归测试** | `tests/` | 如 `test_bio_refiner_causal.py`, `verify_training_loop.py` | 供开发验证与 CI 冒烟测试使用，已被 `.gitignore` 忽略。 |
| **AI 临时探针 / 调试沙盒 / 一次性脚本** | `scratch/` 或 `scratch/probes/` | 如 `_cap_probe_tmp.py`, `test_batch_residual.py` | **所有临时、探针、排查脚本强制存入此处，随用随弃**，已被 `.gitignore` 忽略。 |

### 1.3 绝对工程铁律：同步维护权威文档 (Mandatory Synchronous Documentation)
> [!IMPORTANT]
> **每当向代码库新增功能或产生新的实验结果时，必须在对应的 `docs/` 文档中同步更新：**
> 1. **新增模型/模块**：在 `models/` 编写并注册新模型后，必须同步在 [docs/1_ARCHITECTURE_AND_MODELS.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/1_ARCHITECTURE_AND_MODELS.md) 补充模型架构说明、输入输出、参数量与调用命令。
> 2. **新增主动学习策略**：在 `utils/active_learning/strategies/` 实现并注册新策略后，必须同步在 [docs/2_ACTIVE_LEARNING_GUIDE.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/2_ACTIVE_LEARNING_GUIDE.md) 补充数学设计、采样目标函数与实测性能。
> 3. **新增损失函数/物理约束**：在 `utils/losses.py` 新增损失后，必须同步在 [docs/3_LOSSES_AND_PHYSICS_CONSTRAINTS.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/3_LOSSES_AND_PHYSICS_CONSTRAINTS.md) 补充数学公式与梯度特性。
> 4. **新增数据生成器/归一化**：在 `data/` 编写新生成引擎或验证归一化后，必须同步在 [docs/4_DATASETS_AND_NORMALIZATION.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/4_DATASETS_AND_NORMALIZATION.md) 补充采样空间分布与物理基准。
> 5. **实验完成/评测出炉**：任何批次实验（如 `run_pipeline.sh`, `run_al_budding.sh` 等）运行完毕后，必须运行分析脚本提取关键指标，并同步更新至 [docs/5_BENCHMARK_AND_LEADERBOARD.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/5_BENCHMARK_AND_LEADERBOARD.md)（与对应的专门指南中），严禁只在终端汇报而导致专著文档脱节！

### 1.4 AI 专用分析、探针与评测工具库导航 (AI Tooling Navigation Guide)
代码库为 AI Agent 和科研人员配备了多层次的诊断工具链，当遇到性能瓶颈、收敛异常或需开展归因分析时，**优先复用以下现有成熟工具，避免重复造轮子**：

#### 1. 深度实验分析工具库 (`scripts/analysis/`)
- [analyze_experiments.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/scripts/analysis/analyze_experiments.py)：自动化多实验横向对比，可自动遍历多级目录提取 `global_metrics_denorm.json`，生成 Markdown 排行榜表格与 CSV 矩阵。
- [test_ko_perturbation_sensitivity.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/scripts/analysis/test_ko_perturbation_sensitivity.py)：基因敲除参数微扰敏感度分析器，探测模型在敲除边界条件（如连续 $[0, 10^{-4}]$ 微扰）下的梯度平滑性与鲁棒性。
- [evaluate_sampling_efficacy.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/scripts/analysis/evaluate_sampling_efficacy.py)：定量评测采样策略在高维参数空间中的空间覆盖率（Space-Filling Index）、均匀度与多样性。

#### 2. 数据集高维几何诊断与 Badcase 根因归因库 (`scripts/analysis/dataset_diagnosis/`)
- [analyze_pure_random_badcases.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/scripts/analysis/dataset_diagnosis/analyze_pure_random_badcases.py)：纯随机训练集泛化至 126 突变体的 Badcase 详细逐通道时序对比与根因诊断。
- [analyze_massv2_badcase_attribution.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/scripts/analysis/dataset_diagnosis/analyze_massv2_badcase_attribution.py)：MassV2 等大样本数据集的坏例数据归因分析（分析特征邻域、支撑集稀疏度与相空间流速）。
- [deep_dataset_diagnosis.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/scripts/analysis/dataset_diagnosis/deep_dataset_diagnosis.py)：深入诊断训练集参数流形覆盖与支撑集完整度。
- [deep_genotype_distribution_analysis.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/scripts/analysis/dataset_diagnosis/deep_genotype_distribution_analysis.py)：对比合成数据集与 126 真实实验突变体的基因型频率、敲除模式重叠率与 Sim-to-Real 覆盖盲区。
- [generate_multimetric_top20_badcases.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/scripts/analysis/dataset_diagnosis/generate_multimetric_top20_badcases.py)：多指标综合打分生成 Top-20 最具代表性的真实突变体坏例，并输出高分辨时序曲线与误差热图。

#### 3. 评测与突变体验证工具库 (`scripts/eval/`)
- [eval_ensemble_benchmark.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/scripts/eval/eval_ensemble_benchmark.py)：多模型集成（Ensemble）推断与评测，对比单模型 vs 集成在 126 突变体上的方差缩减与鲁棒性提升。

#### 4. 自动化单元测试与算子验证 (`tests/`)
- [test_bio_refiner_causal.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/tests/test_bio_refiner_causal.py) 等：涵盖因果掩码单向可达性验证、维度对齐校验、前向与反向梯度流通测试。

#### 5. AI 临时探针与调试沙盒 (`scratch/` & `scratch/probes/`)
- 当 AI Agent 需要针对特定模块编写单次排查脚本时，统一写入 `scratch/probes/`，支持快速验证，不会污染版本控制。

---

## 二、科学背景与预测任务定义

这是一个**基于神经算子（Neural Operator）的代理模型（surrogate model）项目**，用于高精度预测出芽酵母（*Saccharomyces cerevisiae*）细胞周期动态演化。

### 2.1 输入输出张量契约
- **输入参数 $P$**：141 维连续/离散动力学参数向量（覆盖 141 个反应速率常数、合成速率、降解速率及基因敲除/过表达覆盖），形状 `(B, 141)`。
- **输入初始条件 $X$**：39 维分子物种的初始浓度及其一阶时序差分，形状 `(B, 39, 2)`。
- **输出轨迹 $Y$**：39 维分子浓度在 500 个时间步（跨度 210 min）上的演化轨迹，形状 `(B, 39, 500)`。
- **39 个分子物种 (`VARIABLE_NAMES`)**：
  `MASS`, `CLN2`, `CLB2`, `CLB5`, `SIC1`, `CDC6`, `C2`, `C5`, `F2`, `F5`, `SIC1P`, `C2P`, `C5P`, `CDC6P`, `F2P`, `F5P`, `SWI5T`, `SWI5`, `IEP`, `CDC20T`, `CDC20`, `CDH1T`, `CDH1`, `CDC14T`, `CDC14`, `NET1T`, `NET1`, `RENT`, `TEM1`, `CDC15`, `PPX`, `PDS1`, `ESP1`, `ORI`, `BUD`, `SPN`, `Vi20`, `lte1`, `BUB2`。
- **9 个核心评估变量 (`CORE_9_INDICES`)**：`[0, 1, 2, 3, 4, 20, 22, 33, 35]`（对应 MASS, CLN2, CLB2, CLB5, SIC1, CDC20, CDH1, ORI, SPN）。

### 2.2 核心挑战：Sim-to-Real 泛化与测试基准
- **训练集**：由 ODE 数值积分器（`solve_ivp`）在纯随机/准随机空间填充分布上生成的数值仿真样本（数万至数十万量级）。
- **测试基准（Ground Truth）**：**126 个具有生物学真值实验验证的酵母突变体（Real Mutants）轨迹**。
- **核心矛盾**：模型只能在合成参数分布上训练，但必须在 126 个真实实验突变体上精准泛化（解决 Sim-to-Real Gap、相变悬崖与双稳态阻滞伪振荡）。

---

## 三、运行环境与执行指令 (Environment & CLI)

### 3.1 Python 解释器与关键依赖
- **Conda 环境**：`neuralop` (Python 3.10)
- **解释器绝对路径**：`/home/users/zsy/.conda/envs/neuralop/bin/python`
- **核心依赖**：`torch`, `pytorch_lightning`, `scipy`, `scikit-learn`, `optuna`, `wandb`, `ptwt` (小波), `torchdiffeq`。

### 3.2 必须设置的 PYTHONPATH
所有内部模块均采用 `budding_yeast_v2.` 前缀规范导入，因此执行脚本前**必须确保父目录在 `PYTHONPATH` 中**：
```bash
export PYTHONPATH=/home/users/zsy/projects/budding_yeast_cell_cycle:$PYTHONPATH
```
*注：[train.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/train.py)、[eval.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/eval.py) 及所有 shell 脚本已内置头部自动探测修正逻辑。*

### 3.3 核心生产入口与流水线
```bash
# 1. 单模型基线训练 (例如 BioRefinerPlus + HybridMSERelLoss)
python train.py \
    --model biom_refiner_plus \
    --loss_type hybrid_mse_rel \
    --dataset_name /home/users/zsy/mydata/budding_yeast/gibbs_boltzmann_100000_210m_500s.npz \
    --modes 24 --width 192 --batch_size 256 --max_epochs 100 \
    --lr 8.8e-4 --warmup_epochs 8 --norm_type log1p --al_strategy none

# 2. 1M 纯随机池 AL4PDE 闭环主动学习批量运行
bash run_al_budding.sh

# 3. 评测与 126 真实突变体轨迹绘制
python eval.py \
    --ckpt experiments/al_pure_random_1m/biom_refiner_plus/hybrid_mse_rel_al4pde_bifurcation/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/checkpoints/best.ckpt \
    --dataset_name /home/users/zsy/mydata/budding_yeast/gibbs_boltzmann_100000_210m_500s.npz \
    --save_dir experiments/eval_output

# 4. 全局实验指标解析与排行榜生成
python scripts/analysis/analyze_experiments.py --exp_dir experiments/al_pure_random_1m
```

---

## 四、模型架构体系 (`models/`)

### 4.1 统一 Forward 接口契约
所有模型在 [models/yeast_lit_module.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/models/yeast_lit_module.py) 中由 `forward_ic_time(x, p)` 统一调度：
- `ic_time_grid: (B, 39, 2, 500)`，包含初始浓度与时间网格。
- `params: (B, 141)`，默认是原始动力学参数逐参数 Z-score，**不是 Min-Max**。
- 返回预测张量：`pred_norm: (B, 39, 500)`。

### 4.2 标准 baseline 体系 (`models/baselines/`)
- 只保留用户采纳的标准骨干：`mlp`, `resnet`, `unet`, `kan`, `transformer`, `mamba`, `deeponet`, `fno`, `wno`, `mwt`, `gnot`, `transolver`, `lno`, `neural_ode`, `sc_fno`, `bregman_fno`, `lamo`, `konode`。
- 禁止给 baseline 添加原方法没有的参数调制、图先验、额外跨注意力等模块；只做参数、完整初值和查询时间的输入输出适配。保留原方法自身的注意力、专家和状态空间机制。
- 已删除的 `pure_*` / `*_ref` / `fno1d` / 时序预测等旧入口不再兼容；不得重新引入别名把修改版冒充标准实现。
- SC-FNO 必须使用 `sc_sensitivity` 及严格匹配数据/归一化的参考灵敏度文件；无该监督时不得以 SC-FNO 训练。
- ours 的原 WNO/LNO 使用 `cross_wno` / `cross_lno`。来源和验证见 [BASELINE_SELECTION_AUDIT.md](docs/BASELINE_SELECTION_AUDIT.md)。

### 4.3 30+ 原创生物算子体系 (`models/ours/`)
- **`BioRefinerPlus` (`biom_refiner_plus`)**：分层生物通量感知 + 多关系因果图卷积细化器（当前 SOTA 主力）。
- **`BioRefinerAttractor` & `BioRefinerAttractorPlus`**：相空间吸引子流形锁定算子（解决阻滞态伪振荡）。
- **`BioRefinerCausalReach` & `BioRefinerCausal`**：生物因果单向可达性门控网络。
- **`HANO` 系列**：分层自适应神经算子（多尺度时空谱图分解）。
- **`BioMassActionRefiner`**：质量作用定律（Mass Action Law）物理通量引导细化器。

---

## 五、损失函数与物理约束体系 (`utils/losses.py`)

[utils/losses.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py) 注册了 17 种现代损失函数（统一通过 `--loss_type` 调用）：

| 损失名称 (`loss_type`) | 对应实现类 | 核心组成与设计目的 |
|:---|:---|:---|
| `mse_only` | `MSEOnlyLoss` | 纯均方误差基线。 |
| `rel_l2` | `RelativeL2Loss` | 相对 $L_2$ 误差损失（规避大动态范围主导）。 |
| `hybrid_mse_rel` | `HybridMSERelLoss` | $\text{MSE} + \lambda_{\text{rel}} \text{RelL2}$（兼顾绝对量纲与相对波形）。 |
| `gradnorm_hybrid` | `GradNormHybridLoss` | 基于梯度范数的自适应多任务动态平衡损失。 |
| `soft_dtw` | `SoftDTWLoss` | 可微动态时间规整损失（对齐相位偏移与时滞）。 |
| `phase_derivative` | `PhaseDerivativeLoss` | 相平面一阶时间导数对齐损失（抑制高频伪振荡）。 |
| `triple_dynamic` | `TripleDynamicLoss` | $\text{MSE} + \text{RelL2} + \text{Derivative}$ 三位一体综合动态损失。 |
| `hybrid_bifurcate` | `HybridBifurcateLoss` | 分岔敏感损失（针对双稳态相变区域施加自适应权重）。 |
| `pinn_residual` | `PINNResidualLoss` | 向量化 ODE 物理微分残差 + 软事件掩码约束。 |
| `h1` | `H1Loss` | 频域 Sobolev $H^1$ 梯度平滑惩罚。 |
| `pino` | `PINOLoss` | PINN 物理残差 + 频域 $H^1$ 联合约束。 |
| `hybrid_pino_collocation`| `HybridPINOCollocationLoss` | 配点自适应物理残差损失。 |
| `hybrid_all` | `HybridAllLoss` | 数据驱动 + 导数 + DTW + 物理残差全能复合损失。 |

---

## 六、主动学习 (Active Learning) 体系 (`utils/al_callback.py`)

**当前组织**：`utils/active_learning/` 按 engine、pool、prediction/features、oracle、dataset/io、strategies 拆分；旧 `utils/al_callback.py` 为 16 行兼容入口。新增策略只向 registry 注册，CLI/ensemble 需求由同一清单派生。`tune.py` 当前明确仅支持 `--al_strategy none`。接口、归档与验证见 [拆包说明](docs/AL_PACKAGE_REFACTOR.md)。

### 6.1 1M 免预仿真纯参数空间候选池架构

**2026-10-30 存储更新**：新AL采用`shared/datasets/<sha>.npz`共享初始块、每次`added_samples.npz`及每轮`dataset.json`清单，不再保存重复累计NPZ。轨迹float32、物理参数/实际初值float64和训练输入float32/bf16保持；数据读取使用`data.dataset_archive.open_dataset`，不要直接对清单调用`np.load`。checkpoint仍严格检查同一数据哈希/切分/归一化；迁移必须逐字段验证一致后才释放旧快照。详见[存储布局](docs/AL_DATASET_STORAGE_LAYOUT.md)。下方旧快照描述保留为历史。
AL 与静态批训练的新实验结果默认根目录为机械盘 `/home/users/zsy/mydata/budding_yeast/experiments/`，两系统 AL 分别使用其下 `al_budding/`、`al_fission/`。`EXP_ROOT` 可覆盖，不把新结果默认写回项目盘。共享输入路径仍由 `DATA_DIR` 控制，Budding默认文件名为`_balanced.npz`，Fission为`_al.npz`，规则版本保留在内部元数据；用户授权删除的旧 AL 根目录输入不再使用。
当前 `run_al_budding.sh` 用 `data/generate.py --recipe budding_pool` 生成参数/初值池，`data/systems/al_initial.py` 按层 KMeans 选取并仿真初始集；之后由 `train.py` 驱动选样、在线 ODE 仿真和外层冷启动训练。两个 AL 脚本均内置 `SEEDS` 和 `EXPERIMENTS` 清单，每行指定模型、损失、策略、超参数和 GPU；Budding有界同卡并行、Fission同卡串行，均跨卡并行，先准备一个共享池及每个 seed 的初始数据，再启动 worker。Budding当前默认FNO的6个基线（Random/Uncertainty/Power/MaxDist/LCMD/BAIT）加ours_response，共7项，GPU0–3任务数2/2/2/1，MAX_JOBS_PER_GPU默认2（设1则串行），Fission仍同卡串行；Fission默认标准fno四策略4项（Random/Uncertainty/Power/BAIT，GPU0–3）；AL W&B 强制本地离线，池大小为 100 万/50 万。Budding候选池使用budding/al_pool.py两级容量配额和逐模式Sobol流（百万池L1/L2/L3=9815/126225/863960）；Fission候选池多核提议仍使用parallel_pool.py（固定256条块、SeedSequence、全局去重）；SIM_WORKERS默认32，新并行RNG契约区别于旧串行流，详见 AL_EXPERIMENT_WORKFLOW.md。Budding当前FNO统一50/50/50/100 epoch、warmup4/4/4/8、batch512、LR1e-3；其他Budding模型和AL方法以此为参考，容量仍按架构定义。Fission默认60/60/60/120 epoch、warmup6/6/6/12不变。标准 FNO width 为 Budding192/Fission128、modes24、4主干块；当前默认清单仅FNO，不以cross_fno的每变量width推导其容量。批次 `tasks.tsv` 按完成顺序记录成功失败，失败默认继续下一项，退出码仍为非零；CONTINUE_ON_ERROR=0不再启动该卡后续项，已运行项正常收尾。Budding ODE默认每实验4进程，可用AL_SIM_WORKERS覆盖。
Budding 仿真后幅值拒绝已取消，旧六种启发式策略的参数门控保持；新增ours_response不加该门控。当前四轮、初始 KMeans + seed 随机切分、逐轮累计 train 重拟合和快照 L 字段契约以 [AL_EXPERIMENT_WORKFLOW.md](docs/AL_EXPERIMENT_WORKFLOW.md) 为准；旧三轮/Pattern 分层/冻结尺度说明不适用于新 AL。Fission AL 已支持 Random/Uncertainty/Power/BAIT/LCMD 五种通用策略；其余 Budding 专用策略明确拒绝。候选初值必须逐株预热，入口与成本边界见 [Fission AL](docs/FISSION_ACTIVE_LEARNING.md)。

### 6.2 当前在线策略与官方对齐状态
保留原 12 种在线策略：`al4pde_random`、`al4pde_uncertainty`、`al4pde_diversity`、`maal`、`al4pde_mass`、`al4pde_novelty`、`al4pde_residual`、`al4pde_residual_diversity`、`al4pde_bifurcation`、`al4pde_bifurcation_entropy`、`al4pde_event_entropy`、`al4pde_fisher_oed`；四个短别名继续兼容。

新增 `al4pde_power`、`al4pde_bait`、`al4pde_lcmd`，新增预测轨迹`al4pde_max_dist`后加上`ours_response`当前合计17种在线策略；Power 复用独立 ensemble，BAIT/LCMD 使用预测轨迹共享投影及实际 bmdal 选择器；LCMD 无 posterior transform。`coreset_maxdist` 尚未实现，不能用输出轨迹替代 latent 冒充支持。官方 8 个预置配置及未移植项见 [清单](docs/AL_POWER_BAIT_AND_OFFICIAL_INVENTORY.md)。

`random/us/is/wrs/vessal/hggs/pial` 已归档至 `utils/archive/al_callback_legacy.py`，主文件和训练入口不再支持这些旧轨迹池策略。归档包含完整原文件，生产代码不可导入它来绕过限制。

当前 uncertainty 已按官方两个独立训练成员和 MSE disagreement/原始方差分支对齐，见 [Ensemble 对齐](docs/AL_ENSEMBLE_UNCERTAINTY.md)；它是项目训练适配，未直接实例化官方 Task/ProbModel。Random 保持固定池。diversity 使用自定义 332 维特征及本地 MaxDist；Fisher 名称对应特征正交化而非 Fisher 信息矩阵；分岔和事件策略是启发式扩展。不要把名称当作官方算法等价性证明。

归档、数值审计与未解决问题见 [官方核对](docs/AL_ARCHIVAL_AND_OFFICIAL_ALIGNMENT_2026-10-21.md)。复查使用 `scripts/analysis/audit_al4pde_alignment.py --output scratch/probes/al_cleanup_2026-10-21/alignment.json`。

---

## 七、数据生成、归一化与生物网络图谱 (`data/`)

### 7.1 归一化规范与最新实测证据
- **当前默认 `norm_type=log1p`**：$Y$ 先 `log1p(max(Y,0))`，再逐变量跨样本/时间 Z-score；$P$ 为原始参数逐坐标 Z-score，**不是 Min-Max，也不是参数取 log**。
- 所有工程路径共用 `data/normalization.py`。`asinh_z/asinh_scale/linear` 为显式选项，禁止在消费者内部写死另一种逆变换。
- **2026-10-19 用户后续授权 train-only**：新训练先按原规则切分，仅在合成 train 拟合 Y/P 均值、标准差与 `asinh_scale.s_c`；统一冻结用于 val/test/real 和 AL 入库。新 checkpoint 标记 `train_synthetic`。旧载荷保留 `all_synthetic_legacy`，无完整载荷的旧权重仅按历史范围重建，禁止默配新统计量。重复样本分组和指标口径仍未修改；pure-random 生成 RNG 的后续修复见 7.2。详见 [train-only 修复](docs/TRAIN_ONLY_NORMALIZATION_2026-10-19.md)。
- **最新实测证据（2026年9月基准）**：
  在 `mutant_aware_pure_random_100000_210m_500s/biom_refiner_plus/` 下的三种归一化对比实测证明：
  - **log1p + Y/P Z-score**：Real Rel L2 = **0.1518**, Real Corr = **0.9723**, Real MAE = **0.0869**（这组三项最好，但 MSE 不是最好）。
  - **Asinh Scale**：Real Rel L2 = 0.2054, Real Corr = 0.9576。
  - **Asinh Z-Score**：Real Rel L2 = 0.1713, Real Corr = 0.9655。
  这是历史同组存档，不是本轮重训结果。物理 MSE 分别为 log1p 5.18461、asinh_z 5.51186、asinh_scale 2.42423。
  checkpoint/启动/自回归/求解保护等当前契约见 [运行链路修复](docs/RUNTIME_CONTRACT_FIX_2026-10-19.md)。

### 7.2 数据集生成引擎清单

**2026-10-22**：当前 systems 静态10/40/50；Budding AL已在2026-10-28改为模块集合/模式两级均衡，不再用5/40/55（Fission仍用原规则），仅精确L1–L3，原始生成器仍保留。修改采样前必须阅读 [三模块规则](docs/EXACT_THREE_MODULE_SAMPLING.md)，不得重新引入独立兜底配额或默默截断旧高阶数据。
**2026-10-19 RNG 后续修复**：`generate_pure_random_dataset.py` 的局部 `RandomState(seed)`
现显式传入幅度采样、随机等位基因处理、冲突重抽和碰撞救援；显式 RNG 的冲突键按固定顺序遍历。
新 NPZ 记录 `sampling_seed` / `sampling_rng_contract=pure_random_single_stream_v1`。
共享辅助函数省略 rng 时仅为兼容旧调用方而保留全局行为；新/迁移的生成器应显式传入同一 rng，
不能据此宣称其他生成器也已完成迁移。ODE、采样范围/概率权重、real 筛选及旧数据未修改。
详见 [RNG 修复与 corr 风险说明](docs/PURE_RANDOM_RNG_FIX_2026-10-19.md)。

`data/` 下的所有数据生成与采样器已全部得到保护与留存：
- `generate_1m_pure_random_pool.py`：1M 纯随机参数候选池生成器。
- `gibbs_sobol_sampler.py` & `gibbs_boltzmann_sampler.py`：加扰 Sobol 与能量分布准随机采样。
- `bams_dataset_generator.py` & `mass_dataset_generator.py`：生物学感知与分层变异采样。
- `generate_kdpp_gated_dataset.py`, `generate_kdpp_curv_gated_dataset.py`, `generate_motif_topology_dataset.py` 等。
- `reachability_check.py`：可达性与零训练探针引擎（支持任意目录执行，默认输出至 `data/reachability/`）。

### 7.3 多关系生物因果拓扑图
当前权威潜在结构由 `data/yeast_graph.py` 从 `mutant_aware.py` 的 ODE/事件源码派生，
统一 `A[target,source]`。关系图由 `data/yeast_relations.py` 的反应项注释覆盖层与
untyped 补充组成，不从旧 NPZ 读取连边；默认 hybrid 为 226 条非自环边。
旧 `data/yeast_multirelational_graphs.npz` 保留供历史复现和 circuit/module 元数据，
不能称为完整 ODE 图。DynamicPlus 模块方向已修复，soft attention 保留。
固定图默认 current 刷新，历史图复现显式使用 checkpoint 策略。完整范围与限制见
[docs/GRAPH_RELATION_MODULE_FIX_2026-10-19.md](docs/GRAPH_RELATION_MODULE_FIX_2026-10-19.md)。

---

## 八、关键避坑指南与科学细节 (Gotchas & Best Practices)

1. **`PARAM_KEYS` 顺序绝对不可更改**：
   `al_callback.py`、`physics_eqns_torch.py`、`mass_dataset_generator.py` 均严格依赖 `sorted([k for k in _defaults.keys() if k != 'is_mutant_104'])` 得到的 141 维特征顺序。
2. **ODE BDF 积分器三层超时保护**：
   在调用 `solve_ivp(method='BDF')` 仿真复杂突变体时，病态参数可能导致底层 Fortran 循环挂起。必须保证：
   - 内置 `SIGALRM` 5秒硬超时中断；
   - Python 父进程 35 秒无完成结果时终止自有 worker；AL 最多重建一次未交付任务，重复失败则中止生成（不是对已完成 future 调用 timeout）；
   - `try...except` 捕获异常返回 `None`。
3. **日志输出与实验管理**：
   所有实验日志唯一写入各自的 `experiments/.../train.log`，不再在根目录生成 `logs/` 或 `logs_v3/` 软链接。
4. **Git 仓库管理**：
   所有大文件（`*.npz`, `*.npy`, `*.ckpt`, `*.zip`）、实验输出（`experiments/`, `eval_result/`）、临时沙盒（`scratch/`）、测试输出（`tests/`）已全部被 `.gitignore` 纳管，严禁提交至版本库。

---

## 九、权威专著文档导航 (`docs/`)

如需查阅更详细的数学公式推导、模型内部实现与评测结果，请直接阅读 `docs/` 下的 5 大专题文档：
- 📘 [docs/1_ARCHITECTURE_AND_MODELS.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/1_ARCHITECTURE_AND_MODELS.md) — 系统架构与 70+ 模型全景速查字典
- 📗 [docs/2_ACTIVE_LEARNING_GUIDE.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/2_ACTIVE_LEARNING_GUIDE.md) — 全量主动学习算法与 AL4PDE 体系详解
- 📙 [docs/3_LOSSES_AND_PHYSICS_CONSTRAINTS.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/3_LOSSES_AND_PHYSICS_CONSTRAINTS.md) — 17 种损失函数、GradNorm 梯度平衡与 PINO 约束
- 📕 [docs/4_DATASETS_AND_NORMALIZATION.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/4_DATASETS_AND_NORMALIZATION.md) — 数据集生成引擎、三大归一化实证与生物图谱
- 📓 [docs/5_BENCHMARK_AND_LEADERBOARD.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/5_BENCHMARK_AND_LEADERBOARD.md) — 126 真实突变体基准评测排行榜与 Badcase 审计

AL精度边界：`_pool_initial_physical`用于精确来源保存（Budding float64），模型缓存由`refresh_model_inputs(dm)`统一生成float32；train.py每轮必须调用它，时间轴匹配IC dtype。不能直接normalize_y原始float64初值后推理；此次六策略首轮故障及真实checkpoint验证见docs/AL_DTYPE_FIX_2026-10-28.md。

2026-10-30 新AL优化：默认 AL_SUITE=optimized，四个 global_* 策略＋六项中断重跑＋修正event；full预算不变，GPU0–3各一项，旧结果不可覆盖。方法/内存修复/验证见 docs/AL_OPTIMIZED_RESTART_2026-10-30.md。
