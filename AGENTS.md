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
  - [4.2 40+ 基线模型体系 (`models/baselines/`)](#42-40-基线模型体系-modelsbaselines)
  - [4.3 30+ 原创生物算子体系 (`models/ours/`)](#43-30-原创生物算子体系-modelsours)
- [五、损失函数与物理约束体系 (`utils/losses.py`)](#五损失函数与物理约束体系-utilslossespy)
- [六、主动学习 (Active Learning) 体系 (`utils/al_callback.py`)](#六主动学习-active-learning-体系-utilsal_callbackpy)
  - [6.1 1M 免预仿真纯参数空间候选池架构](#61-1m-免预仿真纯参数空间候选池架构)
  - [6.2 10 大生产级 AL4PDE 策略速查](#62-10-大生产级-al4pde-策略速查)
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
> 仓库根目录只允许保留核心生产入口（`train.py`, `eval.py`, `tune.py`）、批量运行脚本（`run_pipeline.sh`, `run_al_pure_random_1m.sh`, `run_tune.sh`）以及权威文档（`README.md`, `AGENTS.md`）。

### 1.2 全类型代码存放字典 (Where Code Belongs)

当 AI Agent 根据用户需求编写不同类型的代码时，**必须严格存放在对应专属目录**：

| 代码类型 | 必须存放的目标目录 | 示例 / 说明 | 注册与合规要求 |
|:---|:---|:---|:---|
| **神经网络模型** | `models/baselines/` 或 `models/ours/` | 如新基线 `mamba_no.py`，新原创 `bio_refiner_v3.py` | 必须在 [models/yeast_lit_module.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/models/yeast_lit_module.py) 的 `MODEL_REGISTRY` 中注册，并提供标准 `forward_ic_time` 接口。 |
| **损失函数 / 物理约束** | `utils/losses.py` | 如新损失 `AdaptivePhaseLoss` | 必须在 [utils/losses.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py) 的 `LOSS_REGISTRY` 中注册，支持课程学习权重调度。 |
| **主动学习策略 / 物理残差** | `utils/al_callback.py` / `utils/ode_residual_engine.py` | 如新 AL4PDE 策略实现 | 必须在 [utils/al_callback.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py) 内部实现并支持 GPU 向量化与超时兜底。 |
| **数据生成器 / 空间采样器** | `data/` | 如 `gibbs_sobol_sampler.py`, `bams_dataset_generator.py` | 保持模块化独立运行，支持 `--seed` 与 `--data_dir` 参数。 |
| **持久生产级分析 / 评测工具** | `scripts/analysis/` 或 `scripts/eval/` | 如 `analyze_experiments.py`, `eval_ensemble_benchmark.py` | 必须使用 `argparse` 提供标准命令行参数，长期保留供科研人员复用。 |
| **自动化单元测试 / 回归测试** | `tests/` | 如 `test_bio_refiner_causal.py`, `verify_training_loop.py` | 供开发验证与 CI 冒烟测试使用，已被 `.gitignore` 忽略。 |
| **AI 临时探针 / 调试沙盒 / 一次性脚本** | `scratch/` 或 `scratch/probes/` | 如 `_cap_probe_tmp.py`, `test_batch_residual.py` | **所有临时、探针、排查脚本强制存入此处，随用随弃**，已被 `.gitignore` 忽略。 |

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
bash run_al_pure_random_1m.sh

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
- `params: (B, 141)`，Min-Max 标准化后的动力学参数。
- 返回预测张量：`pred_norm: (B, 39, 500)`。

### 4.2 40+ 基线模型体系 (`models/baselines/`)
- **神经算子族**：`pure_fno`, `fno1d`, `pure_wno`, `cno1d`, `mwt1d`, `deeponet`, `gcgnet`, `gnot`, `transolver1d`, `transolver3_1d`, `dpot1d`, `sc_fno`, `pno1d`, `mpno1d`, `bregman_fno`, `bregman_wno`, `ncflow`, `iganet_transformer` 等。
- **时序 Transformer 族**：`itransformer`, `dlinear`, `nlinear`, `linear`, `autoformer`, `informer`, `iinformer`, `flowformer`, `iflowformer`, `reformer`, `ireformer`, `flashformer`, `iflashformer`, `vanilla_transformer`, `fedformer`, `pyraformer` 等。
- **动力系统基线**：`neural_ode`, `konode`, `koopman_dmd`, `latent_lno`, `lamo`, `wdno`, `icon`, `bampno`, `noble`。

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

### 6.1 1M 免预仿真纯参数空间候选池架构
在现代 AL4PDE 体系中，候选池由 [generate_1m_pure_random_pool.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/generate_1m_pure_random_pool.py) 生成 100 万组纯参数组合（仅包含参数 $P$ 与初始条件 $X$，**无预计算轨迹 $Y$，100% 杜绝数据泄露**）。
由 [al_pool_manager.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/al_pool_manager.py) 进行内存映射管理，主动学习触发时由模型前向预测并打分，仅对 Top-K 选中的高价值样本调用多进程 ODE 求解器进行实时在线标注。

### 6.2 10 大生产级 AL4PDE 策略速查
1. **`al4pde_random`**：无偏高维均匀随机采样基线。
2. **`al4pde_uncertainty`**：MC-Dropout 特征方差预测不确定度。
3. **`al4pde_diversity`**：相空间 332 维时序形状特征抽取 + GPU 向量化 K-Center/MaxDist 贪心空间覆盖。
4. **`al4pde_residual`**：流速自适应相对生化物理残差（含 7 大快变量舍弃机制与 $L_{\text{flux}}$ 动态门控）。
5. **`al4pde_res_div`**：物理残差与几何空间多样性双目标帕累托前沿融合。
6. **`al4pde_entropy`**：双稳态相空间转移概率香农熵采样。
7. **`al4pde_bifurcation`**：生物学分岔指标引导采样（周期突变、阻滞跳变、振幅极值突跃）。
8. **`al4pde_bifurcation_entropy`**：分岔强度与动力学不确定度联合熵。
9. **`al4pde_bifurcation_diversity`**：分岔边界引导 + 形状空间多样性。
10. **`al4pde_full_hybrid`**：四维全能混合采样（不确定度 + 残差 + 分岔 + 多样性）。

*历史遗留策略（`rgs`, `mutant_aware`, `us`, `is`, `wrs`, `vessal`, `hggs`, `pial`）已降级为历史基准，详见 [docs/2_ACTIVE_LEARNING_GUIDE.md](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/docs/2_ACTIVE_LEARNING_GUIDE.md)。*

---

## 七、数据生成、归一化与生物网络图谱 (`data/`)

### 7.1 归一化规范与最新实测证据
- **轨迹 $Y$ 归一化**：**全局必须使用跨时间与样本的独立 Z-Score 归一化**（`mean_y` 与 `std_y` 形状 `(1, 39, 1)`）。
- **参数 $P$ 归一化**：**全局使用 Min-Max 归一化**映射至 $[0, 1]$（保留参数物理边界与敲除零点）。
- **最新实测证据（2026年9月基准）**：
  在 `mutant_aware_pure_random_100000_210m_500s/biom_refiner_plus/` 下的三种归一化对比实测证明：
  - **Standard Z-Score**：Real Rel L2 = **0.1518**, Real Corr = **0.9723**, Real MAE = **0.0869**（**综合断层领先最优**）。
  - **Asinh Scale**：Real Rel L2 = 0.2054, Real Corr = 0.9576。
  - **Asinh Z-Score**：Real Rel L2 = 0.1713, Real Corr = 0.9655。

### 7.2 数据集生成引擎清单
`data/` 下的所有数据生成与采样器已全部得到保护与留存：
- `generate_1m_pure_random_pool.py`：1M 纯随机参数候选池生成器。
- `gibbs_sobol_sampler.py` & `gibbs_boltzmann_sampler.py`：加扰 Sobol 与能量分布准随机采样。
- `bams_dataset_generator.py` & `mass_dataset_generator.py`：生物学感知与分层变异采样。
- `generate_kdpp_gated_dataset.py`, `generate_kdpp_curv_gated_dataset.py`, `generate_motif_topology_dataset.py` 等。
- `reachability_check.py`：可达性与零训练探针引擎（支持任意目录执行，默认输出至 `data/reachability/`）。

### 7.3 多关系生物因果拓扑图
文件 `data/yeast_multirelational_graphs.npz` 定义了出芽酵母 39 个核心分子之间的物理生化相互作用拓扑（包括转录激活、磷酸化/去磷酸化、蛋白质复合体形成、泛素化降解等多关系有向图矩阵）。

---

## 八、关键避坑指南与科学细节 (Gotchas & Best Practices)

1. **`PARAM_KEYS` 顺序绝对不可更改**：
   `al_callback.py`、`physics_eqns_torch.py`、`mass_dataset_generator.py` 均严格依赖 `sorted([k for k in _defaults.keys() if k != 'is_mutant_104'])` 得到的 141 维特征顺序。
2. **ODE BDF 积分器三层超时保护**：
   在调用 `solve_ivp(method='BDF')` 仿真复杂突变体时，病态参数可能导致底层 Fortran 循环挂起。必须保证：
   - 内置 `SIGALRM` 5秒硬超时中断；
   - Python `ProcessPoolExecutor` future 级 35 秒超时兜底；
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
