# Budding Yeast v2 - Architecture & Framework Overview

该文档用于梳理 `budding_yeast_v2` 目前的整体架构，澄清在融合了旧版基线（budding_yeast_v2）和新版主动学习框架（AL4PDE）之后的目录结构、代码模块及其各自的作用。

## 1. 整体架构概览 (High-level Architecture)
本项目目前融合了两套核心范式：
- **经典训练范式 (PINN / FNO Base)**：直接使用预先生成好的大规模静态数据集（如 LHS 采样产生），训练算子模型预测细胞周期系统，并通过 PINN 物理残差和事件重置约束 (Jump Condition) 来约束物理合规性。
- **主动学习范式 (AL4PDE AR+AL)**：通过少量初始数据训练一个自回归模型 (AR Model)，然后利用模型对“候选池 (Candidate Pool)”进行大批量推理，利用 Acquisition Function（多样性、不确定性、距离等策略）挑选最有价值的数据，交由真实 ODE 模拟器生成 Ground Truth，再并入训练集。不断循环以用极小的数据量达到极高的泛化能力。

---

## 2. 目录结构梳理 (Directory Structure)

```text
budding_yeast_v2/
├── configs/                     # 【配置中心】所有超参数的 yaml 存放地
│   └── ar_al_default.yaml       # 专门为 AR (自回归) + AL (主动学习) 准备的配置文件 (移植自 AL4PDE)
├── data/                        # 【数据与生成层】
│   ├── dataset_utils.py         # 数据集读取与切分工具类 (如通用加载、分层采样划分)
│   ├── param_gen.py             # 核心参数采样逻辑
│   ├── lhs_v0.py / lhs_v1.py    # 拉丁超立方抽样 (LHS) 脚本，包含严格的生物学合理性拦截 (Hard Conflict Rejection)
│   ├── lhs_v1_origin.py         # 原始版本 LHS 逻辑备用
│   ├── yeast_datamodule.py      # 旧版 PyTorch Lightning 数据模块，用于加载静态大规模数据集
│   └── ar_datamodule.py         # 新版！AL4PDE 风格，专为将长轨迹切分为短自回归 step 的数据模块
├── models/                      # 【模型与损失层】(最核心模块)
│   ├── components/              # 底层网络结构
│   │   ├── neural_ode.py        # 基于 Neural ODE 的对照基线
│   │   ├── hyper_fno.py         # Hyper-FNO (参数化频域神经算子)
│   │   ├── hp_fno.py            # HPO-FNO 架构
│   │   ├── patchtst.py          # PatchTST 时序预测架构
│   │   ├── cross_fno.py, lno_cross.py, wno_cross.py, pure_fno.py, ultimate_fno.py # 其他各类型的神经算子变体
│   │   └── al4pde_modules/      # 移植自 AL4PDE 的底层模块 (如 fno.py, sinenet_cond_2d.py, unet_cond_1d.py 等)
│   ├── yeast_lit_module.py      # 旧版及基础训练框架：基于 PyTorch Lightning 封装，支持单次 rollout 训练、自回归滚动，和 Curriculum Learning。
│   ├── ar_model.py              # 新版！移植自 AL4PDE 的 ARModelWrapper，不使用 Lightning，专门支持多步自回归的循环训练。
│   └── ar_loss.py               # 新版！自回归专用 Loss，支持 Teacher Forcing Schedule 逐步过渡到纯 AR。
├── acquisition/                 # 【主动学习策略层】(新版新增)
│   ├── al4pde_acquisition/      # 完全移植自 AL4PDE 的主动学习核心算法
│   │   ├── pool_based.py        # 基础的 Pool-based 采样框架
│   │   ├── random.py            # 随机采样策略 (Base)
│   │   ├── uncertainty_based.py # 基于模型不确定性的采样 (Ensemble Uncertainty)
│   │   ├── distance_pool_based.py # 基于特征空间或轨迹距离的多样性采样策略
│   │   ├── batch_selection.py   # Batch 级别的综合筛选策略
│   │   ├── build_selection.py   # 策略构建工厂
│   │   └── data_schedule.py     # 主动学习的采样数量调度计划
│   ├── lcmd_acquisition.py      # LCMD 多样性采集策略
│   ├── pool_based_strategies.py # 额外整合的主动学习策略封装
│   └── random_selection.py      # 独立随机采样策略
├── utils/                       # 【工具与约束层】
│   ├── al_callback.py           # 【极度重要】不仅包含主动学习的控制回调，还集成了极其重要的防崩溃校验 (RGS), PCA 降维多样性评估，以及 NaN/Inf 防御。
│   ├── losses.py                # 物理损失计算库，包含 PINNResidualLoss v2 (采用 Ground Truth 事件掩码，以及基于真实变化方向的跳跃惩罚)。
│   ├── metrics.py               # 模型性能评估指标
│   ├── normalization.py         # 归一化/反归一化模块
│   └── physics_eqns_torch.py    # 基于 PyTorch 的细胞周期 ODE 物理方程
├── scripts/                     # 【新版运行脚本层】
│   ├── run_ar_al.py / .sh       # 新版！用来启动 Auto-Regressive + Active Learning 主流程的脚本
│   └── eval_ar_al.py            # AR 模型专用的评估脚本
├── 根目录脚本 /                    # 【旧版运行与工具脚本】
│   ├── train.py                 # 旧版主训练入口，对接 yeast_lit_module 和静态数据集
│   ├── train_eval_filtered.py   # 过滤数据集后的训练入口
│   ├── eval.py                  # 旧版评估指标和画图脚本
│   ├── tune.py / run_tune.sh    # 超参数搜索脚本
│   ├── run_pipeline.sh          # 旧版/基线跑全量训练的脚本
│   ├── run_filtered_experiment.sh # 旧版过滤实验启动脚本
│   └── analyze_data_distribution.py / analyze_seeds.py # 数据探索分析工具
└── Markdown 文档                 # 其他相关开发记录文档 (如 ALGORITHMS_AL.md 等)
```

---

## 3. 旧版 (基线) vs 新版 (AL4PDE) 工作流对比

为了防止在调参时混淆，请明确两套流程的文件调用链是完全分开的：

### 工作流 A：经典基线训练 (Old / Base Framework)
**定位**：在大规模生成的静态训练集上拟合，验证模型拟合物理系统的极限能力。
- **入口**：`train.py` 
- **数据**：`data/yeast_datamodule.py` 负责读取整个预生成好的大池子，借助 `data/dataset_utils.py` 进行划分。
- **模型**：`models/yeast_lit_module.py`，支持一次性前向输出或自回归步进。采用 PyTorch Lightning 的 Trainer 训练。
- **损失**：`utils/losses.py`，受 Curriculum Learning 自动调整（前 30 轮纯 MSE，后慢慢加入物理损失）。

### 工作流 B：自回归 + 主动学习 (AL4PDE AR+AL Framework)
**定位**：解决数据生成昂贵的问题。用小数据起步，让模型自动寻找对自己最有价值的数据。
- **入口**：`scripts/run_ar_al.py` (配置：`configs/ar_al_default.yaml`)
- **数据**：使用 `data/ar_datamodule.py` 将长轨迹切片成短块 (例如 `num_train_steps=2`)。
- **模型**：使用纯 PyTorch 写的 `models/ar_model.py` (绕开 Lightning)。为了防止早期自回归发散，引入了 `Teacher Forcing Schedule`，在前 50% 的 epoch 中将 Ground Truth 喂给下一步。
- **主动学习交互**：
  1. 生成极大规模候选池 (如 10,000 * Mutants 种参数)，通过 `utils/al_callback.py` 中的 `_validate_param_ranges` 拦截掉荒谬的参数（防溢出）。
  2. 使用 `ar_model.py` 对候选池进行全序列推理。遇到崩溃的残差时，使用 `np.nan_to_num` 防御。
  3. 计算多样性：使用 `_compute_trajectory_diversity_pca` 将推理结果降维到 20 维，然后计算到现有训练集的欧式距离，选出差异最大的。
  4. 选出的候选参数调用 ODE 模拟器获得真实 Ground Truth，然后加入到下一轮 AL 的训练池中。
  5. 循环该过程 (5 轮)。

---

## 4. 重点注意的修改与细节 (Review Notes)

1. **防爆栈 (Defense Mechanisms)**
   目前两套模型中都内置了强大的防溢出约束，这在训练 ODE 的代理模型时是必须的，请在修改或排查 bug 时不要去掉它们：
   - AL Callback 中的参数上限拦截，避免了求解器由于 `dt` 过小而陷入死循环或算出 NaN。
   - `ar_loss.py` 中的 `teacher_forcing`，完美解决了自回归模型一开始瞎猜导致输入序列爆炸的问题。
2. **物理约束 (PINN) 的准确度**
   `losses.py` 里的事件检测现在是绑定的**真值 (Ground Truth)**（`detect_event_mask_torch(target_real)`），这代表模型的物理 Loss（比如 Jump Condition）现在被精确地约束在了真实发生突变的时刻，极大地提升了约束有效性。
3. **PCA 降维对于 AL 的重要性**
   如果在 `budding_yeast` 这样一个拥有 39 变量 x 500 步高维特征的场景下不使用 PCA 直接求距离，会导致**距离趋同现象** (即高维空间下任意两点距离几乎相同)，主动学习将完全失效退化为随机采样。目前的 PCA 保留了 95% 以上的方差，请放心使用。
