# 1. 出芽酵母神经算子代理模型系统架构与 70+ 模型全景 (Architecture & Model Zoo)

> **2026-10-19 运行链路修复**：通用基线构造显式映射 width→d_model、n_blocks→e_layers/n_layers 等，
> 不再依赖会吞参数的 `**kwargs`；旧 checkpoint 保留旧实际架构。Refiner/HANO 自回归训练与推断
> 走同一滚动轨迹，并保留粗预测/事件辅助监督。重复命名的基线架构、软注意力、本系统专属假设未重写。
> 范围、兼容性与测试见 [运行契约修复](RUNTIME_CONTRACT_FIX_2026-10-19.md)。

> **版本**：v3.0 (2026-10)  
> **核心定位**：出芽酵母细胞周期动力学代理模型的核心架构设计与模型全景字典。系统基于 **PyTorch Lightning** 框架，涵盖 40+ 经典/前沿基线（Baselines）与 30+ 专为细胞周期系统生物学定制的原创神经算子（Ours）。

> **2026-10-18 checkpoint 核验**：参考 `BioRefinerPlus` 的粗预测是
> `BioMechPathwayOnly`，未启用 NeuralEventTrigger；两个主力 checkpoint 的
> `graph_bias` 均为单位矩阵，未载入预期的 218 条跨变量图连接。
> 默认图路径从仓库根目录解析失败会静默回退。仍有可学习的稠密跨变量注意力，
> 不能把此发现误读为完全没有跨变量建模。Attractor/CausalReach 的 IC 布局与
> WT 表示空间缺陷及复现见 [当前证据审计](PURE_RANDOM_BADCASE_AUDIT_2026-10-18.md)。
> 上述为修复前证据。随后按用户要求修复了 IC 读取与 WT 特征表示，图保持不变，未重训；
> API、受影响模型及 checkpoint 迁移见 [IC/WT 修复说明](IC_WT_CONTRACT_FIX_2026-10-18.md)。

> **同日图来源统一第一阶段**：已新增由 `mutant_aware.py` ODE/事件源码生成的
> 统一潜在依赖图入口，218 条连续非自环边、事件新增 8 条、并集 226 条。
> 不编码永久正负号，功能模块图/映射仍单独保留。第一阶段尚未切换模型；
> 随后的**第二阶段已迁移单一 39×39 变量图消费者**，默认 hybrid，提供 current /
> checkpoint 两种固定图加载策略及来源指纹。六关系/27 模块图未替换，没有新增训练结果。
> 范围、参数量变化、调用示例和后续问题见 [模型图迁移](GRAPH_MODEL_MIGRATION_2026-10-18.md)。

> **2026-10-19 后续修复**：六关系消费者现使用同源 ODE 反应项覆盖层＋untyped 补充，
> 总支持与选定源码图一致（continuous 218 / hybrid 226 条非自环边），统一 target/source。
> DynamicPlus 27 模块图的邻接与符号方向也已修复；soft attention 保留，未新增可学习参数。
> 消费者清单、未使用图的例外、checkpoint 策略及验证见
> [关系图与模块方向修复](GRAPH_RELATION_MODULE_FIX_2026-10-19.md)。未重训或产生新的泛化指标。

---

## 目录
- [一、系统整体架构设计 (System Architecture)](#一系统整体架构设计-system-architecture)
  - [1.1 核心数据流与张量契约 (Tensor Contracts)](#11-核心数据流与张量契约-tensor-contracts)
  - [1.2 PyTorch Lightning 统一调度生命周期](#12-pytorch-lightning-统一调度生命周期)
  - [1.3 前向推理标准接口 `forward_ic_time`](#13-前向推理标准接口-forward_ic_time)
- [二、40+ 权威基线模型库 (`models/baselines/`)](#二40-权威基线模型库-modelsbaselines)
  - [2.1 经典与前沿神经算子 (Neural Operators)](#21-经典与前沿神经算子-neural-operators)
  - [2.2 时序 Transformer 与大模型结构](#22-时序-transformer-与大模型结构)
  - [2.3 连续动力系统与 Koopman 基线](#23-连续动力系统与-koopman-基线)
- [三、30+ 原创生物神经算子库 (`models/ours/`)](#三30-原创生物神经算子库-modelsours)
  - [3.1 BioRefiner 进化谱系 (BioRefinerPlus / Attractor / CausalReach)](#31-biorefiner-进化谱系-biorefinerplus--attractor--causalreach)
  - [3.2 HANO 多尺度分层自适应神经算子](#32-hano-多尺度分层自适应神经算子)
  - [3.3 物理通量与质量作用定律算子 (BioMassActionRefiner)](#33-物理通量与质量作用定律算子-biomassactionrefiner)
  - [3.4 生物异构图与谱图算子族](#34-生物异构图与谱图算子族)
- [四、模型注册表与扩展开发规范 (`models/yeast_lit_module.py`)](#四模型注册表与扩展开发规范-modelsyeast_lit_modulepy)

---

## 一、系统整体架构设计 (System Architecture)

系统采用模块化分层设计，解耦数据装载、模型定义、损失计算与主动学习回调：

```
                    ┌────────────────────────────────────────────────────────┐
                    │               YeastDataModule (数据流层)               │
                    │ - 默认 Zscore(log1p(Y))，逐通道统计                    │
                    │ - 默认 P 线性 Z-score (1, 141)                        │
                    │ - 突变体家族同进同出 + Pattern 稀有流形保护划分         │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                                ▼
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│                               YeastLitModule (LightningModule 核心层)                       │
│                                                                                            │
│   输入张量:                                                                                │
│     • X: (B, 39, 2, T)  [重复初值, 时间坐标]                                                │
│     • P (突变参数): (B, 141)    [141 维动力学常数]                                         │
│                                                                                            │
│   前向适配器 (forward_ic_time):                                                            │
│     • 构建时间网格 t_grid: (1, 1, 1, 500)                                                  │
│     • 拼接输入张量 ic_time_grid: (B, 39, 2, 500)                                           │
│     • 派发至 MODEL_REGISTRY 选定模型: forward(ic_time_grid, P)                              │
│                                                                                            │
│   输出张量:                                                                                │
│     • pred_norm: (B, 39, 500)  [归一化相空间连续轨迹]                                      │
└───────────────────────────┬────────────────────────────────────────────┬───────────────────┘
                            │                                            │
                            ▼                                            ▼
┌──────────────────────────────────────────────┐    ┌──────────────────────────────────────────────┐
│            utils/losses.py (损失计算)        │    │         utils/al_callback.py (主动学习)      │
│ - 17 种损失函数 (MSE, RelL2, GradNorm 等)    │    │ - 10 大生产级 AL4PDE 策略                    │
│ - 可微物理方程与软事件掩码约束              │    │ - 1M 纯参数候选池内存映射与在线 ODE 标注     │
└──────────────────────────────────────────────┘    └──────────────────────────────────────────────┘
```

### 1.1 核心数据流与张量契约 (Tensor Contracts)
- **输入张量 $X$**：实际 DataModule 输出 `(B,39,2,T)`；`X[:,:,0,:]` 是重复的标准化初值，`X[:,:,1,:]` 是时间坐标，第二项不是导数。初值应从原始 X 提取，不能取交错展平后的前 39 通道。
- **参数向量 $P$**：形状 `(B,141)`；默认 `log1p` 模式对 P 使用线性 Z-score，而非 Min-Max。因果模块的 WT 差分需经同一统计量还原至物理空间，普通学习分支仍使用原模型输入表示。
- **目标时序轨迹 $Y$**：形状 `(B,39,500)`；默认是 `Zscore(log1p(max(Y,0)))`，当前统计量来自全部合成样本，拟合拆分限制见数据审计。
- **预测输出 $\hat{Y}$**：形状 `(B,39,500)`；反归一化须匹配 `norm_type`。默认 log1p 模式需在反 Z-score 后执行 expm1，不能只做线性反变换。

### 1.2 PyTorch Lightning 统一调度生命周期
`YeastLitModule` 封装了标准的深度学习训练循环：
1. **`training_step(batch, batch_idx)`**：
   - 从 `batch` 中解包 `(x, p, y, *extra)`。
   - 调用 `pred = self.forward_ic_time(x, p)`。
   - 提取课程权重并计算 `loss = self.criterion(pred, y, ...)`。
   - 记录 `train_loss`, `train_mse`, `train_rel_l2` 等实时指标至 Wandb / 终端。
2. **`validation_step(batch, batch_idx, dataloader_idx)`**：
   - 系统内置 3 个评估 DataLoader（`dataloader_idx`: 0=合成验证集 `val_loss_synth`，1=合成测试集 `test_loss_synth`，2=真实突变体 `test_loss_real`）。
   - 以 `val_loss_synth` 作为 `ModelCheckpoint` 的核心监控指标（`save_top_k=1, mode='min'`）。
3. **`configure_optimizers()`**：
   - 采用 **AdamW** 优化器（初始学习率 $\eta_0 \approx 8.8 \times 10^{-4}$，权重衰减 $10^{-4}$）。
   - 采用 **CosineAnnealingLR + Linear Warmup** 调度器（前 8~10 epochs 线性预热，随后余弦退火至 $\eta_{\min} = 10^{-6}$）。

### 1.3 前向推理标准接口 `forward_ic_time`
为抹平不同模型架构（谱卷积、注意力、CNN、ODE）的输入接口差异，`YeastLitModule` 统一实现了 `forward_ic_time`：
默认 one_time 调度如下（省略特殊模型与自回归分支）；时间网格已经由 DataModule 构造，
不在此处把第二通道解释为导数或重新拼接。

```python
def forward_ic_time(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    # x: (B,V,2,T)，包含重复初值及时间；params 使用 DataModule 的 P 表示。
    return self.model(x, params)
```

---

## 二、40+ 权威基线模型库 (`models/baselines/`)

代码库在 `models/baselines/` 统一集成了 40 余种经典与前沿基线，所有基线均经过针对 $39 \times 500$ 动力学网格的适配优化：

### 2.1 经典与前沿神经算子 (Neural Operators)
| 模型代号 (`model`) | 类名 | 核心机制与架构 | 论文出处 |
|:---|:---|:---|:---|
| `pure_fno` / `fno1d` | `PureFNO_ICLR2021`, `FNO1d_Standard` | 1D 傅里叶神经算子，频域低频截断卷积。 | Li et al., ICLR 2021 |
| `pure_wno` | `PureWNO_2022` | 离散小波神经算子（DWT+IDWT），具备局部时频多分辨率特性。 | Tripura et al., 2022 |
| `cno1d` | `CNO1d` | 连续神经算子，基于下采样与连续卷积核。 | Raonic et al., 2023 |
| `mwt1d` | `MWT1d` | 多小波变换神经算子（Multiwavelet Transformer）。 | Gupta et al., 2021 |
| `deeponet` | `DeepONet` | 分支网络（Branch Net 处理参数）与主干网络（Trunk Net 处理时空坐标）点积。 | Lu et al., Nat. MI 2021 |
| `gcgnet` | `GCGNetYeast` | 门控因果图神经算子。 | 2024 |
| `gnot` | `GNOT` | 通用神经算子 Transformer，支持多参数交叉注意力。 | Hao et al., ICML 2023 |
| `transolver1d` / `transolver3_1d` | `Transolver1d`, `Transolver3_1d` | 物理注意力 Transformer 神经算子，计算隐式物理切片。 | Wu et al., 2024 |
| `dpot1d` | `DPOT1d` | 去噪预训练算子 Transformer。 | 2024 |
| `sc_fno` | `SC_FNO` | 空间-通道解耦傅里叶神经算子。 | SOTA 2025 |
| `bregman_fno` / `bregman_wno` | `BregmanFNO`, `BregmanWNO` | 基于 Bregman 散度正则化的能量守恒神经算子。 | SOTA 2025 |
| `ncflow` | `NCFlow` | 神经因果连续流算子。 | 2025 |
| `iganet_transformer` | `IgaNetTransformer` | 等几何分析（Isogeometric Analysis）增强 Transformer。 | 2025 |

### 2.2 时序 Transformer 与大模型结构
| 模型代号 (`model`) | 类名 | 核心机制与参数注入方式 | 论文出处 |
|:---|:---|:---|:---|
| `itransformer` | `iTransformer` | 倒置 Transformer（以 39 变量作为 Token，时间步作为通道），FiLM 注入参数。 | Liu et al., ICLR 2024 |
| `dlinear` / `nlinear` / `linear` | `DLinear`, `NLinear`, `Linear` | 趋势-周期分解单层线性投影基线。 | Zeng et al., AAAI 2023 |
| `autoformer` | `Autoformer` | 自动相关机制（Auto-Correlation）与时序分解块。 | Wu et al., NeurIPS 2021 |
| `informer` / `iinformer` | `Informer`, `iInformer` | ProbSparse 自注意力蒸馏机制。 | Zhou et al., AAAI 2021 |
| `flowformer` / `iflowformer` | `Flowformer`, `iFlowformer` | 基于流形注意力（Flow Attention）的高效时序 Transformer。 | 2022 |
| `reformer` / `ireformer` | `Reformer`, `iReformer` | 局部敏感哈希（LSH）注意力机制。 | Kitaev et al., ICLR 2020 |
| `flashformer` / `iflashformer`| `Flashformer`, `iFlashformer` | FlashAttention 极速注意力时序网络。 | 2023 |
| `vanilla_transformer` | `VanillaTransformer` | 标准时序编码器-解码器架构。 | Vaswani et al., 2017 |
| `fedformer` | `FEDformer` | 频域增强时序 Transformer（Frequency Enhanced Decomposition）。 | Zhou et al., ICML 2022 |
| `pyraformer` | `Pyraformer` | 金字塔注意力时序多尺度模型。 | Liu et al., ICLR 2022 |

### 2.3 连续动力系统与 Koopman 基线
| 模型代号 (`model`) | 类名 | 核心机制 | 论文出处 |
|:---|:---|:---|:---|
| `neural_ode` | `NeuralODEModel` | 连续深度伴随神经微分方程（采用 `torchdiffeq` 积分器）。 | Chen et al., NeurIPS 2018 |
| `konode` | `KoNODE` | Koopman 神经常微分方程（线性隐空间连续演化）。 | 2024 |
| `koopman_dmd` | `DeepKoopmanOperator` | 深度动态模态分解（Deep Dynamic Mode Decomposition）。 | Lusch et al., Nat. Comm. 2018 |
| `latent_lno` | `LatentNeuralOperator` | 隐空间 Laplace 神经算子。 | 2023 |
| `lamo` | `LaMO_RegularGrid` | 大规模多模态代理神经算子。 | 2024 |
| `wdno` | `WDNO` | 小波分解神经算子。 | 2023 |
| `icon` | `ICONGPT` | 算子上下文学习（In-Context Operator Learning）。 | Yang et al., NeurIPS 2023 |
| `noble` | `NOBLE` | 非线性算子双层学习引擎。 | 2024 |

---

## 三、30+ 原创生物神经算子库 (`models/ours/`)

针对细胞周期中高度刚性（Stiff）、多尺度事件跳跃（Events）与双稳态相变（Bifurcation）特性，本项目研发了 30 余种专用生物算子：

```
                             ┌───────────────────────────────────────┐
                             │       BioRefiner 谱系演进结构          │
                             └───────────────────┬───────────────────┘
                                                 │
                   ┌─────────────────────────────┼─────────────────────────────┐
                   ▼                             ▼                             ▼
      ┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
      │   BioRefinerPlus (SOTA) │   │   BioRefinerAttractor   │   │  BioRefinerCausalReach  │
      │ • 分层通量感知门控       │   │ • 吸引子平衡态投影锁定  │   │ • 因果可达性单向门控    │
      │ • 多关系因果图卷积      │   │ • 根除阻滞态伪振荡      │   │ • 严格时序因果掩码      │
      │ • 谱域高频残差细化      │   │ • 相空间能量流形收缩    │   │ • 避免未来信息逆流      │
      └─────────────────────────┘   └─────────────────────────┘   └─────────────────────────┘
```

### 3.1 BioRefiner 进化谱系
1. **`BioRefinerPlus` (`biom_refiner_plus`, [models/ours/bio_refiner_plus.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/models/ours/bio_refiner_plus.py))**：
   - **实际机制**：Stage 1 为 `BioMechPathwayOnly`（谱/时域特征与可学习旁路路由）；Stage 2 为 `ModularStiffRefiner`（导数/曲率特征、时间金字塔和 `StoichiometricGraphAttention`）。动态多关系通量图属于其他显式派生模型，不能混称。
   - **冻结结果**：hybrid 的 pooled Rel L2/Corr 为 **0.1518/0.9723**；triple_dynamic 的逐 mutant 平均为 **0.0741/0.8605**。口径不可混用，不能由这两项单独断言跨模型最优；详见当前审计。
2. **`BioRefinerAttractor` & `BioRefinerAttractorPlus` ([models/ours/bio_refiner_attractor_plus.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/models/ours/bio_refiner_attractor_plus.py))**：
   - **攻关目标**：彻底解决不可逆阻滞突变体（Arrested Mutants）在晚期出现的 17 次虚假高频波浪伪振荡。
   - **数学原理**：在网络尾部引入非线性阻尼吸引子流形映射：
     $$\hat{y}(t) = (1 - \alpha(P)) \cdot \hat{y}_{\text{osc}}(t) + \alpha(P) \cdot \mathbf{y}_{\infty}(P)$$
     其中 $\alpha(P) \in [0, 1]$ 由参数网络自适应判定是否进入稳态吸引子。
3. **`BioRefinerCausalReach` & `BioRefinerCausal` ([models/ours/bio_refiner_causal_reach.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/models/ours/bio_refiner_causal_reach.py))**：
   - **攻关目标**：解决传统双向谱卷积带来的未来 336 步信息反向泄露至过去的非物理因果破缺。
   - **实现机制**：强制所有时域卷积使用单向因果掩码（Causal Dilated Conv1d），保证 $t$ 时刻状态严格仅由 $\tau \le t$ 决定。

### 3.2 HANO 多尺度分层自适应神经算子 ([models/ours/hano.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/models/ours/hano.py))
- **设计思想**：Hierarchical Adaptive Neural Operator。将细胞周期轨迹分解为宏观质量增长（低频大时间尺度）、周期素主振荡（中频极限环）与分裂检验点瞬态触发（高频突变）三个层次。
- **消融变体**：`hano_no_event`（无事件支路）、`hano_no_flux`（无通量支路）、`hano_no_graph`（无图拓扑支路）、`hano_refine`。

### 3.3 物理通量与质量作用定律算子 (BioMassActionRefiner)
- **`BioMassActionRefiner` ([models/ours/bio_mass_action_refiner.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/models/ours/bio_mass_action_refiner.py))**：
  直接在隐层编码化学计量矩阵（Stoichiometric Matrix $S$）与反应速率向量 $v(u, p)$，强制网络输出满足物质守恒约束：
  $$\frac{du}{dt} = S \cdot v(u, p) + \text{NeuralCorrection}(u, p)$$

### 3.4 生物异构图与谱图算子族
- `HeteroGraphResNetFiLM` / `GatedGraphResNetFiLM`：结合 39 变量生物邻接矩阵的 FiLM 特征调制网络。
- `MultiScaleDiffGraphNet`：多尺度微分图神经网络。
- `DualPathSpectralGraphNet`：时域图卷积与频域谱图卷积双通道并行解耦网络。
- `ChannelHyperFNO` / `CrossFNO`：通道级超网络与多头交叉注意力神经算子。

---

## 四、模型注册表与扩展开发规范 (`models/yeast_lit_module.py`)

所有模型均由 [models/yeast_lit_module.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/models/yeast_lit_module.py) 中的 `MODEL_REGISTRY` 统一管理。

### 4.1 新增模型开发标准三步法
当 AI Agent 或研发人员需要引入新的神经网络模型时，**必须严格执行以下三步**：

1. **第一步：实现模型类**
   - 基线模型放入 `models/baselines/xxx.py`，原创模型放入 `models/ours/xxx.py`。
   - 必须实现 `forward(ic_time_grid, params)`，接收 `(B, 39, 2, 500)` 与 `(B, 141)`，返回 `(B, 39, 500)`。
2. **第二步：在 `models/yeast_lit_module.py` 中注册**
   - 顶部引入类名：`from budding_yeast_v2.models.ours.xxx import MyNewOperator`
   - 将其字符串代号加入 `MODEL_REGISTRY`：
     ```python
     MODEL_REGISTRY = {
         ...
         'my_new_op': MyNewOperator,
     }
     ```
   - 在 `YeastLitModule.__init__` 中补充参数初始化分支。
3. **第三步：编写单元冒烟测试**
   - 在 `tests/` 下新建 `test_my_new_op.py`，测试 `x = torch.randn(2, 39, 2)` 和 `p = torch.randn(2, 141)` 的前向传播与反向梯度，验证无 NaN 且输出形状严格为 `(2, 39, 500)`。
