> **2026-10-31 AL辅助模型**：PatternClassifier为选样专用ExtraTrees，代码在`utils/active_learning/auxiliary/pattern_classifier.py`，不替换FNO或加入主模型选择。每轮保存joblib及train-only/OOF说明。[实现](AL_MECHANISM_IMPLEMENTATION_2026-10-31.md)。

# 1. 出芽酵母神经算子代理模型系统架构与 标准基线与原创模型全景 (Architecture & Model Zoo)

# 2026-10-30：标记物闭合消融与后续研究设计

新增 `biom_refiner_marker_closure`（`models/ours/bio_refiner_marker_closure.py`），继承
BioRefinerPlus，保持输入 `(B,39,2,T)`、P `(B,141)` 和输出 `(B,39,T)`。
在物理空间用已知线性驱动方程产生 ORI/BUD/SPN，以每次方向性交叉重置，使用并行 affine scan。
其余36个输出保持原样。需要通过 DataModule/checkpoint 的 `set_stats` 绑定统计量；支持四种归一化。
width192/modes24/n4 时与 Plus 同为9,321,111个可训练参数元素（复数按numel计1）。
它是参数无增加的读出消融，尚未实现完整四事件闭环或条件干预训练；无正式重训性能结论。
15项相关回归和真实ODE小数据2步bf16训练/checkpoint重载验证通过。
架构、正式训练调用命令、54份历史配置/20例分析及 ICHFO 设计见
[完整设计与证据](OPERATOR_DESIGN_AND_BADCASES_2026-10-30.md)。

> **2026-10-21 标准 baseline 清理**：`models/baselines` 只保留选定的标准骨干和必要的参数/初值/时间输入输出适配。
> 入口统一为 `mlp, resnet, unet, kan, transformer, mamba, deeponet, fno, wno, mwt, gnot, transolver, lno, neural_ode, sc_fno, bregman_fno, lamo, konode`。
> 旧调制版、时序预测版和简化同名版已删除，不保留兼容别名；下方较早的 checkpoint 兼容说明仅代表历史状态。
> 新增标准一维 U-Net；完整 CLI/调参/训练/checkpoint 验证见 [链路核查](BASELINE_CLI_CHAIN_AUDIT_2026-10-21.md)。
> 来源、适配范围及 SC-FNO 数据要求见 [当前基线清单](BASELINE_SELECTION_AUDIT.md)。

> **2026-10-21 Fission监控复核**：原BIO/pattern保持不变；无pattern训练目标用-1并跳过分类交叉熵，
> W&B按Fission自身BIO/pattern选样，显示16变量/300分钟，不套用Budding的监控编号或六分类。
> 系统标签、checkpoint/评测及HPO兼容边界见 [Fission扩展](FISSION_EXTENSION.md)。


> **2026-10-20 Fission扩展**：共享基线和通用图模型可使用16变量/58参数，训练、调参、checkpoint、
> 评测按系统选择维度与图。8种架构已做前向/反向验证；Budding专用生物结构仍明确限制。
> 模型范围、命令与验证边界见 [Fission扩展](FISSION_EXTENSION.md)。

> **2026-10-19 运行链路修复**：通用基线构造显式映射 width→d_model、n_blocks→e_layers/n_layers 等，
> 不再依赖会吞参数的 `**kwargs`；旧 checkpoint 保留旧实际架构。Refiner/HANO 自回归训练与推断
> 走同一滚动轨迹，并保留粗预测/事件辅助监督。重复命名的基线架构、软注意力、本系统专属假设未重写。
> 范围、兼容性与测试见 [运行契约修复](RUNTIME_CONTRACT_FIX_2026-10-19.md)。

> **版本**：v3.0 (2026-10)  
> **核心定位**：出芽酵母细胞周期动力学代理模型的核心架构设计与模型全景字典。系统基于 **PyTorch Lightning** 框架，涵盖选定的 18 个标准基线实现（Baselines）与 30+ 专为细胞周期系统生物学定制的原创神经算子（Ours）。

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
- [二、标准基线模型库 (`models/baselines/`)](#二40-权威基线模型库-modelsbaselines)
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

## 二、标准基线模型库 (`models/baselines/`)

当前只保留 18 个具体实现（MLP / ResNet 分别计数），均不添加额外的参数调制、图先验或跨注意力。
GNOT 原有的异构注意力/专家门控、Transolver 原有的物理注意力等属于原模型自身机制，予以保留。
所有模型使用完整参数及初值；标准名称不再带 pure / ref / 1d 等重复后缀。

| 入口 | 源文件 | 保留的核心机制 |
| --- | --- | --- |
| `mlp` | `mlp.py` | 全初值、参数、时间拼接后 MLP |
| `resnet` | `resnet.py` | 同输入的普通残差 MLP |
| `unet` | `unet.py` | 一维双卷积、池化、转置卷积和 skip 拼接 |
| `kan` | `kan.py` | efficient-kan 的 B-spline 边函数，逐时间点条件回归 |
| `transformer` | `transformer.py` | Tau_BNO 多头自注意力 encoder 及逐 token 回归 |
| `mamba` | `mamba.py` | Tau_BNO 回归骨干及官方 Mamba-1 选择性状态空间 |
| `deeponet` | `deeponet.py` | branch 编码初值和参数，trunk 编码时间 |
| `fno` | `fno.py` | 谱卷积 + 点卷积 |
| `wno` | `wno.py` | 最粗 low/detail 系数位置核 + 点卷积 |
| `mwt` | `mwt.py` | 多项式滤波、递归 A/B/C 和重建 |
| `gnot` | `gnot.py` | 官方异构归一化注意力及几何专家 |
| `transolver` | `transolver.py` | 官方 slice -> slice-token attention -> deslice |
| `lno` | `lno.py` | 官方 PhCA 编码/解码与潜空间注意力 |
| `neural_ode` | `neural_ode.py` | 条件向量场和数值积分 |
| `sc_fno` | `sc_fno.py` | 同 FNO 骨干 + 参考灵敏度训练损失 |
| `bregman_fno` | `bregman_fno.py` | Bregman 双状态后激活更新 |
| `lamo` | `lamo.py` | slice / Hydra / deslice，未融合 SSD 计算后端 |
| `konode` | `konode.py` | Koopman 模式驱动演化向量场权重 |

新增三类模型的输入输出、固定核心配置、参数量及 Mamba 后端边界见 [适配说明](OPERATORFREE_BASELINES_2026-10-21.md)。`width=64,n_blocks=4,V=39,P=141` 时，KAN / Transformer / Mamba 分别为 **263,680 / 171,687 / 242,343** 个参数。

U-Net 使用 `[全部 y0, 全部 P, t]` 作为输入通道，输出所有变量轨迹。`width=64, n_blocks=4, V=39, P=141` 时共 **10,849,383** 个参数；`n_blocks` 是下采样/上采样层数。时间长度补齐至 `2**n_blocks` 的倍数并在输出裁回，不添加 FiLM、注意力或图模块。

`X:(B,V,2,T)` 的特征 0 是重复初值，特征 1 是时间；`P:(B,D)`；输出 `(B,V,T)`。
训练入口传递真实 T 给 WNO，支持对应分解层数允许的网格长度。FNO/MWT/WNO 要求均匀查询。

```bash
python train.py --model fno --dataset_name /absolute/path/to/dataset.npz --loss_type hybrid_mse_rel --al_strategy none --width 64 --modes 24 --n_blocks 4
python train.py --model unet --dataset_name /absolute/path/to/dataset.npz --loss_type hybrid_mse_rel --al_strategy none --width 32 --n_blocks 3
python train.py --model kan --dataset_name /absolute/path/to/dataset.npz --width 64 --n_blocks 4 --loss_type hybrid_mse_rel --al_strategy none
# 将 kan 替换为 transformer / mamba，参数不变。
# 将 fno 分别替换为 wno / mwt / gnot / transolver / lno 等。
python train.py --model sc_fno --dataset_name /absolute/path/to/dataset.npz --loss_type sc_sensitivity --sensitivity_path /absolute/path/to/reference_sensitivities.npz --al_strategy none
```

SC-FNO 的监督文件、归一化指纹及初值固定偏导要求见审计文档；没有参考灵敏度时明确拒绝训练，
不能自动退化为普通 FNO 后继续以 SC-FNO 名称报告。当前 SC-FNO 参考数据路径只支持 one_time、无 AL。

原 ours 的 WNO/LNO 入口改为 `cross_wno` / `cross_lno`，避免与标准 `wno` / `lno` 混淆。
旧版 checkpoint 不自动转换到标准骨干，旧实验指标也不自动改名为新模型结果。

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
