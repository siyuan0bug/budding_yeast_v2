# 2. 出芽酵母主动学习全景指南与 AL4PDE 闭环体系 (Active Learning Guide)

> **2026-10-22 Budding 合成突变目录精简**：删除 TEM1/CDC15 的 DAmP，以及 CDC5_Proxy、SCC1、CDC28、SBF_MBF、PPX、Mcm1、ORI_BUD_SPN 七个合成模块。当前为 19 个基因模块 + MDT 环境模块，共 65 种模式组合；拓扑、motif、WT-Blind、救援与 AL 候选审计同步更新。保留组合的两套赋值/分布保持一致，39 变量、141 参数、ODE、初值和 126 real 不变；Fission 不变。含已删除突变的旧候选池或派生源会拒绝继续生成，需重新生成；旧 NPZ 不改写，仍可用于原有训练/分析。


> **2026-10-22 当前采样已更新**：Budding/Fission 静态配额 10/40/50、AL 候选池 5/40/55，仅一至三模块。下文较早的四/五层配额属于历史描述；新规则、输出字段和失败处理见 [三模块采样说明](EXACT_THREE_MODULE_SAMPLING.md)。

> **Fission 已适配五种通用 AL 策略**：Random、Uncertainty、Power、LCMD、BAIT 共用 `train.py` 闭环。候选先做原生逐株预热，选中后生成完整轨迹；原 BIO/Pattern/缺失标签保持不变，其他 Budding 专用策略仍拒绝。命令、初值成本和验证范围见 [Fission AL](FISSION_ACTIVE_LEARNING.md)。

> **模块整理已完成**：生产实现移至 `utils/active_learning/`，旧 `al_callback.py` 保留兼容入口；15 种策略和别名由 registry 统一管理。初始化元数据对齐与跨轮缓存问题分别修复，HPO 不再静默挂载无效 callback。当前接口和验证见 [拆包说明](AL_PACKAGE_REFACTOR.md)。

> **Power / BAIT / LCMD 已接入**：新增 `al4pde_power`、`al4pde_bait`、`al4pde_lcmd`，273 项回归通过；LCMD 使用代理预测特征，不要求预仿真。latent coreset 本轮仅作接口设计说明，未修改模型。官方实际 8 个预置配置的逐项对应、ODE 特征适配和命令见 [Power/BAIT 与官方清单](AL_POWER_BAIT_AND_OFFICIAL_INVENTORY.md)。

> **2026-10-21 旧轨迹池策略归档**：`random/us/is/wrs/vessal/hggs/pial` 已退出训练入口；完整快照保存在 `utils/archive/al_callback_legacy.py`，主文件删除 19 个旧专属方法。当前保留 15 种在线策略及别名（新增 Power、BAIT、LCMD）。
> **Uncertainty 后续对齐**：当前默认两个独立训练成员，同一输入按官方默认 MSE disagreement + mean + Top-K 选样；支持官方 raw sample variance 分支，停止使用参数加噪。Random 保持固定池随机选。240 项测试通过，见 [Ensemble 对齐](AL_ENSEMBLE_UNCERTAINTY.md)。尚未直接实例化官方全套 Task/ProbModel；以下历史“十大策略”章节也不能作为当前 API 清单。
> 当前变更、真实算法清单对应关系、231 项验证及待修差异见 [归档与官方核对](AL_ARCHIVAL_AND_OFFICIAL_ALIGNMENT_2026-10-21.md)。


> **2026-10-21 标签接入**：保留BIO和ABC，六分类以中性补充分组保存到未来AL快照，暂不参与选样或替换模型头。
> 在线仿真原BIO不再丢弃；历史快照缺失BIO时明确标记未知。修复追加样本的ABC训练ID全为A的占位，
> 并在直接建初始集时传入实际ABC目标。其余策略、筛选、配额、归一化规则不变；历史结果未重跑。
> ABC对坏例诊断偏粗，但六分类是否能改善训练/补采仍需对照实验，见 [数据标签接口](4_DATASETS_AND_NORMALIZATION.md)。


> **新旧Pattern标签的AL实测比较**：7种完成第2轮的策略、25个训练快照表明，六分类有助于区分
> 绝对误差风险，但对relL2风险变差；保留BIO后替换pattern的综合优势尚不确定。
> 新临界标记未显示附加收益，历史AL算法也未使用该新标记。该比较阶段仅分析，未改策略或训练；后续中性标签接入见上方。
> 方法、稳定坏例与覆盖证据见 [标签比较](AL_PATTERN_LABEL_COMPARISON.md)。

> **2026-10-19 后续归一化边界**：新训练统计量仅在源 NPZ 的合成 train 拆分拟合，
> 后续 AL 轮次/入库继续复用，不含合成 val/test 或 real。没有改为每轮只在已标注 AL 子集重新拟合，
> 也未修改初始子集、候选筛选或 real-envelope 规则；不能据此声称完整的 AL 标签盲态。
> 详见 [train-only 修复](TRAIN_ONLY_NORMALIZATION_2026-10-19.md)。

> **2026-10-19 工程修复**：al4pde 的预存池加载、即时构池、标注入库统一使用
> `data/normalization.py` 编码 IC/P/Y，支持全部四种模式；在线 MASS 求解拒绝失败/截断解，
> 父进程等待带 watchdog，按原候选顺序消费结果，配额提前结束也有受限清理。
> 用户明确要求**保留所有筛选规则**（包括 real-envelope/nominal-flux 门控），不修改算法或阈值，
> 因此不能将本轮修复称为零泄漏或完整 AL 状态恢复。见 [运行契约修复](RUNTIME_CONTRACT_FIX_2026-10-19.md)。

> **版本**：v3.0 (2026-10)  
> **核心定位**：出芽酵母代理模型主动学习（Active Learning, AL）全景技术指南。以 [utils/al_callback.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py) 为唯一代码真理源，系统梳理历史经典策略的演进脉络，深度剖析现代 **AL4PDE 闭环架构与 10 大生产级主动学习算法**。

---

## 目录
- [一、主动学习演进脉络与范式跃迁](#一主动学习演进脉络与范式跃迁)
  - [1.1 历史经典策略与局限性简析 (RGS / US / PIAL 等)](#11-历史经典策略与局限性简析-rgs--us--pial-等)
  - [1.2 现代 AL4PDE 范式跃迁：1M 免预仿真纯参数空间候选池](#12-现代-al4pde-范式跃迁1m-免预仿真纯参数空间候选池)
- [二、AL4PDE 闭环系统架构与数据流](#二al4pde-闭环系统架构与数据流)
  - [2.1 候选池内存映射与生命周期 (`ALPoolManager`)](#21-候选池内存映射与生命周期-alpoolmanager)
  - [2.2 多进程 ODE 在线高速标注与三层防死锁机制](#22-多进程-ode-在线高速标注与三层防死锁机制)
  - [2.3 训练节奏控制与增量 DataLoader 动态重构](#23-训练节奏控制与增量-dataloader-动态重构)
- [三、10 大生产级 AL4PDE 主动学习策略详解](#三10-大生产级-al4pde-主动学习策略详解)
  - [3.1 `al4pde_random` (无偏高维均匀基线)](#31-al4pde_random-无偏高维均匀基线)
  - [3.2 `al4pde_uncertainty` (MC-Dropout 预测不确定度)](#32-al4pde_uncertainty-mc-dropout-预测不确定度)
  - [3.3 `al4pde_diversity` (332 维时序形状感知 + GPU 向量化 K-Center)](#33-al4pde_diversity-332-维时序形状感知--gpu-向量化-k-center)
  - [3.4 `al4pde_residual` (流速自适应相对生化物理残差)](#34-al4pde_residual-流速自适应相对生化物理残差)
  - [3.5 `al4pde_res_div` (残差-多样性双目标帕累托前沿融合)](#35-al4pde_res_div-残差-多样性双目标帕累托前沿融合)
  - [3.6 `al4pde_entropy` (双稳态相空间转移概率香农熵)](#36-al4pde_entropy-双稳态相空间转移概率香农熵)
  - [3.7 `al4pde_bifurcation` (生物学分岔指标引导采样)](#37-al4pde_bifurcation-生物学分岔指标引导采样)
  - [3.8 `al4pde_bifurcation_entropy` (分岔强度与动力学不确定度联合熵)](#38-al4pde_bifurcation_entropy-分岔强度与动力学不确定度联合熵)
  - [3.9 `al4pde_bifurcation_diversity` (分岔边界引导 + 形状空间多样性)](#39-al4pde_bifurcation_diversity-分岔边界引导--形状空间多样性)
  - [3.10 `al4pde_full_hybrid` (四维全能混合采样)](#310-al4pde_full_hybrid-四维全能混合采样)
- [四、生化物理残差计算与剪枝引擎 (`ode_residual_engine.py`)](#四生化物理残差计算与剪枝引擎-ode_residual_enginepy)
  - [4.1 7 大极快变量剪枝机制](#41-7-大极快变量剪枝机制)
  - [4.2 通量归一化门控因子 $L_{\text{flux}}$](#42-通量归一化门控因子-l_textflux)
- [五、实验配置与快速启动指南](#五实验配置与快速启动指南)

---

## 一、主动学习演进脉络与范式跃迁

在出芽酵母细胞周期代理模型中，主动学习的核心使命是：**以最小的 ODE 数值求解计算开销（标注成本），挑选最具信息量的参数扰动样本加入训练集，使神经算子在 126 个未见过的真实突变体（Real Mutants）上达到最佳泛化精度。**

```
                  ┌────────────────────────────────────────────────────────┐
                  │ 历史初代 AL 探索 (已淘汰)                              │
                  │ - RGS (利用 Real 测试集残差定向扰动，存在 Data Leak)   │
                  │ - US / IS / VeSSAL / HGGS (基于预仿真静态小数据池)    │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │ 现代 AL4PDE 体系 (v2.2 生产级基准)                     │
                  │ - 1M 纯参数空间候选池 (零预仿真，无标签，0% 泄露)      │
                  │ - 模型前向打分 (残差/不确定度/分岔/多样性)              │
                  │ - 极速 Top-K 选拔 + 多进程 ODE 批量在线求解扩容        │
                  └────────────────────────────────────────────────────────┘
```

### 1.1 历史经典策略与局限性简析
在项目早期（v1-v6 时代），曾探索过以下几类策略：
1. **查询合成策略（Query Synthesis: RGS, Mutant-Aware）**：
   - *机制*：在评估集上计算各突变体的 MAE，针对 MAE 大于阈值的突变体，在其 141 维参数邻域内施加高斯微扰生成新参数，并调用 ODE 求解器。
   - *致命缺陷*：**测试集数据泄露（Data Leak）**。该方法实质上利用了未见测试集的反馈来合成训练集，无法通过严苛的学术同行评审。
2. **静态池化策略（Pool-based: US, IS, WRS, VeSSAL, HGGS, PIAL）**：
   - *机制*：预先使用 ODE 求解器仿真 50,000 个静态样本构成候选池，通过 MC-Dropout 方差（US）或流形几何覆盖（HGGS）从中抽样。
   - *致命缺陷*：需要耗费数十小时预先仿真全量轨迹，失去了主动学习“按需标注以降低求解成本”的本质意义。

### 1.2 现代 AL4PDE 范式跃迁：1M 免预仿真纯参数空间候选池
现代 AL4PDE 体系彻底解决了以上痛点：
- **候选池纯净化**：由加扰 Sobol / 准随机采样生成 1,000,000 组纯参数向量 $P \in \mathbb{R}^{1000000 \times 141}$ 与初始条件 $X$。**整个池子没有任何预仿真的时序轨迹 $Y$**。
- **零测试集接触**：采集函数（Acquisition Function）严格仅依赖模型自身的预测不确定度、时序形状多样性、生化物理方程残差以及生物学分岔指标。
- **在线按需标注**：每轮主动学习选出最优的 $K$ 个样本后，后台多进程瞬间完成这 $K$ 个样本的 ODE 积分并扩充进当前训练集。

---

## 二、AL4PDE 闭环系统架构与数据流

```
       ┌────────────────────────────────────────────────────────────────────────┐
       │                 1,000,000 纯参数空间候选池 (Memory Mapped)             │
       └───────────────────────────────────┬────────────────────────────────────┘
                                           │ 分批加载 (Batch Inference)
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                              当前训练中的神经网络模型 (BioRefiner)                           │
│                                                                                              │
│   1. 前向预测时序轨迹:  \hat{Y}_i = \text{Model}(X_i, P_i)                                   │
│   2. 抽取相空间动力学特征: 332-D 特征向量 (MASS 包络 + 一阶导数 + 统计量)                    │
│   3. 计算物理生化残差:   \mathcal{R}(P_i, \hat{Y}_i)                                         │
│   4. 评估分岔熵/不确定度: \mathcal{H}_{\text{bif}}(P_i), \sigma^2(P_i)                       │
└──────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                            采集函数 Top-K 选拔 (Acquisition Engine)                          │
│          根据选定策略 (如 al4pde_bifurcation_entropy) 计算综合打分并选取 Top-K 样本          │
└──────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                       多进程 ODE 数值求解器 (Online Parallel solve_ivp)                      │
│     并发 32~40 Workers 积分选出的 K 个参数 -> 获取真值轨迹 Y_new -> 拼入 train_dataset_lhs   │
└──────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                           │
                                           ▼
                             下一 Epoch 继续高效训练 (Dataloader Reload)
```

### 2.1 候选池内存映射与生命周期 (`ALPoolManager`)
[data/al_pool_manager.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/al_pool_manager.py) 负责 1M 候选池的生命周期管理：
- 内存常驻仅加载参数矩阵 $P \in \mathbb{R}^{1000000 \times 141}$（约 564 MB），显存零压力。
- 支持 `init_kmeans` 冷启动：在训练前使用 `MiniBatchKMeans` 在 141 维参数空间中挑选出 $K=10,000$ 个最分散的参数点，作为第 0 轮初始训练集。

### 2.2 多进程 ODE 在线高速标注与三层防死锁机制
在细胞周期刚性 ODE 系统中，复杂突变参数可能导致 `scipy.integrate.solve_ivp(method='BDF')` 在 Fortran 底层陷入死循环。AL4PDE 部署了**三层防卡死保护架构**：
1. **内核级 SIGALRM 硬超时**：单样本积分超过 5.0 秒直接由操作系统内核发送信号中断并抛出 `TimeoutError`。
2. **线程级 Future 超时兜底**：`ProcessPoolExecutor` 收集结果时设置 `timeout=35s`，捕获 `FutureTimeoutError`。
3. **全局异常容错捕获**：对失败或超时的病态样本直接标记为 `None` 并剔除，绝不阻塞训练主进程。

### 2.3 训练节奏控制与增量 DataLoader 动态重构
- **指数倍增轮次 (Geometric Progression)**：
  - Round 0: 初始冷启动 $N_0 = 10,000$ 样本，训练至稳定；
  - Round 1: 智能采入 $10,000$ 样本 $\to$ 总训练量 $20,000$；
  - Round 2: 智能采入 $20,000$ 样本 $\to$ 总训练量 $40,000$。
- **DataLoader 动态重建**：触发主动学习后，新样本通过 `torch.cat` 拼接进 `train_dataset_lhs`，并在 Lightning 框架中触发 `trainer.reset_train_dataloader()`，实现下一 Epoch 无缝生效。

---

## 三、10 大生产级 AL4PDE 主动学习策略详解

以下 10 大策略在 [utils/al_callback.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/al_callback.py) 中通过 `--al_strategy <name>` 统一调用：

### 3.1 `al4pde_random` (无偏高维均匀基线)
- **原理**：从未标注的 1M 候选池中完全随机均匀抽取 $K$ 个样本。
- **用途**：作为无偏空间填充对照基线，用于验证各智能策略的信息增益。

### 3.2 `al4pde_uncertainty` (MC-Dropout 预测不确定度)
- **原理**：开启 Dropout 执行 $T=10$ 次模型前向传播，计算输出轨迹的方差：
  $$\mathcal{U}(x_i) = \frac{1}{V \cdot T_s} \sum_{v=1}^{39} \sum_{t=1}^{500} \text{Var}\left(\hat{y}_{i, v, t}^{(1)}, \dots, \hat{y}_{i, v, t}^{(T)}\right)$$
- **特点**：优先选取模型认知盲区与高方差振荡区域。

### 3.3 `al4pde_diversity` (332 维时序形状感知 + GPU 向量化 K-Center)
- **原理**：
  1. 通过模型前向预测抽取 **332 维时序动力学形状特征向量** $\mathbf{z}_i$（含 MASS 细胞生长包络、一阶时序导数、全变量极值与周期统计量）；
  2. 在特征空间对候选样本与当前训练集做 Z-score 标准化；
  3. 通过 GPU 向量化实现的贪心最大距离选择（MaxDist / K-Center），最大化候选样本到当前训练集的最小几何距离：
     $$\text{Acq}(x_i) = \min_{x_j \in \mathcal{D}_{\text{train}}} \|\mathbf{z}_i - \mathbf{z}_j\|_2$$
- **特点**：保证新样本在相空间动力学流形上具有最大几何覆盖度，杜绝数据冗余。

### 3.4 `al4pde_residual` (流速自适应相对生化物理残差)
- **原理**：
  利用模型预测轨迹 $\hat{u}_i(t)$，通过时间差分计算 $\frac{d\hat{u}_i}{dt}$，代入出芽酵母真实生化 ODE 右端方程 $f(\hat{u}_i, P_i)$ 计算生化流速残差：
  $$\mathcal{R}_i = \frac{\|\frac{d\hat{u}_i}{dt} - f(\hat{u}_i, P_i)\|_2}{\|\frac{d\hat{u}_i}{dt}\|_2 + \|f(\hat{u}_i, P_i)\|_2 + \epsilon}$$
  内嵌 **7 大极快变量剪枝机制** 与 **$L_{\text{flux}}$ 动态门控**（详见第四节）。
- **特点**：直接从第一性原理物理定律出发，挑选最违反生化反应动力学的样本。

### 3.5 `al4pde_res_div` (残差-多样性双目标帕累托前沿融合)
- **原理**：
  将物理残差得分 $\mathcal{R}_i$ 与相空间多样性得分 $\mathcal{D}_i$ 归一化至 $[0, 1]$，通过几何调和帕累托前沿进行融合：
  $$\text{Score}_i = \alpha \cdot \mathcal{R}_i + (1 - \alpha) \cdot \mathcal{D}_i$$
- **特点**：兼顾物理守恒违背度与动力学流形空间覆盖。

### 3.6 `al4pde_entropy` (双稳态相空间转移概率香农熵)
- **原理**：
  将 39 维轨迹在关键相平面（如 Clb2-Sic1 极限环平面）投影为离散状态转移概率分布 $p_k(x_i)$，计算动力学轨迹的香农熵：
  $$\mathcal{H}(x_i) = -\sum_{k} p_k(x_i) \log p_k(x_i)$$
- **特点**：优先采入相变复杂、混沌程度高与长尾过渡态样本。

### 3.7 `al4pde_bifurcation` (生物学分岔指标引导采样)
- **原理**：
  根据细胞周期核心生物学指标（Clb2 振幅突变、周期 length 突变、G1/M 期不可逆阻滞跳变）构建分岔临界打分函数 $\mathcal{B}(P_i)$：
  $$\mathcal{B}(P_i) = w_1 \Delta \text{Period} + w_2 \Delta \text{PeakAmplitude} + w_3 \mathbb{I}(\text{ArrestFlip})$$
- **特点**：极其高效地锁定野生型到突变体发生双稳态相变的分岔边界悬崖。

### 3.8 `al4pde_bifurcation_entropy` (分岔强度与动力学不确定度联合熵)
- **原理**：
  将分岔指标 $\mathcal{B}(P_i)$ 与相空间转移熵 $\mathcal{H}(x_i)$ 进行联合概率乘积建模：
  $$\text{Score}_i = \mathcal{B}(P_i) \times \mathcal{H}(x_i)$$
- **特点**：当前在 1M 纯随机池实验中展现出极强的 Badcase 消除能力。

### 3.9 `al4pde_bifurcation_diversity` (分岔边界引导 + 形状空间多样性)
- **原理**：
  先通过分岔指标 $\mathcal{B}(P_i)$ 筛选出 Top 20% 的高分岔候选子集，再在子集内部运行 GPU K-Center 多样性覆盖。

### 3.10 `al4pde_full_hybrid` (四维全能混合采样)
- **原理**：
  融合模型预测不确定度、物理残差、分岔指标与几何多样性四大信号：
  $$\text{Score}_i = \text{Rank}(\mathcal{U}_i) + \text{Rank}(\mathcal{R}_i) + \text{Rank}(\mathcal{B}_i) + \text{Rank}(\mathcal{D}_i)$$

---

## 四、生化物理残差计算与剪枝引擎 (`ode_residual_engine.py`)

生化反应方程包含极度刚性（Stiff）的瞬态快变量，直接计算全变量残差会导致数值爆炸。[utils/ode_residual_engine.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/ode_residual_engine.py) 实现了工业级残差流速引擎：

### 4.1 7 大极快变量剪枝机制
在计算 ODE 残差时，自动**舍弃 7 个微秒级准稳态快变量**：
- `C2P`, `C5P`, `F2P`, `F5P`（高周转率瞬态磷酸化复合物）
- `Vi20`, `lte1`, `PPX`（超快中间过渡态酶）
仅保留其余 32 个主控周期素与关键转录因子（Clb2, Clb5, Sic1, Cdh1, Cdc14, Swi5 等），使残差梯度具有高信噪比。

### 4.2 通量归一化门控因子 $L_{\text{flux}}$
残差计算采用流速自适应分母：
$$\text{Res}_{v} = \frac{|\dot{u}_v - f_v(u, p)|}{\max(|\dot{u}_v|, |f_v(u, p)|) + \epsilon_{\text{floor}}}$$
其中 $\epsilon_{\text{floor}} = 0.1$，彻底防止低浓度物种产生除以零的数值溢出。

---

## 五、实验配置与快速启动指南

在 [run_al_pure_random_1m.sh](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/run_al_pure_random_1m.sh) 中配置并启动多卡并行主动学习实验：

```bash
# 运行 1M 纯随机池 AL4PDE 流水线
bash run_al_pure_random_1m.sh
```

流水线会自动完成：
1. 校验/生成 100 万组准随机参数候选池 `pure_random_1m_pool.npz`；
2. 校验/生成 Iteration 0 初始训练集（10,000 样本）；
3. 绑定 GPU 独立启动选定的 AL4PDE 算法；
4. 训练完成后自动在 126 真实突变体上进行全量评测并输出多维分析报表。

---

## 六、1M 纯参数空间候选池 AL4PDE 八大策略最新实测基准与科学归因剖析

在统一的 `pure_random_1m_pool.npz` 候选池（100 万组纯参数 $P \in \mathbb{R}^{1000000 \times 141}$，免预仿真无标签）与基线初始集（10,000 样本）上，系统运行了 8 大生产级主动学习策略（每轮迭代扩充 2,000 样本，经过 2 轮迭代增至 14,000 样本），在 126 个真实生物突变体上的完整测试基准如下：

### 6.1 八大主动学习策略演进与泛化排行榜
| 主动学习策略 (`al_strategy`) | Round 0 Rel $L_2$ | Round 1 Rel $L_2$ | Round 2 Rel $L_2 \downarrow$ | Round 2 MAE $\downarrow$ | Round 2 Corr $\uparrow$ | 演进趋势与策略综合评价 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 🥇 **`al4pde_uncertainty`** | 0.1650 | 0.1492 | **0.1417** | **0.0763** | **0.9782** | **表现最佳**，MC-Dropout 稳定定位认识不确定性 |
| 🥈 **`al4pde_res_div`** | 0.1650 | 0.1540 | **0.1479** | 0.0801 | 0.9750 | 帕累托前沿平衡物理违背度与时序流形分布 |
| 🥉 **`al4pde_residual`** | 0.1650 | 0.1582 | **0.1512** | 0.0824 | 0.9731 | 纯生化方程物理残差挑选刚性违反样本 |
| 4. **`al4pde_random`** (基线) | 0.1650 | 0.1610 | **0.1568** | 0.0860 | 0.9715 | 无偏高维均匀被动扩展，收敛速度较慢 |
| 5. **`al4pde_diversity`** | 0.1650 | 0.1634 | **0.1590** | 0.0872 | 0.9702 | 纯 332-D 几何流形覆盖，忽略了生化临界区域 |
| 6. **`al4pde_bifurcation`** | 0.1650 | 0.1690 | **0.1745** | 0.0910 | 0.9650 | 集中采集分岔边界，引入了局部高频振荡噪声 |
| 7. **`al4pde_entropy`** | 0.1650 | 0.1752 | **0.1832** | 0.0954 | 0.9610 | 状态转移熵过拟合于相空间高混叠区域 |
| 8. **`al4pde_bifurcation_entropy`** | 0.1650 | 0.1820 | **0.2030** | 0.1089 | 0.9520 | **出现负向退化**，分岔与不确定度极值产生病态聚集 |

---

### 6.2 真实突变体三大表型模式（Pattern A / B / C）误差断层分析
将 126 个真实酵母突变体按动力学生物表型细分为三大模式，各策略呈现出高度一致的表型性能断层：

```
                      [真实突变体三大生物表型误差对比]
  Rel L2 误差
    0.35 ┤                                                 ┌─────────┐
    0.30 ┤                                                 │ Pattern C
    0.25 ┤                                                 │  0.3009 │
    0.20 ┤                             ┌─────────┐         │         │
    0.15 ┤         ┌─────────┐         │ Pattern B         │         │
    0.10 ┤         │ Pattern A         │  0.1694 │         │         │
    0.05 ┤         │  0.0884 │         │         │         │         │
    0.00 ┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────
                   单基因阻滞态        双稳态正常振荡       多重敲除/挽救突变
                   (Arrest Mutants)   (Viable / Limit Cyc) (Multi-Gene Knockout)
```

1. **Pattern A（单基因突变阻滞态，如单敲除 `cln2Δ`, `cdc20-ts`）**：
   - **平均 Rel $L_2 = 0.0884$**（高保真拟合）。系统趋于单一稳定不动点，动力学流场简单，神经算子能够精准锁定定态。
2. **Pattern B（正常振荡与极限环突变体，如 `clb5Δ`, `sic1Δ`）**：
   - **平均 Rel $L_2 = 0.1694$**（良好拟合）。包含非线性极限环振荡，存在微小的周期与振幅相移，但轨迹形态基本贴合。
3. **Pattern C（极端多重基因敲除/复合拯救突变体，如 `cln1Δ cln2Δ cln3Δ apc-ts`）**：
   - **平均 Rel $L_2 = 0.3009$**（**严重坍缩**），部分极端多突变体相对误差甚至高达 **$3.657$**！模型预测出现剧烈的双稳态误判或虚假高频自激振荡。

---

### 6.3 科学根因数学证明：连续参数候选池在离散敲除超平面上的“零测度定理”

为什么主动学习在 Pattern C 上无法通过多轮采样自适应修复？通过高维测度几何分析得出了严格的数学根因：

> [!CAUTION]
> **连续参数池在离散多敲除流形上的 Lebesgue 测度为零（Zero Measure on Discrete Knockout Hyperplanes）**：
> 1. 1M 纯随机候选池 `pool_1m_pure_random.npz` 是在 141 维连续对数空间 $\mathbb{R}^{141}$ 内部均匀生成的，任意坐标 $P_i$ 落在严格零点 $\{0\}$ 的概率为：
>    $$\mathbb{P}(P_i = 0) = 0$$
> 2. 对于真实多基因敲除突变体（例如三重敲除 `cln1Δ cln2Δ cln3Δ`），其动力学参数流形位于 141 维超立方体边界的极低维子超平面：
>    $$\mathcal{M}_{\text{KO}} = \{P \in \mathbb{R}^{141} \mid P_{k_1} = 0, P_{k_2} = 0, P_{k_3} = 0\}$$
> 3. 在 141 维 Lebesgue 测度下，$\mu_{141}(\mathcal{M}_{\text{KO}}) = 0$。因此，**无论 AL4PDE 算法前向打分如何优秀，连续候选池中永远不可能自然抽中严格为 0 的多敲除点！**
> 4. 模型只能在靠近边界的微小连续点（如 $P_i \approx 10^{-4}$）上采样，而在高度非线性的生化反馈网络中，微小的非零残留表达常数足以激活隐式正反馈开关，导致预测轨迹与真实阻滞表型截然相反。

### 6.4 下一代架构解决方案：混合候选池 (Hybrid Candidate Pool)
为了攻克 Pattern C 崩溃瓶颈，下一代主动学习候选池必须升级为**双轨制混合候选池**：
- **80% 连续探索池 (Continuous Exploration Pool)**：保留 80 万连续非线性点，负责探测相空间连续分岔与未知鞍点；
- **20% 离散基因型格点池 (Discrete Genotype Lattice Pool)**：通过离散组合生成包含 1~4 个关键基因严格置零 ($P_i = 0$) 或强启动子倍增 ($P_j = 20\times$) 的组合扰动样本，为多基因敲除外推提供坚实的支撑集支持。

