# 数据集全景、生物图先验与归一化规范 (Datasets, Biological Graphs & Normalization)

本文档系统介绍 `budding_yeast_v2` 框架中使用的数据集体系、生化多关系先验网络、全量数据归一化方案及其在 126 个真实生物突变体上的消融审计结论。

---

## 一、数据集体系概览 (Dataset Overview)

系统涵盖由动力学 ODE 求解器生成的 **10 万量级预训练/基准数据集**、**100 万级主动学习未标注候选池**，以及 **126 个真实酵母突变体金标准测试集**：

| 数据集名称 | 样本规模 | 时长与步数 | 动力学参数生成机理 | 核心用途与评估维度 |
| :--- | :--- | :--- | :--- | :--- |
| **`REAL_MUTANTS`** | **126** | $210\text{ min}$, $500\text{ steps}$ | 严格对应 Chen et al. (2004) 真实酵母表型实验 | **金标准终极评估基准**（不可参与训练） |
| **`mutant_aware_pure_random_100000_210m_500s.npz`** | $100,126$ | $210\text{ min}$, $500\text{ steps}$ | 纯随机均匀参数空间采样 + 126 突变体镜像扰动 | 当前全模型基准评测统一基线数据集 |
| **`mass_100000_dataset_210min_500steps_dual_labels.npz`** | $100,126$ | $210\text{ min}$, $500\text{ steps}$ | 质量反馈自适应参数生成 + 表型双标签 | 经典双标签对比基线数据集 |
| **`pool_1m_pure_random.npz`** | **1,000,000** | 仅参数 ($141$ 维) | 宽域非线性参数均匀随机采样（无时序） | **AL4PDE 主动学习百万候选池** |
| **`bams_100000_210m_500s.npz`** | $100,000$ | $210\text{ min}$, $500\text{ steps}$ | 盲态流形相空间多样性采样 (BAMS) | 流形相空间表征消融 |
| **`gibbs_boltzmann_100000_210m_500s.npz`** | $100,000$ | $210\text{ min}$, $500\text{ steps}$ | 基于能量梯度的 MCMC 吉布斯采样 | 鞍点与相变边缘参数富集 |
| **`kdpp_gated_100000_210m_500s.npz`** | $100,000$ | $210\text{ min}$, $500\text{ steps}$ | 核行列式点过程 (k-DPP) 门控采样 | 排斥性几何正交采样 |
| **`motif_topology_100000_210m_500s.npz`** | $100,000$ | $210\text{ min}$, $500\text{ steps}$ | 生物网络模体结构感知参数变异 | 网络拓扑相关变异分布 |

### 数据集生成器脚本清单 (`data/`)
- [data/generate_pure_random_dataset.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/generate_pure_random_dataset.py)：生成均匀随机大基准集。
- [data/generate_1m_pure_random_pool.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/generate_1m_pure_random_pool.py)：生成 100 万未仿真主动学习参数候选池。
- [data/bams_dataset_generator.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/bams_dataset_generator.py)：生成盲态流形相空间多样性采样集。
- [data/gibbs_boltzmann_sampler.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/gibbs_boltzmann_sampler.py)：吉布斯-玻尔兹曼能量自适应采样器。
- [data/gibbs_sobol_sampler.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/gibbs_sobol_sampler.py)：吉布斯准蒙特卡洛低差异序列采样器。
- [data/generate_kdpp_gated_dataset.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/generate_kdpp_gated_dataset.py)：k-DPP 门控排斥性采样。
- [data/generate_motif_topology_dataset.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/generate_motif_topology_dataset.py)：模体拓扑扰动数据生成器。
- [data/reachability_check.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/reachability_check.py)：参数与轨迹空间可达性（Reachability）审计引擎。

---

## 二、酵母生化多关系先验网络 (Biological Multirelational Graphs)

在细胞周期动力学中，39 个蛋白质与生化复合物之间存在已知的严谨生化反应机理。框架在 [data/build_bio_multirelational_graphs.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/build_bio_multirelational_graphs.py) 中构建了生物学先验图结构：

```
39 Nodes (Yeast Cyclins, Kinases, Complexes)
├── yeast_boolean_adjacency_matrix.npy   (39x39 无向连通拓扑)
├── yeast_signed_adjacency_matrix.npy    (39x39 带符号激活/抑制矩阵: +1 激活, -1 抑制)
└── yeast_multirelational_graphs.npz     (多关系异构张量: 激活, 抑制, 络合, 磷酸化, 泛素化降解)
```

图神经网络与混合模型（如 `BioMRefinerEventHybrid`, `HGNN`, `GNN-FNO`）通过 `data/yeast_datamodule.py` 自动加载图张量，用于空间拓扑消息传递与生物图注意力权重约束。

---

## 三、归一化算法综合审计与最优架构方案

在深度算子网络中，归一化不仅是数值缩放，它直接决定了 **FiLM 条件层调制增益、Hessian 矩阵条件数与反向传播梯度的稳定性**。

### 1. 为什么历史方案会在突变体上崩溃？
```
                 [参数输入空间]                           [模型内部表现]                     [突变体评测结果]
旧版线性:  120x 过表达 -> Z = +65 ~ +600   ====>  FiLM 激活剧烈饱和，梯度爆炸   ====>  15号、44号等过表达突变体崩溃
新版Log10: 完全敲除 0.0 -> Z = -7.0 ~ -23.8 ====>  三敲除模长暴涨至 15 ~ 76     ====>  19号、76号等多重敲除突变体崩溃
统一asinh: 敲除 ~ -1.8, 120x ~ +3.9         ====>  全量模长稳固在 5 ~ 12         ====>  两类 Badcase 同时彻底修复
```

- **旧版线性 Z 分数**：强启动子过表达（120~1500 倍）导致 $Z = +65 \sim +600$，单样本梯度是正常样本的数千倍，冲毁优化器动量；
- **Log10 截断方案**：完全敲除 $P=0$ 引入 $\log_{10}(10^{-8})$，导致 $Z = -7.0 \sim -23.8$。三重敲除突变体参数模长冲至 76.11，强行使网络 FiLM 通道静默。

### 2. 统一双曲反正弦标准化（Unified `asinh(Z)`）
**数学公式：**
$$Z_p = \frac{P - \mu_p}{\sigma_p + \epsilon}, \quad P_{\text{norm}} = \text{asinh}(Z_p) = \ln\left(Z_p + \sqrt{Z_p^2 + 1}\right)$$
- **全量 100,126 样本审计**：所有 1410 万个参数坐标严格落在 **$[-3.10, +4.00]$**，离群点比例为 **$0.000\%$**，最大向量模长由 Log10 的 76.11 降至 **12.99**；
- **严格单调可微**：完全敲除 $P=0$ 平滑映射到 $-1.8$，微量弱效突变（$0.0005$）保留清晰可辨的梯度台阶。

---

## 四、真实突变体上的归一化方案消融基准 (`BioRefinerPlus`)

我们在统一数据集 `mutant_aware_pure_random_100000_210m_500s` 上，对 `BioRefinerPlus` 模型进行了 3 种状态变量 $Y$ 归一化方案的严格实测评测（评估 126 个真实生物突变体）：

| 归一化方案 (Normalization Scheme) | Real Mutant Rel $L_2$ $\downarrow$ | Pearson Corr $\uparrow$ | Real Mutant MAE $\downarrow$ | Real Mutant MSE $\downarrow$ | 综合评价 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **标准 Z-Score (`standard_zscore`)** | **0.1518** | **0.9723** | **0.0869** | **0.0631** | **综合指标最优，最稳健 baseline** |
| **Asinh Z-Score (`asinh_zscore`)** | 0.1713 | 0.9655 | 0.1018 | 0.0842 | 表现稳健，强非线性尺度压缩 |
| **Asinh Scale (`asinh_scale`)** | 0.2054 | 0.9576 | 0.0922 | 0.0886 | $\sinh$ 双向指数反变换使微小扰动放大 |

> [!TIP]
> **最佳实践建议**：
> - **参数端 ($P$)**：全局推荐且默认采用 `asinh(Z)` 变换。
> - **状态变量端 ($Y$)**：默认推荐 `standard_zscore`，其反归一化物理保真度最高。
