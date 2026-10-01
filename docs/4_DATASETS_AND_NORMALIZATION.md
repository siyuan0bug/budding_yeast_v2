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

### 1.1 数据集生成与采样引擎完整清单 (`data/`, 共 16 个核心工具)
代码库 `data/` 目录下配备了面向不同高维几何分布与生物学先验的生成器，全部支持 `--seed` 与 `--data_dir` 灵活调用：

| 引擎脚本 | 核心算法与采样机理 | 关键参数与功能定位 |
|:---|:---|:---|
| [generate_pure_random_dataset.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/generate_pure_random_dataset.py) | 141 维超立方体对数均匀连续采样 | 大规模基础纯随机基准集生成 |
| [generate_1m_pure_random_pool.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/generate_1m_pure_random_pool.py) | 100 万未仿真纯参数点 ($P, X_0$) 生成 | **AL4PDE 百万免预仿真候选池**（无 $Y$ 输出，防泄露） |
| [bams_dataset_generator.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/bams_dataset_generator.py) | 盲态流形相空间多样性 (BAMS) 采样 | 相空间测度平衡采样 |
| [gibbs_boltzmann_sampler.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/gibbs_boltzmann_sampler.py) | 基于能量梯度的 MCMC 吉布斯采样 | 鞍点与相变边缘参数富集 |
| [gibbs_sobol_sampler.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/gibbs_sobol_sampler.py) | 准蒙特卡洛 (QMC) Sobol 低差异序列扰动 | 高维空间均匀充盈采样 (Space-filling) |
| [generate_kdpp_gated_dataset.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/generate_kdpp_gated_dataset.py) | 核行列式点过程 (k-DPP) 排斥性采样 | 抑制样本冗余，最大化子空间体积 |
| [generate_kdpp_curv_gated_dataset.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/generate_kdpp_curv_gated_dataset.py) | 曲率感知 k-DPP 门控采样 | 在非线性流形高曲率区域自适应增密 |
| [generate_motif_topology_dataset.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/generate_motif_topology_dataset.py) | 生物因果模体（Motif）拓扑扰动生成 | 覆盖反馈回路、前馈回路等功能模体 |
| [reachability_check.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/reachability_check.py) | 零训练突变体参数可达性与流形流速审计 | 探测 126 突变体在各数据集中的几何距离 |
| [mass_dataset_generator.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/mass_dataset_generator.py) | 质量反馈驱动参数变异生成 (Mass V1/V2) | 支持表型双标签生成的经典基准集 |
| [mass_wt_blind_sampler.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/mass_wt_blind_sampler.py) | 野生型固定点不动点代数反演采样 | 100% 盲态（无任何突变体先验）空间采样 |
| [mirror_pure_random_ko_interval.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/mirror_pure_random_ko_interval.py) | 敲除微小连续区间 $[0, 10^{-4}]$ 采样器 | 探究平滑敲除在神经算子中的连续性 |
| [al_pool_manager.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/al_pool_manager.py) | 内存映射 (Memory-mapped) 动态候选池管理器 | 支持千万级参数池低显存高吞吐流式访问 |
| [build_bio_multirelational_graphs.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/build_bio_multirelational_graphs.py) | 39 节点生化因果图与拓扑张量构建 | 输出激活、抑制、络合等多关系矩阵 |
| [yeast_datamodule.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/yeast_datamodule.py) | PyTorch Lightning 统一高效数据管道 | 负责多进程加载、在线切片与归一化 |
| [yeast_multirelational_graphs.npz](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/yeast_multirelational_graphs.npz) | 预编译生物先验异构图结构张量 | 提供给 `BioRefinerPlus` 等图算子使用 |

---

## 二、四大参数采样空间系统与真实等位基因扰动数学建模

在出芽酵母动力学仿真中，参数分布设计直接决定了代理模型学习到的流形结构。系统总结并沉淀了四大采样体系：

### 2.1 四大参数采样空间体系 (Systems A ~ D)
1. **System A（连续对数均匀 + 离散突变等位基因，如 `mass_dataset_generator.py`）**：
   在野生型 $P_{\text{wt}}$ 基础上施加 $\log_{10}$-正态或均匀连续扰动，并注入预设的生物突变等位基因。早期版本存在突变体特征泄露，现已全面解耦。
2. **System B（离散对数拷贝数 + 物理锚点包络 + 边界缓冲，如 `mutant_aware.py`）**：
   引入物理边界保护（如针对 `ts` 温度敏感突变体设置 $\le 1\%$ 的边界微扰缓冲带），避免去重合并时发生奇异坍缩（如彻底解决历史出现的 #73 `cdc20-ts` 特征重叠）。
3. **System C（连续 $[0, 10^{-4}]$ 区间平滑敲除，如 `mirror_pure_random_ko_interval.py`）**：
   针对神经算子在严格零点 $P=0$ 处可能产生的激活偏置，将敲除建模为连续近零区间，增强导数平滑性。
4. **System D（野生型不动点代数反演盲态采样，如 `mass_wt_blind_sampler.py`）**：
   严格零突变体先验，通过稳定稳态极限环代数方程反演可行参数流形，实现严苛的 Sim-to-Real 盲态评测。

### 2.2 常见等位基因变异的精确代数映射方程
| 等位基因表型 | 生物学含义 | 精确参数代数变换方程 | 典型影响关键基因 |
|:---|:---|:---|:---|
| **KO (Null/$\Delta$)** | 完全基因敲除 (Gene Knockout) | $P_{\text{synth}} = 0.0, \quad P_{\text{deg}} \in [P_{\text{wt}}, 2P_{\text{wt}}]$ | `cln2`, `clb2`, `sic1`, `cdh1` |
| **MC (Multiple Copies)** | 稳定低拷贝过表达 (2~4 copies) | $P_i = k \cdot P_{\text{wt}}, \quad k \in [2.0, 4.0]$ | `CLN2`, `CLB5` |
| **ts (Temp-sensitive)** | 温度敏感突变 (功能衰减 90%~99%) | $P_i = \alpha \cdot P_{\text{wt}}, \quad \alpha \in [0.01, 0.10]$ | `cdc20-ts`, `cdc15-ts`, `apc-ts` |
| **GAL-OE (Galactose)** | 强启动子超强过表达 (5~20 倍) | $P_i = \beta \cdot P_{\text{wt}}, \quad \beta \in [5.0, 20.0]$ | `GAL-CLB2`, `GAL-CDC14`, `GAL-SIC1` |
| **db (Destruction Box)** | 泛素化降解框突变 (不可降解) | $P_{\text{deg}} = \epsilon \cdot P_{\text{wt}}, \quad \epsilon \le 0.05$ | `clb2-db`, `pds1-db` |
| **APC-A** | APC 催化亚基磷酸化突变 | 激活速率降解至 $0.15\times$，解聚常数 $3\times$ | `apc-a` |
| **TAB6-1** | 磷酸酶亲和力变异 | 亲和常数 $K_m$ 偏移 $0.2 \sim 0.5\times$ | `tab6-1` |
| **DAmP** | mRNA 衰变加速 (表达量下调 70%~80%) | $P_i = \gamma \cdot P_{\text{wt}}, \quad \gamma \in [0.20, 0.30]$ | 必需基因亚活性研究 |

> [!NOTE]
> **动力学高敏感基因集合**：
> - 周期启动引擎：`CLN2`（对初速度与 Start 检查点敏感）；
> - 有丝分裂退出与纺锤体组装：`TEM1`, `ESP1`, `CDC20`, `CDC15`, `APC`；
> - 主振荡抑制核心：`CLB2`（对降解速率常数与 Sic1 络合亲和力极度敏感）。

---

## 三、19 大合成数据集泛化至 126 真实突变体终极排行榜

通过系统性评测（由 `scripts/analysis/dataset_diagnosis/deep_dataset_diagnosis.py` 驱动），19 大合成数据集训练的模型在 126 个真实酵母突变体上的泛化排行榜如下：

| 排名 | 训练数据集名称 | 样本规模 | Real Rel $L_2 \downarrow$ | Real MAE $\downarrow$ | Pearson Corr $\uparrow$ | 核心优势 / 泛化机理 |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| 🥇 1 | **`mutant_aware_100000`** | 100k | **0.0980** | **0.0463** | **0.9852** | **全场唯一破 0.10**，物理包络覆盖突变边界 |
| 🥈 2 | **`kdpp_curv_gated_100000`** | 100k | 0.1235 | 0.0592 | 0.9780 | 曲率引导增密，精确刻画相变极限环突跃 |
| 🥉 3 | **`motif_topology_100000`** | 100k | 0.1342 | 0.0610 | 0.9745 | 因果模体拓扑变异打破单调拟合 |
| 4 | `gibbs_boltzmann_100000` | 100k | 0.1420 | 0.0658 | 0.9712 | 鞍点与相空间过渡态富集 |
| 5 | `kdpp_gated_100000` | 100k | 0.1488 | 0.0712 | 0.9690 | 几何正交排斥，降低冗余度 |
| 6 | `bams_100000` | 100k | 0.1550 | 0.0740 | 0.9664 | 盲态流形相空间多样性覆盖 |
| 7 | `gibbs_sobol_100000` | 100k | 0.1586 | 0.0781 | 0.9631 | QMC 空间充盈性佳，但在突变极值区稀疏 |
| 8 | `mass_100000_dual_labels` | 100k | 0.1652 | 0.0815 | 0.9602 | 质量反馈提供动态生物约束 |
| 9 | `mass_v2_100000` | 100k | 0.1770 | 0.0863 | 0.9580 | 样本规模大但高流速区存在梯度欠拟合 |
| 10 | `mirror_pure_random_ko` | 100k | 0.1985 | 0.0945 | 0.9510 | 连续敲除区间改善了边缘不连续性 |
| ... | ... | ... | ... | ... | ... | ... |
| 19 | **`pure_random_100000`** | 100k | **0.2431** | **0.1210** | **0.9320** | **基线最弱**，高维均匀分布在流形边缘极度稀疏 |

---

## 四、生物因果模体（Motif 100K）vs 纯随机（Random 100K）消融实证

对比 `motif_topology_100000` 与 `pure_random_100000` 的严格消融实验揭示了生物先验的决定性价值：

```
[评价维度]                   [纯随机 Random 100K]       [模体拓扑 Motif 100K]       [提升幅度]
真实突变体平均 MAE:               0.0920                   0.0508            ==> 误差降低 44.8%
致命坏例数 (Fatal Badcases):       12 个                     3 个             ==> 坏例锐减 75%
双稳态阻滞态准确率:                81.3%                    96.8%             ==> 阻滞判决提升 15.5%
```

**机理深度解释**：
纯随机采样生成的参数往往破坏了酵母细胞周期关键的**拯救通路（Rescue Motifs）**。例如在双敲除突变体 `sic1Δ clb2Δ` 中，细胞能够恢复存活正是依赖于替代激酶通路的补偿；纯随机样本由于未施加网络因果关联，使得模型在遇到此类突变时将其误判为不可逆有丝分裂阻滞。模体拓扑采样通过联合扰动互补功能子网络，成功教会了算子网络识别细胞的动态补偿能力。

---

## 五、合成数据集中的五大泛化瓶颈（P0 ~ P4 根因诊断）

通过对 MassV2 与纯随机数据集进行高维几何流形与时序导数归因分析（`analyze_massv2_badcase_attribution.py`），发现了 Sim-to-Real 的五大本质瓶颈：

1. **P0 瓶颈：高维特征空间 $k$-NN 几何鸿沟（占致命 Badcase 的 42%）**
   - 真实突变体与合成训练样本的 141 维欧氏距离普遍在 $12.0 \sim 16.5$ 之间，远大于训练集内部的平均距离（$3.2$）。
   - **皮尔逊相关系数强负相关**：$k$-NN 几何距离与预测轨迹的相关度 $r$ 呈现强负相关（**$r = -0.78$**），证明当前绝大部分高误差来自于外推（Out-of-Distribution）。
2. **P1 瓶颈：局部瞬态极大值归一化畸变（占高 MAE Badcase 的 25%）**
   - 在强过表达突变体（如 `GAL-CLB2`）中，特定蛋白质浓度可冲至正常峰值的 15~20 倍（达到 $15\sigma$ 异常值），造成归一化挤压，使模型丢失基线细节。
3. **P2 瓶颈：陡峭相变悬崖与高流速时变区（占低 Corr Badcase 的 18%）**
   - 有丝分裂退出期（M/G1 transition）内，Cdh1 激活与 Clb2 降解的导数绝对值达到 $15 \sim 30 \text{ min}^{-1}$。纯随机数据在此区域采样密度低，导致时间导数预测出现相移延迟。
4. **P3 瓶颈：倍增时间与营养源碳源解耦（$Mdt = 150\text{ min}$ 耦合性）**
   - 真实突变体实验中，部分慢生菌株倍增时间拉长至 150 分钟以上，合成数据若固定生长速率常数，将导致细胞生长与分裂周期相位失配。
5. **P4 瓶颈：高维超球体表面的超稀疏邻域（Curse of Dimensionality）**
   - 141 维参数空间中，即便采样 100 万点，其局部超球体填充率依然趋近于 0，必须依靠降维因果投影与流形约束。

---

## 六、51.6% 真实突变体零样本（Zero-Shot）基因型错配困境

对 126 真实酵母突变体的基因型与 10 万合成训练样本进行精准模式匹配统计发现：
- **51.6%（65/126）的真实实验突变体，在 100,000 规模的合成数据集中完全没有任何精确匹配的基因型！**
- 尤其对于多重基因突变体（如三敲除 `cln1Δ cln2Δ cln3Δ`、敲除伴随温度敏感突变 `cln1Δ cln2Δ apc-ts` 等），合成数据集随机命中概率在数学测度上为零。
- 这决定了**本项目的本质不是简单内插（Interpolation），而是极端跨模态多基因突变外推（Multi-Gene Zero-Shot Extrapolation）**！

---

## 七、严格盲态协议 (Strict Blind Protocol) 与因果拓扑消融证明

为了确立科研成果的严密性与无可争议性，项目实施了三层防泄露与因果消融验证：
1. **严格盲态验证 (Strict Blind Protocol)**：
   - 候选池构建代码 [generate_1m_pure_random_pool.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/generate_1m_pure_random_pool.py) 与训练流程全面禁止导入 126 突变体参数字典；
   - 评测脚本 [eval.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/eval.py) 仅在最终 Checkpoint 推断阶段一次性载入真实生物学金标准。
2. **因果拓扑消融证明 (Causal Ablation Proof)**：
   - 当在 `BioRefinerPlus` 中完全移除 [yeast_multirelational_graphs.npz](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/yeast_multirelational_graphs.npz) 的生物因果先验图时，模型在 65 个零样本突变体上的 Rel $L_2$ 误差剧烈上升 **$38.4\%$**（由 0.0741 恶化至 0.1025）；
   - 确凿证明了神经算子必须依赖生物因果拓扑的结构性约束，才能在未知基因型空间实现高保真物理泛化。

---

## 八、酵母生化多关系先验网络 (Biological Multirelational Graphs)

在细胞周期动力学中，39 个蛋白质与生化复合物之间存在已知的严谨生化反应机理。框架在 [data/build_bio_multirelational_graphs.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/build_bio_multirelational_graphs.py) 中构建了生物学先验图结构：

```
39 Nodes (Yeast Cyclins, Kinases, Complexes)
├── yeast_boolean_adjacency_matrix.npy   (39x39 无向连通拓扑)
├── yeast_signed_adjacency_matrix.npy    (39x39 带符号激活/抑制矩阵: +1 激活, -1 抑制)
└── yeast_multirelational_graphs.npz     (多关系异构张量: 激活, 抑制, 络合, 磷酸化, 泛素化降解)
```

图神经网络与混合模型（如 `BioMRefinerEventHybrid`, `HGNN`, `GNN-FNO`）通过 `data/yeast_datamodule.py` 自动加载图张量，用于空间拓扑消息传递与生物图注意力权重约束。

---

## 九、归一化算法综合审计与最优架构方案

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

## 十、真实突变体上的归一化方案消融基准 (`BioRefinerPlus`)

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

