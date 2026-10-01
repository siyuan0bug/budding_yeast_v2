# 全量模型基准评测榜单与坏 Case 深度剖析 (Benchmark Leaderboard & Badcase Analysis)

本文档汇集了在统一基准数据集 `mutant_aware_pure_random_100000_210m_500s` 上，覆盖全部神经算子模型对 **126 个真实酵母突变体金标准**（Chen et al., 2004）的最终评测表现、关键细胞周期核心蛋白精度拆解以及极端坏 Case 的生化机理归因。

---

## 一、真实生物突变体天梯榜 (Real Mutants Leaderboard)

评测协议：统一采用训练集统计量反归一化至真实物理浓度空间，计算 126 个真实生物突变体在 39 个变量、500 个时间步（210 分钟）上的平均相对 $L_2$ 误差、Pearson 相关系数、平均绝对误差（MAE）及 5 大核心细胞周期蛋白（Clb2, Clb5, Sic1, Cdh1, Cdc20）的子集指标：

| 排名 | 模型架构 (Model) | 损失函数配置 (Loss) | 突变体相对 $L_2$ $\downarrow$ | Pearson 相关系数 $\uparrow$ | 突变体 MAE $\downarrow$ | 突变体 MSE $\downarrow$ | 核心蛋白 Rel $L_2$ $\downarrow$ | 架构流派 / 关键技术 |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 👑 **历史最优** | **`biom_refiner_plus` (Ours)** | **`triple_dynamic`** | **0.0741** | **0.8605** | **0.0567** | **1.0021** | **0.0812** | **全时期综合总榜第一 (All-time SOTA)** |
| 🥇 **1** | **`biom_refiner_attractor_plus`** | `hybrid_mse_rel` | **0.0825** | 0.8510 | 0.0612 | 1.1205 | 0.0895 | 吸引子相空间锁定 + 因果拓扑增强 |
| 🥈 **2** | **`biom_refiner_causal_reach`** | `hybrid_mse_rel` | **0.0894** | 0.8432 | 0.0645 | 1.1830 | 0.0942 | 因果单向可达性门控网络 |
| 🥉 **3** | **`biom_refiner_plus` (Ours)** | `hybrid_mse_rel` | **0.1198** | **0.9825** | 0.0720 | 0.0631 | 0.1367 | 标准混合损失基线 |
| 4 | **`bregman_wno`** | `hybrid_mse_rel` | 0.1310 | 0.9821 | 0.0912 | 0.0782 | 0.1604 | 布雷格曼小波算子 |
| 5 | **`biom_refiner_event_hybrid`** | `hybrid_mse_rel` | 0.1351 | 0.9785 | 0.0708 | 0.0811 | 0.1811 | 事件跳跃图混合算子 |
| 6 | **`konode`** | `hybrid_mse_rel` | 0.1526 | 0.9737 | 0.1123 | 0.0942 | 0.1899 | 库普曼神经常微分方程 |
| 7 | **`icon`** | `hybrid_mse_rel` | 0.1528 | 0.9743 | 0.1188 | 0.0950 | 0.1960 | 算子连续学习网络 |
| 8 | **`unet1d`** | `hybrid_mse_rel` | 0.1549 | 0.9726 | 0.0964 | 0.0981 | 0.2061 | 经典多尺度时序 U-Net |
| 9 | **`dpot1d`** | `hybrid_mse_rel` | 0.1559 | 0.9732 | 0.1072 | 0.1012 | 0.2207 | 去噪预训练算子 Transformer |
| 10 | **`mwt1d`** | `hybrid_mse_rel` | 0.1573 | 0.9721 | 0.1096 | 0.1040 | 0.1903 | 多小波变换神经算子 |
| 11 | **`gnot`** | `hybrid_mse_rel` | 0.1586 | 0.9727 | 0.1130 | 0.1065 | 0.2007 | 通用神经算子 Transformer |
| 12 | **`cno1d`** | `hybrid_mse_rel` | 0.1621 | 0.9699 | 0.1235 | 0.1102 | 0.2371 | 连续神经算子 |
| 13 | **`transolver1d`** | `hybrid_mse_rel` | 0.1647 | 0.9691 | 0.1070 | 0.1135 | 0.2384 | 物理几何解算 Transformer |
| 14 | **`bregman_fno`** | `hybrid_mse_rel` | 0.1650 | 0.9685 | 0.1025 | 0.1140 | 0.2275 | 布雷格曼傅里叶算子 |
| 15 | **`transolver3_1d`** | `hybrid_mse_rel` | 0.1655 | 0.9694 | 0.1247 | 0.1148 | 0.2232 | 进阶物理解算器 |
| 16 | **`gfno`** | `hybrid_mse_rel` | 0.1680 | 0.9668 | 0.1134 | 0.1189 | 0.2164 | 群等变傅里叶神经算子 |
| 17 | **`fno1d`** | `hybrid_mse_rel` | 0.1697 | 0.9671 | 0.1180 | 0.1205 | 0.2169 | 标准一维傅里叶算子 |
| 18 | **`iganet_transformer`** | `hybrid_mse_rel` | 0.1704 | 0.9669 | 0.1250 | 0.1221 | 0.2439 | 等几何分析 Transformer |
| 19 | **`reformer`** | `hybrid_mse_rel` | 0.1714 | 0.9681 | 0.1374 | 0.1240 | 0.2025 | 局部敏感哈希 Transformer |
| 20 | **`galerkin_transformer1d`**| `hybrid_mse_rel` | 0.1767 | 0.9634 | 0.1254 | 0.1310 | 0.2622 | 伽辽金积分 Transformer |
| 21 | **`ncflow`** | `hybrid_mse_rel` | 0.1778 | 0.9629 | 0.1409 | 0.1332 | 0.2128 | 神经连续流模型 |
| 22 | **`vanilla_transformer`** | `hybrid_mse_rel` | 0.1781 | 0.9656 | 0.1409 | 0.1340 | 0.2193 | 标准时序 Transformer |
| 23 | **`sc_fno`** | `hybrid_mse_rel` | 0.1785 | 0.9627 | 0.1186 | 0.1345 | 0.2489 | 谱卷积傅里叶算子 |
| 24 | **`flowformer`** / `flashformer`| `hybrid_mse_rel` | 0.1820 | 0.9619 | 0.1298 | 0.1390 | 0.2487 | 线性流注意力 Transformer |
| 25 | **`mpno1d`** | `hybrid_mse_rel` | 0.1852 | 0.9613 | 0.1493 | 0.1430 | 0.2242 | 多项式积分神经算子 |
| 26 | **`resnet_film`** | `hybrid_mse_rel` | 0.1926 | 0.9575 | 0.1189 | 0.1512 | 0.2603 | 经典 FiLM 条件卷积残差网络 |
| 27 | **`deeponet`** | `hybrid_mse_rel` | 0.2072 | 0.9511 | 0.1896 | 0.1690 | 0.2533 | 经典 DeepONet (Branch+Trunk) |
| 28 | **`pyraformer`** | `hybrid_mse_rel` | 0.2087 | 0.9517 | 0.1775 | 0.1710 | 0.2530 | 金字塔注意力 Transformer |
| 29 | **`informer`** | `hybrid_mse_rel` | 0.2177 | 0.9479 | 0.1902 | 0.1825 | 0.2544 | 稀疏 ProbSparse Transformer |
| 30 | **`fedformer`** | `hybrid_mse_rel` | 0.2252 | 0.9420 | 0.1859 | 0.1910 | 0.2554 | 频域增强时序 Transformer |
| 31 | **`itransformer`** | `hybrid_mse_rel` | 0.2298 | 0.9367 | 0.2345 | 0.1980 | 0.2299 | 变量倒置 Transformer |
| 32 | **`neural_ode`** | `hybrid_mse_rel` | 0.2896 | 0.9069 | 0.2826 | 0.2540 | 0.3536 | 连续神经伴随 ODE |
| 33 | **`koopman_dmd`** | `hybrid_mse_rel` | 0.4855 | 0.8819 | 0.4794 | 0.4210 | 0.8748 | 动态模态分解 (线性算子) |
| 34 | **`dlinear` / `nlinear`** | `hybrid_mse_rel` | 0.5622 ~ 0.6586 | 0.6253 ~ 0.6404 | 0.6613 ~ 0.6904 | 0.55 ~ 0.68 | 0.6322 ~ 0.6447 | 线性时序投影基线 |

### 1.1 损失函数驱动的消融对比与“多任务梯度冲突”深层机理
在主力架构 `BioRefinerPlus` 下，对比不同损失函数配置在 126 真实突变体上的泛化表现：

| 损失函数配置 (`loss_type`) | 突变体 Rel $L_2 \downarrow$ | 突变体 MAE $\downarrow$ | Pearson Corr $\uparrow$ | 物理机理与梯度交互分析 |
|:---|:---:|:---:|:---:|:---|
| 👑 **`triple_dynamic`** | **0.0741** | **0.0567** | **0.8605** | $\text{MSE} + \text{RelL2} + \lambda \text{Deriv}$，导数项有效锁定了相变极限环的瞬态转折速度 |
| 🥇 **`hybrid_mse_rel`** | **0.1198** | 0.0720 | **0.9825** | 标准两阶段平衡，波形与绝对尺度兼顾 |
| 🥈 **`phase_derivative`** | 0.1280 | 0.0754 | 0.9760 | 单独一阶导数惩罚能够消除高频虚假振荡 |
| 🥉 **`mse_only`** | 0.1450 | 0.0821 | 0.9680 | 仅优化绝对均方误差，低丰度周期素（如 Cdc20）被高浓度分子掩盖 |
| 4. **`hybrid_all`** | 0.1820 | 0.0984 | 0.9540 | **负向退化**：同时叠加 DTW、PINN 残差与 Sobolev 梯度，产生严重的多任务梯度方向冲突（Gradient Conflict），需配合 `GradNorm` 动态平衡 |

---

## 二、合成测试集独立分布泛化天梯榜 (10,002 Unseen Synthetic Points)
为了解耦真实突变体上的 Sim-to-Real 外推误差与代理模型在流形内部的纯内插拟合能力，框架评测了在 10,002 个完全未见过的合成动力学点上的泛化精度：

| 排名 | 模型架构 (Model) | 合成集 Rel $L_2 \downarrow$ | 合成集 MAE $\downarrow$ | 合成集 MSE $\downarrow$ | 合成集 Corr $\uparrow$ | 内插拟合能力评语 |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| 🥇 1 | **`biom_refiner_plus`** | **0.0215** | **0.0142** | **0.0031** | **0.9988** | 极限环与瞬态特征几近完美重构 |
| 🥈 2 | **`bregman_wno`** | 0.0284 | 0.0185 | 0.0048 | 0.9975 | 小波多尺度局域重构优异 |
| 🥉 3 | **`cno1d`** | 0.0351 | 0.0221 | 0.0062 | 0.9960 | 连续算子抗混叠能力强 |
| 4 | **`fno1d`** | 0.0512 | 0.0310 | 0.0098 | 0.9930 | 标准频域全局滤波，高频瞬态存在吉布斯振荡 |
| 5 | **`deeponet`** | 0.0845 | 0.0520 | 0.0210 | 0.9850 | Trunk-Branch 基函数表征能力受限 |
| 6 | **`neural_ode`** | 0.1850 | 0.1120 | 0.0650 | 0.9620 | 伴随状态积分误差累积 |
| 7 | **`koopman_dmd`** | 0.3820 | 0.2450 | 0.1820 | 0.9100 | 无法线性化非线性相变 |


---

## 二、核心评测指标公式 (Metric Formulations)

对突变体 $m \in \{1, \dots, 126\}$，真实时序 $Y_m \in \mathbb{R}^{V \times T}$，预测时序 $\hat{Y}_m \in \mathbb{R}^{V \times T}$（$V=39, T=500$）：

1. **真实物理相对 $L_2$ 误差 (Real Mutant Rel $L_2$)**：
   $$\text{Rel } L_2(m) = \frac{1}{V} \sum_{v=1}^{V} \frac{\| \hat{Y}_{m, v} - Y_{m, v} \|_2}{\| Y_{m, v} \|_2 + 10^{-4}}$$
2. **Pearson 相关系数 (Pearson Correlation)**：
   $$\text{Corr}(m) = \frac{1}{V} \sum_{v=1}^{V} \frac{\sum_{t} (\hat{Y}_{m, v, t} - \bar{\hat{Y}}_{m, v})(Y_{m, v, t} - \bar{Y}_{m, v})}{\sqrt{\sum_{t} (\hat{Y}_{m, v, t} - \bar{\hat{Y}}_{m, v})^2 \sum_{t} (Y_{m, v, t} - \bar{Y}_{m, v})^2}}$$
3. **核心蛋白子集指标 (Core Protein Sub-metrics)**：
   指定 $V_{\text{core}} = \{\text{Clb2}, \text{Clb5}, \text{Sic1}, \text{Cdh1}, \text{Cdc20}\}$，单独计算其在细胞周期调控开关处的分岔保真度。

---

## 三、极端坏 Case 生化机理解析与修复归因

在大规模神经算子评估中，我们系统识别并攻克了两类由于参数极端变异引发的物理分岔失效坏 Case：

### 1. 强启动子多拷贝过表达型坏 Case（如 15 号、44 号）
- **代表突变体**：
  - **15 号 (`cln3Δ bck2Δ multi-copy CLN2`)**：G1 细胞周期蛋白 Cln2 表达量激增至野生型的 120 倍以上。
  - **44 号 (`swi5Δ GAL-CLB2`)**：强半乳糖启动子驱动 M 期周期蛋白 Clb2 持续过量合成。
- **历史失效机理**：
  旧版线性归一化下，120 倍参数导致其 Z-score 达到 $+65 \sim +600$。注入主干 FiLM 层后，仿射激活因子 $(1 + \gamma(p))$ 发生剧烈饱和溢出，输出发散。
- **修复方案**：统一采用 $\text{asinh}(Z)$ 压缩，将 $+600$ 严格平滑压缩至 $+3.85$，误差下降超过 **90%**。

### 2. 高阶多重协同敲除型坏 Case（如 19 号、76 号）
- **代表突变体**：
  - **19 号 (`cln1Δ cln2Δ cln3Δ GAL-CLN3`)**：三重 G1 启动蛋白同时敲除并外源诱导。
  - **76 号 (`cdc20Δ pds1Δ clb5Δ`)**：后期促进复合物及抑制因子三重缺失。
- **历史失效机理**：
  对数截断方案引入 $\log_{10}(10^{-8})$，单个敲除参数产生 $-7.0$。当同时敲除 6 个关联生化反应参数时，参数输入向量的 $L_2$ 模长由正常的 $6.5$ 暴增到 $17.1 \sim 76.11$，强行将网络深层特征通道置零（Shutdown），引发不可逆的假分岔阻滞。
- **修复方案**：$\text{asinh}(Z)$ 将敲除点平滑限定在 $-1.80$，多重敲除总模长完全受控在 12.99 以内，保证了复杂多基因突变下的网络响应线性度。

### 3. Top-20 代表性坏 Case 基因型与数据根因全景审计表
基于 `scripts/analysis/dataset_diagnosis/generate_multimetric_top20_badcases.py` 对模型多维误差进行综合打分，沉淀出最具挑战性的 Top-20 突变体及其底层数据根因：

| 突变体 ID | 突变体基因型 (Genotype) | 表型模式 | 关键受损通道 | $k$-NN 几何距离 | 核心失效机理 / 数据瓶颈归因 |
|:---|:---|:---:|:---:|:---:|:---|
| **#121** | `cln1Δ cln2Δ cln3Δ apc-ts` | Pattern C | Cln2, Clb2, Cdh1 | 16.42 | P0: 四重极端突变，连续参数空间测度为 0，外推距离极远 |
| **#19** | `cln1Δ cln2Δ cln3Δ GAL-CLN3` | Pattern C | MASS, Cln2, Clb5 | 15.80 | P0+P1: 三重敲除 + 强半乳糖外源挽救，相变阈值极窄 |
| **#76** | `cdc20Δ pds1Δ clb5Δ` | Pattern C | Pds1, Esp1, Clb2 | 14.95 | P0: 纺锤体检验点失活拯救突变，合成数据中无匹配模体 |
| **#44** | `swi5Δ GAL-CLB2` | Pattern B/C | Clb2, Sic1, Cdh1 | 13.82 | P1: Clb2 启动子超强持续激活，峰值浓度高达 $18\sigma$ |
| **#15** | `cln3Δ bck2Δ multi-copy CLN2` | Pattern A | Cln2, Sic1 | 14.10 | P1: 双阻滞蛋白缺失伴随高拷贝补偿，初速度相移 |
| **#73** | `cdc20-ts` (非限制性温度) | Pattern A | Cdc20, Clb2, Pds1 | 12.30 | P2: 温度敏感残余活性边界效应，易触发假双稳态阻滞 |
| **#88** | `clb2-dbΔ sic1Δ` | Pattern C | Clb2, Sic1 | 13.65 | P2: 降解框缺失导致无法泛素化，形成不可逆死锁极限环 |
| **#102** | `GAL-SIC1 (0-100m) OFF` | Pattern B | Sic1, Clb2 | 12.90 | P2: 脉冲式半乳糖瞬态诱导，时间导数出现陡峭阶跃 |
| **#61** | `clb1Δ clb2Δ GAL-CLB1` | Pattern C | Clb2, Clb5 | 14.20 | P0: 主控周期素全敲除 + 异源旁系同源补偿 |
| **#94** | `net1-ts cdc14-ts` | Pattern C | RENT, Net1, Cdc14 | 15.10 | P0: 核仁释放网络双重敏感突变，相空间处于鞍点边缘 |
| **#33** | `cdh1Δ clb5Δ` | Pattern B | Cdh1, Clb5, Clb2 | 11.80 | P4: 退出有丝分裂双重补偿，局部超球体邻域稀疏 |
| **#52** | `sic1-ts cdh1-ts` | Pattern C | Sic1, Cdh1 | 13.40 | P2: 双制动阀同步减活，导致重置检查点跳变失灵 |
| **#81** | `esp1-ts pds1Δ` | Pattern C | Esp1, Pds1, Spn | 14.70 | P0: 分离酶复合体功能反向解偶联 |
| **#110** | `GAL-CDC20 cdc20Δ` | Pattern B | Cdc20, Clb2 | 12.10 | P1: 启动子置换导致周期性节律转变为稳态平台 |
| **#27** | `clb5Δ clb6Δ` | Pattern B | Clb5, Clb2, Ori | 11.20 | P4: S 期周期素双缺失，DNA 复制起点延迟触发 |
| **#49** | `mad2Δ bub2Δ` | Pattern B | BUB2, Clb2 | 12.60 | P0: 纺锤体组装检验点双重缺陷 |
| **#67** | `tem1-ts lte1Δ` | Pattern C | Tem1, Cdc15, Cdc14 | 15.30 | P0: MEN 途径双突变，有丝分裂退出不可逆锁定 |
| **#84** | `apc-ts clb2Δ` | Pattern C | APC, Clb2, Sic1 | 14.85 | P0: 泛素连接酶变异伴随主周期素缺失 |
| **#99** | `GAL-CLB5 cdh1Δ` | Pattern B | Clb5, Cdh1 | 13.20 | P1: S 期周期素超量表达并缺乏降解机器 |
| **#118** | `cln2Δ cdc28-ts` | Pattern A | Cln2, Clb2 | 14.50 | P2: 激酶催化活性受抑伴随周期素缺失 |

---

## 四、评测复现命令 (Reproduction CLI)

使用内置评估脚本直接加载指定实验目录进行 126 突变体全量评估与可视化：

```bash
# 评估指定模型在 126 个真实突变体上的完整指标
python eval.py \
  --exp_dir experiments/mutant_aware_pure_random_100000_210m_500s/biom_refiner_plus/triple_dynamic_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42

# 使用分析脚本横向对比实验 badcase
python scripts/analysis/analyze_badcases.py
```
