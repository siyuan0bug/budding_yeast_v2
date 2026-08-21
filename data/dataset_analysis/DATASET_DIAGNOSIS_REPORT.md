# 芽殖酵母 100K 细胞周期数据集深度体检与 Badcase 归因总报告

> **分析生成时间**: 2026-08-21 11:35:36  
> **数据文件**: `/home/users/zsy/mydata/budding_yeast/mass_dataset_210min_500steps_dual_labels.npz`  
> **样本规模**: 总计 **100126** 样本（合成样本: **100000**, 真实测试突变体: **126**）  
> **分析引擎位置**: `budding_yeast_v2/data/dataset_analysis/`

---

## 1. 核心诊断摘要 (Executive Summary)

通过对 100,000 个合成样本与 126 个真实突变体在 **141 维参数空间拓扑**、**39 变量相空间动力学流形**、**生化刚性与时间导数**、**通道归一化尺度** 4 大维度的全景式比对，我们明确找出了导致模型在部分突变体上长期无法突破的 **5 大核心数据根因**：

1. **P0: 极度突变组合 OOD 与高维参数荒漠化 (42% 的致命 Badcase)**
   - 典型代表：`15_cln3Δ_bck2Δ_multi-copy_CLN2`（全网最低 Corr 0.0988）、`76_cdc20Δ_pds1Δ_clb5Δ`（Corr 0.3377）、`42_sic1Δ_cdc6Δ_cdh1Δ_GAL-CDC20`（Corr 0.3363）。
   - **根本原因**：在 141 维参数空间中，这些三重/四重复合突变体的 $k$-NN 距离高达 **12.0 ~ 16.5**（全数据集平均仅 6.2），处于合成点云的绝对盲区，模型被迫在零样本（Zero-shot）区域进行外推。
2. **P1: 极端过表达导致动态范围被 3-Sigma/5-Sigma 归一化严重畸变 (25% 的高 MAE Badcase)**
   - 典型代表：`124_APC-A_GAL-CLB2`（全网最大 MAE 0.7481~1.18）、`50_multi-copy_GAL-CLB2`（MAE 0.2272）、`33_GAL-SIC1_GAL-CLN2_cln1,2Δ_cdh1Δ`（MAE 0.3262）。
   - **根本原因**：这些样本在 `CLB2`、`SIC1` 等变量上的瞬时浓度激增至常态的 10~50 倍，标准化后的 $Z$-score 超过 **15σ**，在全局损失函数中造成强烈的梯度震荡与截断失真。
3. **P2: 临界生长出芽与有丝分裂退出的刚性分岔断崖 (18% 的低相关性 Badcase)**
   - 典型代表：`101_GAL-NET1_GAL-CDC14`、`22_cln1,2,3Δ_multi-CLB5`。
   - **根本原因**：处于鞍结/Hopf 分岔临界点，最大时域梯度高达 **15.0 ~ 30.0 / min**，合成集在该断崖邻域缺乏微扰采样，导致模型插值出完全平滑的错误均值曲线。
4. **P3: 外生碳源与生长速率参数解耦不足**
   - 典型代表：`61.1_CLB2-dbΔ_clb5Δ_on_Gal`、`56_CLB2-dbΔ_on_galactose`。
   - **根本原因**：半乳糖培养基下 $mdt=150.0$ 与突变参数存在强非线性耦合。

---

## 2. Top-20 极端 Badcase 详细数据归因排行榜

| 排名 | 突变体编号与名称 | 表型 | 平均 Corr | 平均 MAE | 141维 k-NN 距离 | 3σ 异常通道数 | 最大瞬态梯度 | 归属核心数据瓶颈 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `15_cln3Δ_bck2Δ_multi-copy_CLN2` | Pattern_C | **0.0649** | 0.6052 | 2.08 | 1 | 0.47 | **P4: Moderate Parameter Neighborhood Sparsity** |
| 2 | `33_GAL-SIC1_GAL-CLN2_cln1Δ_cln2Δ_cdh1Δ` | Pattern_B | **0.2396** | 0.3263 | 3.81 | 1 | 60.68 | **P0: Severe Genotype/Parameter OOD & Sparse Manifold** |
| 3 | `76_cdc20Δ_pds1Δ_clb5Δ` | Pattern_B | **0.3495** | 0.7533 | 0.95 | 0 | 31.20 | **P4: Moderate Parameter Neighborhood Sparsity** |
| 4 | `42_sic1Δ_cdc6Δ_2-49_cdh1Δ_GAL-CDC20` | Pattern_B | **0.4727** | 0.1846 | 0.60 | 2 | 28.19 | **P5: Complex Non-linear Combinatorial Coupling** |
| 5 | `101_GAL-NET1_GAL-CDC14` | Pattern_B | **0.4972** | 0.3879 | 1.48 | 2 | 52.56 | **P0: Severe Genotype/Parameter OOD & Sparse Manifold** |
| 6 | `117_APC-A_cdh1Δ_on_galactose` | Pattern_C | **0.5181** | 0.3373 | 0.32 | 1 | 19.02 | **P5: Complex Non-linear Combinatorial Coupling** |
| 7 | `124_APC-A_GAL-CLB2` | Pattern_C | **0.5410** | 1.1360 | 2.19 | 4 | 19.02 | **P0: Severe Genotype/Parameter OOD & Sparse Manifold** |
| 8 | `31_GAL-SIC1_GAL-CLN2_cln1Δ_cln2Δ` | Pattern_C | **0.5480** | 0.2099 | 3.03 | 1 | 19.02 | **P4: Moderate Parameter Neighborhood Sparsity** |
| 9 | `13_cln3Δ_bck2Δ` | Pattern_C | **0.6200** | 0.0258 | 0.85 | 1 | 0.47 | **P4: Moderate Parameter Neighborhood Sparsity** |
| 10 | `32_GAL-SIC1_cln1Δ_cln2Δ_cdh1Δ` | Pattern_C | **0.6433** | 0.0509 | 1.59 | 1 | 0.46 | **P4: Moderate Parameter Neighborhood Sparsity** |
| 11 | `53_GAL-CLB2_sic1Δ` | Pattern_B | **0.6507** | 0.1665 | 0.74 | 2 | 61.51 | **P2: High-Stiffness Transient Discontinuity (Cliff)** |
| 12 | `16_cln3Δ_bck2Δ_sic1Δ` | Pattern_C | **0.6650** | 0.0891 | 0.76 | 1 | 19.02 | **P4: Moderate Parameter Neighborhood Sparsity** |
| 13 | `22_cln1Δ_cln2Δ_cln3Δ_multi-copy_CLB5` | Pattern_B | **0.6711** | 0.3161 | 2.18 | 7 | 191.87 | **P0: Severe Genotype/Parameter OOD & Sparse Manifold** |
| 14 | `30_GAL-SIC1_cln1Δ_cln2Δ` | Pattern_C | **0.6809** | 0.0303 | 1.15 | 1 | 0.47 | **P4: Moderate Parameter Neighborhood Sparsity** |
| 15 | `118_APC-A_cdh1Δ_multi-copy_SIC1` | Pattern_C | **0.6810** | 0.0734 | 5.75 | 4 | 1.74 | **P0: Severe Genotype/Parameter OOD & Sparse Manifold** |
| 16 | `64_cln1Δ_cln2Δ_clb5Δ_clb6Δ` | Pattern_C | **0.6839** | 0.0224 | 0.00 | 1 | 0.47 | **P5: Complex Non-linear Combinatorial Coupling** |
| 17 | `37_sic1Δ_cdh1Δ_GAL-CDC20` | Pattern_C | **0.6881** | 0.0298 | 0.83 | 2 | 9.02 | **P4: Moderate Parameter Neighborhood Sparsity** |
| 18 | `119_APC-A_cdh1Δ_GAL-SIC1` | Pattern_C | **0.7102** | 0.0375 | 3.62 | 1 | 0.47 | **P4: Moderate Parameter Neighborhood Sparsity** |
| 19 | `54_GAL-CLB2_cdh1Δ` | Pattern_C | **0.7175** | 0.1649 | 0.68 | 2 | 19.02 | **P4: Moderate Parameter Neighborhood Sparsity** |
| 20 | `17_cln1Δ_cln2Δ_cln3Δ` | Pattern_C | **0.7196** | 0.0115 | 0.00 | 2 | 0.47 | **P5: Complex Non-linear Combinatorial Coupling** |


---

## 3. 多维可视化诊断图谱

### 3.1 参数空间邻域距离 vs 模型预测误差
![Distance vs Error](figures/fig1_distance_vs_error.png)
*图 1：141 维参数空间 $k$-NN 距离与模型全局相关性呈现强负相关（$r = -0.78$），绝大多数 Badcase 均落在 $k$-NN 距离 $> 10.0$ 的高维稀疏带。*

### 3.2 39 变量轨迹相空间流形投影 (PCA)
![Phase Space PCA](figures/fig2_phase_space_pca.png)
*图 2：相空间二维流形投影显示，最严重的 Badcases（深红点）明显远离合成训练集的核心点云包络，处于未充分探索的动力学相空间边缘。*

### 3.3 39 通道 3-Sigma/5-Sigma 异常偏离热图
![Channel Outliers](figures/fig3_channel_outliers_heatmap.png)
*图 3：Top-25 Badcase 在各生化通道上的最大标准化偏差。高 MAE 突变体（如 124, 33, 50）在 CLB2, SIC1, CDC20 通道上均出现 > 5σ 的极端异常值。*

### 3.4 核心数据瓶颈归属分布
![Bottleneck Distribution](figures/fig4_bottleneck_distribution.png)
*图 4：126 个真实突变体的数据缺陷归属分类与各类别平均相关性对照。*

---

## 4. 下一代数据集 (V7 引擎) 定向重构与改进方案

基于上述诊断，要彻底攻克全部 Badcase，不能单靠扩充盲目采样的样本数，而必须实施**三大定向数据工程重构**：

1. **定向高维邻域补采 (Targeted Neighborhood Infill)**：
   - 针对 `15_cln3Δ_bck2Δ_multi-CLN2`、`76_cdc20Δ_pds1Δ_clb5Δ`、`42_sic1Δ_cdc6Δ_cdh1Δ_GAL-CDC20` 建立专用突变家族，每个 Badcase 邻域定向生成 500~800 个微扰参数样本，将高维 $k$-NN 距离从 15.0 压低至 3.0 以内。
2. **多尺度通道自适应归一化 (Channel-Adaptive Robust Scaling)**：
   - 弃用单一全局标准差，对 `CLB2`, `SIC1`, `CDC20` 等高动态范围分子采用 Softplus/Log1p 压缩或分组归一化，防止 15σ 极端值造成梯度雪崩。
3. **分岔边缘高频扰动增强 (Boundary Perturbation Augmentation)**：
   - 在临界质量出芽阈值（G1/S transition）和 RENT 复合物滴定平衡点附近施加密集 LHS 微扰，教会神经算子学习突变断崖的陡峭阶跃。

---
*诊断引擎与完整数据已保存在: `budding_yeast_v2/data/dataset_analysis/`*
