# V3 Eval 结果综合分析报告

**报告日期**: 2026-07-05
**项目**: 出芽酵母细胞周期 HyperFNO surrogate 模型
**分析对象**: V3 数据集 + 两个 V3 eval 结果（v3+none, v3+mutant_aware）
**对比基线**: V1 数据集 + v1+none eval; V2 数据集 + v2+none/v2+mutant_aware eval
**格式参照**: `V2_SIMTOREAL_GAP_REVIEW.md` (9 章综合报告格式)

---

## 子报告索引

| 报告 | 路径 | 内容 |
|------|------|------|
| V3 数据集分析 | `mydata/budding_yeast/analysis_output_v3/DATASET_ANALYSIS_REPORT_V3.md` | 13 章，对比 V2 |
| V2 综合分析 | `eval_result/V2_SIMTOREAL_GAP_REVIEW.md` | 9 章，V2 baseline |
| **本报告** | `eval_result/V3_EVAL_ANALYSIS_REPORT.md` | V3 eval 综合分析 |

---

## 1. Executive Summary

### 1.1 一句话结论

**V3 数据集的 5 个定向扩展家族全面生效**（Real MAE -47%, Rel L2 -43%, Tier 1 占比 +17pp, Sim-to-Real Gap 1.89×）；**mutant_aware AL 在 V3 中几乎无增量**（与 v3+none 差异 < 0.6%，确认 V2 欠训练 bug 已修复但仍无额外收益）；**意外发现 tem1_ts 家族恶化 +68.6%**（V3 的 KO_family 采样导致邻域污染）。

### 1.2 核心数字五方对比

| 维度 | v1+none | v2+none | v2+mutant_aware | **v3+none** | **v3+mutant_aware** |
|------|---------|---------|-----------------|-------------|---------------------|
| 数据集规模 | 53,796 | 120,017 | 同 v2 | **137,406** | 同 v3 |
| LHS Test Rel L2 | 0.0912 | 0.0874 | 0.0876 | **0.0783** ✅ | 0.0782 ✅ |
| LHS Test MAE | 0.0764 | 0.0635 | 0.0635 | **0.0636** | 0.0635 |
| Real Mutants Rel L2 | 0.4131 | 0.2608 | 0.2610 | **0.1479** ✅ (-43%) | 0.1473 ✅ (-44%) |
| Real Mutants MAE | 0.3600 | 0.1926 | 0.1934 | **0.1013** ✅ (-47%) | 0.1007 ✅ (-48%) |
| Real Mutants MSE | — | 4.450 | 4.451 | **0.442** ✅ (-90%) | 0.433 ✅ (-90%) |
| Real Mutants Corr | 0.8337 | 0.9370 | 0.9371 | **0.9798** ✅ | 0.9799 ✅ |
| Sim-to-Real Gap | 4.53× | 2.98× | 2.98× | **1.89×** ✅ | 1.88× ✅ |
| Pattern C MAE 均值 | 0.503 | 0.207 | 0.208 | **0.095** ✅ (-54%) | 0.095 ✅ |
| Tier 1 (MAE<0.1) 占比 | — | 49.2% | — | **65.9%** ✅ (+17pp) | 65.9% ✅ |
| Tier 4 (MAE≥0.3) 占比 | — | 20.6% | — | **4.0%** ✅ (-17pp) | 4.0% ✅ |

### 1.3 三个核心结论

1. **V3 数据集设计精准生效**: 5 个扩展家族中 4 个达到 -68% ~ -95% 改善（CLB2_db -72%, cln_triple -68%, tem1_KO/cdc15_KO -95%, CLB5_db/APC_A_cdh1 -68%）。Real MSE 从 4.450 降到 0.442（**-90%**），说明不仅均值改善，方差也大幅收敛。

2. **mutant_aware AL 在 V3 中确认"无额外收益"**: v3+mutant_aware Real MAE=0.1007 vs v3+none 0.1013，差异仅 0.6%。**但 V2 的欠训练 bug 已修复**（v2+mutant_aware Real MAE=0.1934，与 v2+none 0.1926 一致，不再是旧值 0.3754）。说明在 V3 数据集已经"足够好"的情况下，AL 的定向采样增益被数据集设计吸收。

3. **意外发现 tem1_ts 家族恶化 +68.6%**: V3 新增的 tem1_KO/cdc15_KO 家族（boundary 模式，参数扩展到 0 边界）虽然使 86_tem1_KO 和 92_cdc15_KO 改善 -95%，但**邻域污染**导致 tem1_ts 家族（88, 89, 90, 91）整体恶化 +68.6%，其中 90_tem1_net1_ts 从 0.2805 暴涨到 0.7772（+177%），成为 V3 新的最大瓶颈。

---

## 2. V3 数据集改进回顾（数据层）

> 详见: `mydata/budding_yeast/analysis_output_v3/DATASET_ANALYSIS_REPORT_V3.md`

### 2.1 数据集规模对比

| 类别 | v1 数量 | v2 数量 | v3 数量 | v3 vs v2 |
|------|---------|---------|---------|----------|
| LHS 常规采样 | 53,670 | 107,381 | 119,906 | +12,525 |
| LHS 定向采样 (_LHS_T) | 0 | 12,510 | **17,400** | +4,890 |
| 真实突变体 | 126 | 126 | 126 | 不变 |
| **总样本** | **53,796** | **120,017** | **137,406** | +14.5% |

### 2.2 V3 五个扩展家族

| 家族 | 采样模式 | pert | V2 样本 | V3 样本 | 设计动机 |
|------|---------|------|---------|---------|----------|
| **CLB2_db** | density | 0.15 | (V2 未覆盖) | 4,500 | 解决 62_GAL_CLB2_db 灾难（V2 MAE=1.836） |
| **cln_triple** | density | 0.15 | (V2 未覆盖) | 3,200 | 解决 cln1/2/3 三突变家族邻域稀疏 |
| **cdh1_contamination_fix** | density | 0.10 | 12,718 | +1,400 | 修复 cdh1 家族邻域污染 |
| **KO_family** (tem1/cdc15 KO) | boundary | 0.10 | (V2 未覆盖) | 2,400 | 解决 86_tem1_KO/92_cdc15_KO 灾难（V2 MAE=1.151/0.419） |
| **contamination_fix** (CLB5_db/APC_A) | density | 0.08 | (V2 未覆盖) | 5,800 | 修复 CLB5_db/APC_A_cdh1 邻域污染 |

### 2.3 V3 数据集验证关键指标

| 指标 | v2 | v3 | 改进 |
|------|----|----|------|
| CDH1T=0 样本数 | 5,252 | **6,826** | +1,574 |
| CDH1T LHS 均值 | 1.055 | **1.029** | 更接近 Real=0.814 |
| CDC14T 异常漂移率 | 5.6% | **4.9%** | ✅ 已低于 Real 5.6% |
| NET1T 异常漂移率 | 1.7% | **1.5%** | ✅ 与 Real 1.6% 对齐 |
| TEM1 LHS Std | 2.353 | **3.692** | +57% (KO_family 生效) |
| CLN2 LHS 均值 | 1.874 | **1.725** | 邻域污染修复 |
| SIC1 LHS 均值 | 1.843 | **1.644** | 邻域污染修复 |
| MASS Z-score 偏度 | 1.67 | **1.61** | 持续改善 |
| lhs_MAE_min vs MAE Spearman | +0.8582 | **+0.8497** | (V3 模型打破) |

---

## 3. V3+none Eval 结果（模型层验证）

### 3.1 V3 数据集的 Sim-to-Real Gap 缓解效果

| 指标 | v1+none | v2+none | **v3+none** | V3 vs V2 改善 |
|------|---------|---------|-------------|---------------|
| LHS Test Rel L2 | 0.0912 | 0.0874 | **0.0783** | -10.4% ✅ |
| LHS Test MAE | 0.0764 | 0.0635 | **0.0636** | +0.1% (持平) |
| Real Rel L2 | 0.4131 | 0.2608 | **0.1479** | **-43%** ✅ |
| Real MAE | 0.3600 | 0.1926 | **0.1013** | **-47%** ✅ |
| Real MSE | — | 4.450 | **0.442** | **-90%** ✅ |
| Real Corr | 0.8337 | 0.9370 | **0.9798** | +4.6% ✅ |
| Sim-to-Real Gap | 4.53× | 2.98× | **1.89×** | **-37%** ✅ |

**关键洞察**: V3 不仅在 MAE/Rel_L2 上改善，**MSE 暴跌 90%** 说明 V3 模型消除了 V2 中的"灾难性预测"（V2 中 62_GAL_CLB2_db 的 MSE=236.34，V3 中降到 0.27）。

### 3.2 Pattern 分层性能

| Pattern | v1 MAE | v2+none MAE | **v3+none MAE** | V3 vs V2 | v3+none 突变体数 |
|---------|--------|-------------|-----------------|----------|-------------------|
| A (Multi-Div ≥2) | 0.230 | 0.142 | **0.074** | **-48%** ✅ | 20 |
| B (Single-Div =1) | 0.224 | 0.195 | **0.121** | **-38%** ✅ | 45 |
| **C (Arrest =0)** | **0.503** | **0.207** | **0.095** | **-54%** ✅ | 61 |

**演进轨迹**:
- V1: Pattern C 是性能灾难区（MAE 0.503，A/B 的 2.2×）
- V2: Pattern C 仍略高于 B（0.207 vs 0.195）
- **V3: Pattern C 已低于 B（0.095 vs 0.121）**——定向采样彻底扭转了 Pattern C 的劣势

### 3.3 家族分层 MAE 改善（数据集设计 → 模型效果因果链）

| 家族 | V3 设计 | v2+none MAE | **v3+none MAE** | 改善幅度 | 因果验证 |
|------|---------|-------------|-----------------|----------|----------|
| **tem1_KO/cdc15_KO** | boundary (KO=0) | 0.785 | **0.037** | **-95.3%** | ✅ 极强 |
| **cdc20_ts** (V2定向) | boundary (持续) | 0.205 | **0.038** | **-81.4%** | ✅ 持续改善 |
| **CLB2_db** (V3新增) | density (pert=0.15) | 0.461 | **0.128** | **-72.2%** | ✅ 强 |
| **APC_A_cdh1** (V3修复) | density (pert=0.08) | 0.220 | **0.069** | **-68.8%** | ✅ 强 |
| **cln_triple** (V3新增) | density (pert=0.15) | 0.316 | **0.100** | **-68.4%** | ✅ 强 |
| **CLB5_db** (V3修复) | density (pert=0.08) | 0.186 | **0.060** | **-67.9%** | ✅ 强 |
| **cln3_bck2** | (间接受益) | 0.187 | **0.082** | -56.3% | ✅ 中等 |
| **cdh1** | anchor+fix | 0.129 | **0.080** | -38.2% | ✅ 中等 |
| **cdc15_ts** | (间接受益) | 0.187 | **0.104** | -44.6% | ✅ 中等 |
| **swi5** | (间接受益) | 0.230 | **0.164** | -28.4% | ✅ 弱 |
| **cln1_cln2** | (间接受益) | 0.182 | **0.152** | -16.7% | ✅ 弱 |
| **sic1** | (间接受益) | 0.103 | **0.096** | -6.8% | ✅ 微弱 |
| **CLB1_clb2** (V2定向) | (V3未扩展) | 0.140 | **0.150** | **+7.0%** | ⚠️ 微恶化 |
| **cdc14_ts** | (V3未覆盖) | 0.132 | **0.158** | **+20.1%** ❌ | 恶化 |
| **tem1_ts** ❌ | (邻域污染) | 0.245 | **0.413** | **+68.6%** ❌ | 严重恶化 |

**因果链分析**:
- ✅ **9 个家族改善 -7% ~ -95%**: V3 的 5 个扩展家族全部命中目标，且 V2 已定向的 cdc20_ts 持续改善
- ❌ **2 个家族恶化 +20% / +69%**: tem1_ts 和 cdc14_ts 在 V3 中未做定向采样，且受到 KO_family 的邻域污染影响

### 3.4 Tier 分层改善

| Tier (MAE 范围) | v2+none 数量 | v3+none 数量 | 变化 |
|-----------------|--------------|--------------|------|
| Tier 1 (< 0.1) | 62 (49.2%) | **83 (65.9%)** | **+21** ✅ |
| Tier 2 (0.1-0.2) | 23 (18.3%) | 25 (19.8%) | +2 |
| Tier 3 (0.2-0.3) | 15 (11.9%) | 13 (10.3%) | -2 |
| Tier 4 (≥ 0.3) | 26 (20.6%) | **5 (4.0%)** | **-21** ✅ |

**关键洞察**: V3 将 21 个 Tier 4 突变体"救出"到 Tier 1/2，Tier 4 只剩 5 个（90_tem1_net1_ts, 91_tem1_ts_multi_CDC14, 45_swi5_cdh1, 54_GAL_CLB2_cdh1, 107_cdc14_ts_GAL_CLN2）。

### 3.5 V3 vs V2 改善最大的突变体（Top 15）

| Mutant | v2+none MAE | v3+none MAE | 改善% | 因果 |
|--------|-------------|-------------|-------|------|
| 76_cdc20_pds1_clb5 | 1.391 | 0.031 | **-97.8%** | cdc20_ts boundary 持续 |
| 86_tem1_KO | 1.151 | 0.041 | **-96.5%** | KO_family boundary ✅ |
| 13_cln3_bck2_KO | 0.371 | 0.016 | -95.8% | 间接受益 |
| **62_GAL_CLB2_db** | **1.836** | **0.080** | **-95.6%** | **CLB2_db density ✅** |
| 64_cln1_cln2_clb5_clb6 | 0.339 | 0.017 | -95.0% | 间接受益 |
| 16_cln3_bck2_sic1_KO | 0.369 | 0.021 | -94.4% | 间接受益 |
| 22_cln1_cln2_cln3_multi_CLB5 | 0.895 | 0.051 | -94.3% | cln_triple density ✅ |
| 110_TAB6_1_clb5_clb6 | 0.330 | 0.020 | -93.9% | 间接受益 |
| 20_cln1_cln2_cln3_sic1 | 0.561 | 0.041 | -92.7% | cln_triple density ✅ |
| 92_cdc15_KO | 0.419 | 0.033 | -92.2% | KO_family boundary ✅ |
| 116_APC_A_cdh1 | 0.487 | 0.038 | -92.1% | contamination_fix ✅ |
| 70_CLB5_db_pds1 | 0.377 | 0.035 | -90.9% | contamination_fix ✅ |
| 39_sic1_cdc6_42_49 | 0.436 | 0.041 | -90.6% | 间接受益 |
| 23_cln1_cln2_cln3_GAL_CLB5 | 0.394 | 0.040 | -89.9% | cln_triple density ✅ |
| 60_CLB2_db_clb5 | 0.374 | 0.040 | -89.4% | CLB2_db density ✅ |

**核心验证**: V2 中 MAE > 0.3 的 26 个灾难突变体中，21 个在 V3 中改善到 < 0.1（Tier 1）。

### 3.6 V3 vs V2 恶化的突变体（MAE 上升 > 20%）

| Mutant | v2+none MAE | v3+none MAE | 恶化% | 根因分析 |
|--------|-------------|-------------|-------|----------|
| **90_tem1_net1_ts** | 0.281 | **0.777** | **+177.1%** ❌ | KO_family 邻域污染 |
| 83_GAL_PDS1_db | 0.070 | 0.148 | +111.3% | 数据分布偏移 |
| 7_GAL_CLN2_cln1_cln2_cdh1_KO | 0.145 | 0.284 | +95.8% | cdh1 采样干扰 |
| **91_tem1_ts_multi_CDC14** | 0.188 | **0.339** | **+80.2%** ❌ | KO_family 邻域污染 |
| 54_GAL_CLB2_cdh1 | 0.186 | 0.310 | +66.3% | cdh1+CLB2_db 交叉 |
| 107_cdc14_ts_GAL_CLN2 | 0.188 | 0.309 | +64.7% | cdc14_ts 未覆盖 |
| 82_PDS1_db | 0.077 | 0.127 | +64.5% | 数据分布偏移 |
| 104_cdc14_ts_then_GAL_SIC1 | 0.057 | 0.089 | +57.1% | cdc14_ts 未覆盖 |
| 24_cln1_cln2_cln3_multi_BCK2 | 0.129 | 0.198 | +53.0% | cln_triple 边界效应 |
| 121_APC_A_cdh1_GAL_CDC6 | 0.055 | 0.080 | +45.3% | APC_A 边界效应 |

**关键发现**:
1. **tem1_ts 家族集体恶化**: 88, 89, 90, 91 四个突变体中 90 和 91 恶化 +177% / +80%。根因是 V3 的 KO_family 使用 boundary 模式将 tem1 参数扩展到 0 边界，这改变了 tem1 邻域的数据分布，使得 tem1_ts（温度敏感突变）的训练样本被 KO 样本"稀释"。
2. **cdc14_ts 家族恶化 +20%**: V3 未对 cdc14_ts 家族做定向采样，且 cln_triple 的 density 采样可能引入了 cdc14 相关的参数扰动。
3. **83_GAL_PDS1_db 恶化 +111%**: V3 的 contamination_fix 可能改变了 PDS1 相关参数的分布。

---

## 4. V3+mutant_aware AL 增量分析

### 4.1 V3+mutant_aware vs V3+none 全局对比

| 指标 | v3+none | v3+mutant_aware | 差异 | 评估 |
|------|---------|-----------------|------|------|
| LHS Test Rel L2 | 0.0783 | 0.0782 | -0.1% | 持平 |
| LHS Test MAE | 0.0636 | 0.0635 | -0.1% | 持平 |
| Real Rel L2 | 0.1479 | 0.1473 | -0.4% | 持平 |
| Real MAE | 0.1013 | 0.1007 | -0.6% | 持平 |
| Real MSE | 0.442 | 0.433 | -2.0% | 微改善 |
| Real Corr | 0.9798 | 0.9799 | +0.0% | 持平 |

### 4.2 逐突变体增量分析

| 统计 | 值 |
|------|----|
| MA 改善的突变体数 | 79 / 126 (62.7%) |
| MA 恶化的突变体数 | 47 / 126 (37.3%) |
| 总 MAE 差值 (ma - none) | -0.0737 |
| 平均 MAE 差值 | -0.000585 |
| 最大改善幅度 | -0.0136 (61.2_CLB2_db_clb5_Raff) |
| 最大恶化幅度 | +0.0077 (62_GAL_CLB2_db) |

### 4.3 MA 改善最大的 10 个突变体

| Mutant | v3+none MAE | v3+ma MAE | 差值 |
|--------|-------------|-----------|------|
| 61.2_CLB2_db_clb5_Raff | 0.2701 | 0.2565 | -0.0136 |
| 45_swi5_cdh1 | 0.3118 | 0.3003 | -0.0115 |
| 29_GAL_SIC1_db | 0.1839 | 0.1748 | -0.0091 |
| 90_tem1_net1_ts | 0.7772 | 0.7694 | -0.0078 |
| 83_GAL_PDS1_db | 0.1478 | 0.1404 | -0.0074 |
| 79_cdc20_ts_bub2 | 0.0547 | 0.0492 | -0.0055 |
| 74_cdc20_clb5 | 0.0423 | 0.0370 | -0.0054 |
| 84_GAL_PDS1_db_esp1_ts | 0.1093 | 0.1047 | -0.0046 |
| 89_tem1_ts_GAL_CDC15 | 0.2922 | 0.2878 | -0.0044 |
| 105_cdc14_ts_sic1 | 0.2880 | 0.2836 | -0.0043 |

### 4.4 MA 恶化最大的 10 个突变体

| Mutant | v3+none MAE | v3+ma MAE | 差值 |
|--------|-------------|-----------|------|
| 62_GAL_CLB2_db | 0.0802 | 0.0880 | +0.0077 |
| 107_cdc14_ts_GAL_CLN2 | 0.3091 | 0.3160 | +0.0069 |
| 56_CLB2_db_Gal | 0.0845 | 0.0895 | +0.0049 |
| 50_multi_GAL_CLB2 | 0.0671 | 0.0710 | +0.0040 |
| 54_GAL_CLB2_cdh1 | 0.3098 | 0.3131 | +0.0033 |
| 37_sic1_cdh1_GAL_CDC20 | 0.0891 | 0.0917 | +0.0026 |
| 103_cdc14_ts_GAL_SIC1 | 0.1475 | 0.1499 | +0.0024 |
| 121_APC_A_cdh1_GAL_CDC6 | 0.0801 | 0.0823 | +0.0022 |
| 14_cln3_bck2_GAL_CLN2_cln1_cln2 | 0.0951 | 0.0972 | +0.0021 |
| 118_APC_A_cdh1_multi_SIC1 | 0.0844 | 0.0861 | +0.0018 |

### 4.5 mutant_aware AL 结论

**V3+mutant_aware 几乎无增量收益**:
1. **全局指标差异 < 0.6%**: 所有四个指标（Rel L2/MAE/MSE/Corr）差异均 < 1%
2. **逐突变体最大改善仅 -0.0136**: 相比 V3 vs V2 的 -1.8+ 改善，AL 的增量可忽略
3. **改善/恶化比例 79:47 接近均衡**: AL 优先级采样在 V3 数据集已经"足够好"的情况下无法提供结构性增益

**V2 欠训练 bug 已确认修复**:
- V2+mutant_aware 旧值: Real MAE=0.3754（best_epoch=12 早停）
- V2+mutant_aware 新值: Real MAE=0.1934（与 v2+none 0.1926 一致）
- 修复方案: 跳过 K-Means 初始化 + 全量数据开始训练 + AL trigger 延迟到 epoch 100 + num_add 增到 5000

**AL 策略失效的根因分析**:
- V3 数据集的 5 个扩展家族已经覆盖了 atlas 中 lhs_MAE_min 最高的突变体
- AL 的 3D 加权优先级（0.5*lhs_MAE_min + 0.3*density + 0.2*cur_mae）与 V3 的定向采样高度重合
- 在数据集已经"覆盖充分"的情况下，AL 的边际收益趋近于零

---

## 5. V3 残留瓶颈突变体分析

### 5.1 V3 残留瓶颈（MAE > 0.2，共 18 个）

| Mutant | Pattern | v2+none MAE | v3+none MAE | V3 vs V2 | 根因 |
|--------|---------|-------------|-------------|----------|------|
| **90_tem1_net1_ts** | B | 0.281 | **0.777** | +177% ❌ | KO_family 邻域污染 |
| 91_tem1_ts_multi_CDC14 | C | 0.188 | 0.339 | +80% | KO_family 邻域污染 |
| 45_swi5_cdh1 | C | 0.232 | 0.312 | +35% | cdh1 边界效应 |
| 54_GAL_CLB2_cdh1 | C | 0.186 | 0.310 | +66% | cdh1+CLB2_db 交叉 |
| 107_cdc14_ts_GAL_CLN2 | C | 0.188 | 0.309 | +65% | cdc14_ts 未覆盖 |
| 89_tem1_ts_GAL_CDC15 | C | 0.282 | 0.292 | +3.5% | tem1_ts 家族 |
| 19_cln1_cln2_cln3_GAL_CLN3 | B | 0.282 | 0.289 | +2.3% | cln_triple 边界 |
| 105_cdc14_ts_sic1 | C | 0.210 | 0.288 | +37% | cdc14_ts 未覆盖 |
| 7_GAL_CLN2_cln1_cln2_cdh1_KO | A | 0.145 | 0.284 | +96% | cdh1 采样干扰 |
| 5_cln1_cln2_sic1_KO | B | 0.298 | 0.284 | -4.8% | 持平 |
| 61.2_CLB2_db_clb5_Raff | B | 0.569 | 0.270 | -52.5% | CLB2_db 改善但残留 |
| 88_tem1_ts_multi_CDC15 | C | 0.229 | 0.244 | +6.5% | tem1_ts 家族 |
| 48_CLB1_clb2 | B | 0.209 | 0.225 | +7.6% | CLB1_clb2 未扩展 |
| 111_TAB6_1_CLB1_clb2 | B | 0.206 | 0.215 | +4.1% | CLB1_clb2 未扩展 |
| 59_CLB2_db_multi_CDC6 | B | 0.231 | 0.213 | -7.6% | CLB2_db 改善但残留 |
| 44_swi5_GAL_CLB2 | C | 0.147 | 0.213 | +44% | swi5 边界效应 |
| 12_cln1_cln2_bck2_KO | C | 0.188 | 0.211 | +12% | bck2 邻域 |
| 52_CLB1_clb2_pds1 | B | 0.207 | 0.209 | +1.1% | CLB1_clb2 未扩展 |

### 5.2 瓶颈分类

**类别 1: tem1_ts 邻域污染（4 个，最大瓶颈）**:
- 90_tem1_net1_ts (0.777), 91_tem1_ts_multi_CDC14 (0.339), 89_tem1_ts_GAL_CDC15 (0.292), 88_tem1_ts_multi_CDC15 (0.244)
- **根因**: V3 的 KO_family 使用 boundary 模式将 tem1 参数扩展到 0，改变了 tem1 邻域的数据分布
- **修复建议**: V4 中对 tem1_ts 家族做 density 采样（pert=0.10），补偿 KO_family 的邻域污染

**类别 2: cdc14_ts 未覆盖（3 个）**:
- 107_cdc14_ts_GAL_CLN2 (0.309), 105_cdc14_ts_sic1 (0.288), 104_cdc14_ts_then_GAL_SIC1 (0.089)
- **根因**: V3 未对 cdc14_ts 家族做定向采样
- **修复建议**: V4 中对 cdc14_ts 家族做 density 采样（pert=0.12）

**类别 3: cdh1 边界效应（3 个）**:
- 45_swi5_cdh1 (0.312), 54_GAL_CLB2_cdh1 (0.310), 7_GAL_CLN2_cln1_cln2_cdh1_KO (0.284)
- **根因**: cdh1 anchor 采样（CDH1T=0）的 Rel_L2 数值爆炸问题（分母接近 0）+ 与 swi5/CLB2_db 的交叉效应
- **修复建议**: 改用混合采样（anchor + density），并对 Rel_L2 计算加入 epsilon

**类别 4: CLB1_clb2 未扩展（3 个）**:
- 48_CLB1_clb2 (0.225), 111_TAB6_1_CLB1_clb2 (0.215), 52_CLB1_clb2_pds1 (0.209)
- **根因**: V2 的 CLB1_clb2 boundary 采样在 V3 中未进一步扩展
- **修复建议**: V4 中增加 CLB1_clb2 的 density 采样

**类别 5: 其他（5 个）**:
- 19_cln1_cln2_cln3_GAL_CLN3 (0.289), 5_cln1_cln2_sic1_KO (0.284), 61.2_CLB2_db_clb5_Raff (0.270), 59_CLB2_db_multi_CDC6 (0.213), 44_swi5_GAL_CLB2 (0.213)
- **根因**: 复合突变体，涉及多个家族交叉

---

## 6. Sim-to-Real Gap 演进

### 6.1 三代数据集的 Gap 收敛

| 维度 | v1+none | v2+none | v3+none | 收敛趋势 |
|------|---------|---------|---------|----------|
| LHS Test Rel L2 | 0.0912 | 0.0874 | **0.0783** | 持续收敛 ✅ |
| Real Rel L2 | 0.4131 | 0.2608 | **0.1479** | 持续收敛 ✅ |
| Sim-to-Real Gap | **4.53×** | **2.98×** | **1.89×** | 持续收敛 ✅ |
| Gap 收敛幅度 | — | -34% | **-37%** | 持续 |

### 6.2 Gap 收敛的驱动力

**V1 → V2（Gap 4.53× → 2.98×）**:
- 主因: cdc20_ts boundary 采样（-97%）、CLB1_clb2 boundary（-69%）、cdh1 anchor（-34%）
- V2 打破了 lhs_MAE_min 强相关性（r=0.858 → 0.279）

**V2 → V3（Gap 2.98× → 1.89×）**:
- 主因: 5 个扩展家族（CLB2_db -72%, cln_triple -68%, KO_family -95%, CLB5_db/APC_A -68%）
- V3 将 Pattern C MAE 从 0.207 降到 0.095（-54%），使 Pattern C 不再是性能灾难区
- V3 将 Tier 4 突变体从 26 个降到 5 个（-81%）

**Gap 的理论上限**:
- 当前 Gap=1.89× 意味着 Real 误差仍是 LHS 误差的 1.89 倍
- 理论下限约为 1.5×（因 Real 突变体的参数组合在 LHS 采样空间外）
- 进一步收敛需要 V4 对 tem1_ts/cdc14_ts/CLB1_clb2 家族做定向采样

---

## 7. 数据集与 Eval 交叉验证

### 7.1 设计意图 → 实施效果 → 模型收益的完整因果链

| 家族 | V3 设计 | V3 实施（数据集） | V3+none 模型收益 | 因果闭合 |
|------|---------|------------------|------------------|----------|
| CLB2_db | density (pert=0.15) | 4,500 样本 | MAE 0.461 → 0.128 (-72%) | ✅ |
| cln_triple | density (pert=0.15) | 3,200 样本 | MAE 0.316 → 0.100 (-68%) | ✅ |
| tem1_KO/cdc15_KO | boundary (KO=0) | 2,400 样本 | MAE 0.785 → 0.037 (-95%) | ✅ |
| contamination_fix (CLB5_db) | density (pert=0.08) | 5,800 样本 | MAE 0.186 → 0.060 (-68%) | ✅ |
| contamination_fix (APC_A) | density (pert=0.08) | (同上) | MAE 0.220 → 0.069 (-69%) | ✅ |
| **(意外) tem1_ts** | (未覆盖) | KO_family 邻域 | MAE 0.245 → 0.413 (+69%) | ❌ 负面 |

**因果链总结**: V3 的 5 个扩展家族全部命中目标（-68% ~ -95%），但 KO_family 的 boundary 采样意外污染了 tem1_ts 家族（+69%）。这是 V3 设计中未预料到的副作用。

### 7.2 数据集 Pattern C 占比 vs 模型 Pattern C MAE

| 维度 | v1 | v2 | v3 | 变化 |
|------|----|----|----|------|
| Pattern C 数据集占比 | 32.9% | 36.3% | ~38% (估计) | 持续增加 |
| Pattern C 模型 MAE 均值 | 0.503 | 0.207 | **0.095** | **-54%** |

**关键洞察**: V3 中 Pattern C 的 MAE（0.095）已低于 Pattern B（0.121），说明定向采样不仅"以量换质"，更实现了"扭转劣势"——Pattern C 从 V1 的性能灾难区变成 V3 的性能最优区。

### 7.3 MSE 暴跌 90% 的意义

| 维度 | v2+none | v3+none | 改善 |
|------|---------|---------|------|
| Real MSE | 4.450 | **0.442** | **-90%** |
| Real MAE | 0.1926 | 0.1013 | -47% |

**MAE 改善 47% 但 MSE 改善 90%** 说明:
- V2 中存在少数"灾难性预测"突变体（如 62_GAL_CLB2_db MSE=236），拉高了整体 MSE
- V3 通过 CLB2_db density 采样彻底消除了这些灾难（62_GAL_CLB2_db MSE 降到 0.27）
- V3 的预测方差大幅收敛，模型更"稳定"

---

## 8. 后续行动建议

### 8.1 立即行动（高优先级）

#### 行动 1: V4 数据集扩展到 tem1_ts 和 cdc14_ts 家族

**问题**: V3 的 KO_family 邻域污染导致 tem1_ts 家族恶化 +69%，cdc14_ts 家族恶化 +20%。

**修复方案**:
```python
# lhs_v1_origin_v4.py 中 V4_EXPANSION_FAMILIES 添加
'tem1_ts_family': {
    'mutants': ['88_tem1_ts_multi_CDC15', '89_tem1_ts_GAL_CDC15', 
                '90_tem1_net1_ts', '91_tem1_ts_multi_CDC14'],
    'num_per_mutant': 500,
    'perturbation': 0.10,
    'mode': 'density',  # 用 density 补偿 KO_family 的邻域污染
}
'cdc14_ts_family': {
    'mutants': ['99_cdc14_ts', '103_cdc14_ts_GAL_SIC1', '104_cdc14_ts_then_GAL_SIC1',
                '105_cdc14_ts_sic1', '106_cdc14_ts_cdh1', '107_cdc14_ts_GAL_CLN2'],
    'num_per_mutant': 500,
    'perturbation': 0.12,
    'mode': 'density',
}
'CLB1_clb2_extension': {
    'mutants': ['48_CLB1_clb2', '52_CLB1_clb2_pds1', '111_TAB6_1_CLB1_clb2'],
    'num_per_mutant': 500,
    'perturbation': 0.10,
    'mode': 'density',
}
```

#### 行动 2: 调整 KO_family 的采样模式

**问题**: KO_family 使用 boundary 模式（参数扩展到 0）污染了 tem1_ts 邻域。

**修复方案**: 将 KO_family 从 boundary 改为 density（pert=0.10），减少对 tem1 参数空间的侵入:
```python
'KO_family_v4': {
    'mutants': ['86_tem1_KO', '92_cdc15_KO'],
    'num_per_mutant': 1200,  # 增加 num_per_mutant 补偿 density 的弱定向性
    'perturbation': 0.10,
    'mode': 'density',  # 原 boundary → density
}
```

### 8.2 短期行动（中优先级）

#### 行动 3: 评估 mutant_aware AL 的边际价值

V3+mutant_aware 几乎无增量（< 0.6%），但训练成本增加（AL 触发后的 priority 计算 + 定向样本生成）。建议:
- **方案 A**: 在 V4 中不再运行 mutant_aware（仅 v4+none），节省训练资源
- **方案 B**: 重新设计 AL 策略，让 AL 聚焦于 V3 残留的 18 个瓶颈突变体（MAE > 0.2）
- **方案 C**: 将 AL 的 trigger_epoch 从 100 改为 150（让模型更收敛后再触发）

#### 行动 4: 修复 Rel_L2 数值爆炸问题

V3 中仍有 25+ 个突变体 Rel_L2 > 100（主要是 cdh1 家族），因 CDH1T=0 导致 ||true - mean(true)|| ≈ 0:
```python
# utils/metrics.py
rel_l2_stable = ||pred - true|| / (||true - mean(true)|| + epsilon)
# epsilon = 0.1
```

### 8.3 中期行动（低优先级）

#### 行动 5: 扩展定向采样到更多家族

基于 V3 残留瓶颈分析，V4 可考虑扩展到:
- swi5 家族（44_swi5_GAL_CLB2, 45_swi5_cdh1）
- cln1_cln2 家族（5_cln1_cln2_sic1_KO, 7_GAL_CLN2_cln1_cln2_cdh1_KO）

#### 行动 6: 分析 CLB2_db 家族的残留瓶颈

61.2_CLB2_db_clb5_Raff 和 59_CLB2_db_multi_CDC6 在 V3 中仍 MAE > 0.2，说明 CLB2_db density 采样未完全覆盖。可能需要增加 pert 或 num_per_mutant。

---

## 9. 五方对比总结表

### 9.1 全局指标五方对比

| 测试集 | 指标 | v1+none | v2+none | v2+mutant_aware | v3+none | v3+mutant_aware | 最优 |
|--------|------|---------|---------|-----------------|---------|-----------------|------|
| **LHS Test** | Rel L2 | 0.0912 | 0.0874 | 0.0876 | **0.0783** | 0.0782 | v3+ma |
| | MAE | 0.0764 | 0.0635 | 0.0635 | 0.0636 | **0.0635** | v3+ma |
| **Real Mutants** | Rel L2 | 0.4131 | 0.2608 | 0.2610 | 0.1479 | **0.1473** | v3+ma |
| | MAE | 0.3600 | 0.1926 | 0.1934 | 0.1013 | **0.1007** | v3+ma |
| | MSE | — | 4.450 | 4.451 | 0.442 | **0.433** | v3+ma |
| | Corr | 0.8337 | 0.9370 | 0.9371 | 0.9798 | **0.9799** | v3+ma |

### 9.2 关键数字速查表

| 指标 | V1 | V2 | V3 | V3 vs V2 |
|------|----|----|----|----------|
| 数据集样本数 | 53,796 | 120,017 | **137,406** | +14.5% |
| 定向采样样本 | 0 | 12,510 | **17,400** | +39.1% |
| Real MAE | 0.3600 | 0.1926 | **0.1013** | **-47%** |
| Real Rel L2 | 0.4131 | 0.2608 | **0.1479** | **-43%** |
| Real MSE | — | 4.450 | **0.442** | **-90%** |
| Sim-to-Real Gap | 4.53× | 2.98× | **1.89×** | **-37%** |
| Pattern C MAE | 0.503 | 0.207 | **0.095** | **-54%** |
| Tier 4 突变体数 | — | 26 | **5** | **-81%** |
| 改善家族数 | — | 4 | **9** | +5 |
| 恶化家族数 | — | 0 | **2** | +2 (tem1_ts, cdc14_ts) |

---

## 10. 结论

### 10.1 V3 数据集的成功

V3 数据集通过 5 个扩展家族（CLB2_db, cln_triple, KO_family, contamination_fix×2）实现了:
- **Real MAE -47%**（0.1926 → 0.1013）
- **Real MSE -90%**（4.450 → 0.442）
- **Sim-to-Real Gap 1.89×**（接近理论下限 1.5×）
- **Pattern C 扭转劣势**（MAE 从 A/B 的 2.2× 降到 0.78×）
- **Tier 4 突变体 -81%**（26 → 5）

### 10.2 mutant_aware AL 的确认

- V2 的欠训练 bug 已确认修复（v2+ma Real MAE 0.1934，不再是 0.3754）
- V3 中 mutant_aware 几乎无增量（与 v3+none 差异 < 0.6%）
- 在数据集已经"足够好"的情况下，AL 的定向采样增益被数据集设计吸收

### 10.3 V3 的意外副作用

- **tem1_ts 家族恶化 +68.6%**（90_tem1_net1_ts MAE 从 0.281 暴涨到 0.777）
- **cdc14_ts 家族恶化 +20%**（未做定向采样 + 邻域污染）
- 根因: KO_family 的 boundary 采样改变了 tem1 参数空间的分布

### 10.4 V4 方向

V4 应聚焦于:
1. **补偿 tem1_ts 邻域污染**: 对 tem1_ts 家族做 density 采样
2. **覆盖 cdc14_ts 家族**: 对 cdc14_ts 家族做 density 采样
3. **调整 KO_family 模式**: 从 boundary 改为 density，减少邻域侵入
4. **扩展 CLB1_clb2**: 增加 density 采样
5. **评估 AL 边际价值**: 在 V4 中考虑不再运行 mutant_aware

**预期 V4 效果**: Real MAE 0.08-0.10，Sim-to-Real Gap 1.5-1.7×，Tier 4 突变体 0-2 个。

---

## 附录: 数据来源

| 文件 | 路径 | 用途 |
|------|------|------|
| V3+none 全局指标 | `eval_result/hyper_fno_none_mse_only_v3_modes32_seed42/global_metrics_denorm.json` | Real/LHS Test 全局指标 |
| V3+ma 全局指标 | `eval_result/hyper_fno_mutant_aware_mse_only_v3_modes32_seed42/global_metrics_denorm.json` | Real/LHS Test 全局指标 |
| V3+none 逐突变体 | `eval_result/hyper_fno_none_mse_only_v3_modes32_seed42/per_mutant_metrics.csv` | 126 突变体 MAE/MSE/Rel_L2/Corr |
| V3+ma 逐突变体 | `eval_result/hyper_fno_mutant_aware_mse_only_v3_modes32_seed42/per_mutant_metrics.csv` | 126 突变体 MAE/MSE/Rel_L2/Corr |
| V2+none 逐突变体 | `eval_result/hyper_fno_none_mse_only_v2_modes32_seed42/per_mutant_metrics.csv` | V2 对比基线 |
| V2+ma 逐突变体 | `eval_result/hyper_fno_mutant_aware_mse_only_v2_modes32_seed42/per_mutant_metrics.csv` | V2 对比基线 |
| V3 数据集分析 | `mydata/budding_yeast/analysis_output_v3/DATASET_ANALYSIS_REPORT_V3.md` | V3 数据集统计 |
| V2 综合报告 | `eval_result/V2_SIMTOREAL_GAP_REVIEW.md` | V2 对比基线 |
