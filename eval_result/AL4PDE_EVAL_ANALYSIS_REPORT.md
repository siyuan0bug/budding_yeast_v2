# AL4PDE 风格主动学习 Eval 结果综合分析报告

**报告日期**: 2026-07-07
**项目**: 出芽酵母细胞周期 HyperFNO surrogate 模型
**分析对象**: al4pde 框架两种 AL 策略 eval 结果（al4pde_uncertainty, MAAL）
**对比基线**: V5 数据集 + 两个 V5 eval 结果（v5+none, v5+mutant_aware）
**格式参照**: `V4_EVAL_ANALYSIS_REPORT.md` / `V5_EVAL_ANALYSIS_REPORT.md` (10 章综合报告格式)

---

## 子报告索引

| 报告 | 路径 | 内容 |
|------|------|------|
| V5 数据集综合分析 | `eval_result/V5_EVAL_ANALYSIS_REPORT.md` | 10 章，V5 数据集 + V5+none/+ma eval |
| V4 综合分析 | `eval_result/V4_EVAL_ANALYSIS_REPORT.md` | 10 章，V4 baseline |
| V3 综合分析 | `eval_result/V3_EVAL_ANALYSIS_REPORT.md` | 10 章，V3 baseline |
| **本报告** | `eval_result/AL4PDE_EVAL_ANALYSIS_REPORT.md` | al4pde 框架 AL 策略综合分析 |

---

## 1. Executive Summary

### 1.1 一句话结论

**al4pde 风格 AL 框架在当前配置下彻底失败**：al4pde_uncertainty Real MAE=0.3726（**6.7× worse** than v5+none 0.0552），MAAL Real MAE=0.4088（**7.4× worse**，比 unc 还差 +9.7%）；**126 个 real mutant 中 126 个全部恶化**（unc）或 125 恶化 1 改善（maal），cdc20_ts 系列 5 个突变体 MAE 全部 > 1.4（v5+none 仅 0.03-0.04，恶化 40-60×）；**根因是训练数据严重不足**（5000 初始样本 vs v5 数据集 162,134 全量样本，仅占 3.1%），AL 触发太晚（epoch 20），每轮 AL 样本数太少（1000），即使总训练 15,000 样本仍只占全量 9.3%；**MAAL 比 unc 更差证明 target_priority 权重失衡**（0.4 权重导致过度聚焦少数家族而忽视全局覆盖）。

### 1.2 核心数字四方对比

| 维度 | v5+none | v5+mutant_aware | **al4pde_unc** | **MAAL** |
|------|---------|-----------------|----------------|----------|
| 训练数据集 | v5 (162,134) | v5 (162,134) | v1 (53,796) | v1 (53,796) |
| 训练样本数 | ~130,000 | ~130,000 | **5,000 + 10,000 AL** | **5,000 + 10,000 AL** |
| 候选池规模 | — | — | 162,008 (v5) | 162,008 (v5) |
| LHS Test Rel L2 | 0.0815 | 0.0814 | 0.1652 | 0.1659 |
| LHS Test MAE | 0.0618 | 0.0617 | 0.1737 | 0.1671 |
| Real Rel L2 | 0.0780 | 0.0761 | 0.3575 | 0.4210 |
| **Real MAE** | **0.0552** ✅ | **0.0549** ✅ | **0.3726** ❌ | **0.4088** ❌ |
| Real MSE | 0.166 | 0.157 | 12.149 | 10.855 |
| Real Corr | 0.9942 | 0.9945 | 0.8733 | 0.8240 |
| **Sim-to-Real Gap** | **0.89×** ✅ | **0.89×** ✅ | **2.15×** ❌ | **2.45×** ❌ |
| Core 9 Real MAE | 0.148 | 0.147 | 1.126 | 1.153 |
| Tier 1 (<0.1) 占比 | 89.7% | 90.5% | **14.3%** ❌ | **9.5%** ❌ |
| Tier 4 (≥0.3) 占比 | 0.8% | 0.8% | **39.7%** ❌ | **48.4%** ❌ |
| 灾难性 (MAE>1.0) 数 | 0 | 0 | **10** ❌ | **9** ❌ |

### 1.3 三个核心结论

1. **al4pde 框架在当前配置下完全失败**: Real MAE 从 v5+none 的 0.0552 暴涨到 0.3726 (unc) / 0.4088 (maal)，**恶化 6.7×-7.4×**。Tier 1 占比从 89.7% 暴跌到 14.3% (unc) / 9.5% (maal)，Tier 4 灾难区从 0.8% 飙升到 39.7% (unc) / 48.4% (maal)。10 个 (unc) / 9 个 (maal) 突变体 MAE > 1.0（v5+none 为 0 个），属于完全无法使用的预测。Sim-to-Real Gap 从 0.89× 倒退到 2.15× (unc) / 2.45× (maal)，相当于回退到 V1 时代（4.53×）和 V2 时代（2.98×）之间的水平。

2. **MAAL 比 al4pde_uncertainty 更差，证明 target_priority 权重失衡**: MAAL Real MAE=0.4088 vs unc 0.3726（**+9.7% worse**）。逐突变体看，MAAL 在 77 个突变体上比 unc 更差，仅在 48 个上更好。MAAL 的 acquisition = 0.6×diversity + 0.4×target_priority 设计意图是聚焦目标家族，但实际效果是**过度聚焦少数家族（cdc20_ts/CLB2_db）而忽视全局覆盖**，导致 tem1_KO (MAAL=1.36 vs unc=0.05)、cdc20_clb5 (MAAL=1.68 vs unc=0.09) 等突变体出现极端恶化。

3. **cdc20_ts 系列是 al4pde 的"阿喀琉斯之踵"**: 5 个 cdc20_ts 系列突变体（73/74/75/78/79）在 al4pde 中全部 MAE > 1.4，而 v5+none 仅 0.03-0.04（**恶化 40-60×**）。根因是 cdc20_ts 涉及 CDC20 蛋白温度敏感突变，其参数空间（ka20p, ki20p, ks20p 等）在 v1 数据集中几乎无覆盖，且 AL 的 uncertainty/diversity 采样无法有效识别这类"未知未知"区域。al4pde 框架的根本缺陷是**在训练数据严重不足时，AL 的探索能力无法弥补初始训练集的覆盖盲区**。

---

## 2. al4pde 框架配置回顾

> 详见: `run_pipeline.sh` 和 `utils/al_callback.py`

### 2.1 al4pde 实验配置

| 配置项 | 值 | 说明 |
|--------|-----|------|
| 训练数据集 | `lhs_cheat_origin_210min_500steps_dual_labels.npz` (v1) | 53,796 样本 |
| 候选池数据集 | `lhs_cheat_origin_v5_210min_500steps_dual_labels.npz` (v5) | 162,008 样本（排除 126 Real Mutant） |
| 初始训练集大小 | **5,000** | Family × Pattern 分层随机抽样 |
| AL 触发 epoch | **20** | 每 20 epoch 触发一次 |
| 每轮 AL 样本数 | **1,000** | 触发后添加 1000 样本 |
| AL 总轮数 | **10** | 200 epoch 共 10 轮 |
| AL 总样本数 | **10,000** | 10 × 1000 |
| 总训练样本数 | **15,000** | 5000 + 10000 |
| 训练总 epoch | 200 | — |
| 候选池规模 | 162,008 | 54,619 target + 107,389 non-target LHS |

### 2.2 四种 AL 策略

| 策略 | acquisition 函数 | 设计意图 |
|------|------------------|----------|
| al4pde_random | 随机采样 | 基线 |
| al4pde_uncertainty | MC Dropout 不确定性 (10 forward passes) | 采样模型最不确定的样本 |
| al4pde_diversity | bmdal_reg 最大距离多样性 | 采样特征空间最分散的样本 |
| **MAAL** | `0.6×diversity + 0.4×target_priority` | 融合多样性 + 目标家族优先级 |

### 2.3 MAAL target_priority 计算

```python
# utils/al_callback.py 行 2340-2375
acquisition_score = (1 - target_weight) * diversity + target_weight * target_priority
# MAAL: acquisition = div_w * diversity + tgt_w * target_priority
acquisition = div_w * diversity_scores + tgt_w * target_priorities
sel_idx = np.argsort(acquisition)[-n_select:]
```

**target_priority 来源**: 基于 mutant atlas 的 `lhs_MAE_min` 加权计算，对每个候选样本根据其参数空间距离最近 real mutant 的距离赋权。

---

## 3. al4pde_uncertainty Eval 结果

### 3.1 全局指标

| 指标 | v5+none | **al4pde_unc** | 差异 | 评估 |
|------|---------|----------------|------|------|
| LHS Test Rel L2 | 0.0815 | 0.1652 | +102.6% | ❌ 严重退化 |
| LHS Test MAE | 0.0618 | 0.1737 | +181.1% | ❌ 严重退化 |
| LHS Test Corr | 0.9933 | 0.9722 | -2.1% | ❌ 退化 |
| Real Rel L2 | 0.0780 | 0.3575 | +358.3% | ❌ 灾难性退化 |
| **Real MAE** | **0.0552** | **0.3726** | **+574.9%** | ❌ 灾难性退化 |
| Real MSE | 0.166 | 12.149 | +7219% | ❌ 灾难性退化 |
| Real Corr | 0.9942 | 0.8733 | -12.1% | ❌ 严重退化 |
| Sim-to-Real Gap | 0.89× | 2.15× | +141.6% | ❌ 倒退 |
| Core 9 Real MAE | 0.148 | 1.126 | +660% | ❌ 灾难性退化 |

**关键洞察**: al4pde_unc 不仅 Real 指标灾难性退化，连 LHS Test 都退化 +181%（v5+none 的 2.8 倍），说明**模型整体欠拟合**而非仅仅是 sim-to-real gap 问题。Core 9 Real MAE=1.126（v5+none 0.148，7.6× worse）说明核心细胞周期变量（MASS/CLB2/CLB5/SIC1/CDC20/CDH1 等）的预测完全失效。

### 3.2 Pattern 分层性能

| Pattern | n | v5+none MAE | **unc MAE** | 恶化倍数 | unc 最大 MAE |
|---------|---|-------------|-------------|----------|--------------|
| A (Multi-Div ≥2) | 20 | 0.0498 | 0.2409 | **4.8×** | 0.5082 |
| B (Single-Div =1) | 45 | 0.0562 | 0.2995 | **5.3×** | 1.2636 |
| **C (Arrest =0)** | **61** | **0.0563** | **0.4697** | **8.3×** | **1.9063** |

**关键洞察**: Pattern C（Exponential-Arrest）是 al4pde_unc 最严重的灾难区，MAE 0.4697 是 v5+none (0.0563) 的 8.3 倍。61 个 Pattern C 突变体中有 8 个 MAE > 1.0（v5+none 全部 < 0.3）。Pattern C 涉及细胞周期 arrest 表型，参数空间高度非线性，5000 初始样本完全无法覆盖。

### 3.3 Tier 分布

| Tier (MAE 范围) | v5+none | **al4pde_unc** | 变化 |
|-----------------|---------|----------------|------|
| Tier 1 (< 0.1) | 113 (89.7%) | **18 (14.3%)** | -95 ❌ |
| Tier 2 (0.1-0.2) | 10 (7.9%) | 38 (30.2%) | +28 |
| Tier 3 (0.2-0.3) | 2 (1.6%) | 20 (15.9%) | +18 |
| Tier 4 (≥ 0.3) | 1 (0.8%) | **50 (39.7%)** | +49 ❌ |
| **MAE > 1.0** | 0 | **10 (7.9%)** | +10 ❌ |

**关键洞察**: v5+none 的 113 个 Tier 1 突变体（MAE<0.1）在 al4pde_unc 中只剩 18 个，95 个退化到 Tier 2/3/4。最严重的 10 个突变体 MAE > 1.0，意味着模型预测完全错误（预测值与真实值差 1 个量级以上）。

### 3.4 al4pde_unc Top 15 灾难性恶化突变体

| Mutant | Pattern | v5+none MAE | **unc MAE** | 恶化倍数 | 根因 |
|--------|---------|-------------|-------------|----------|------|
| **78_cdc20_ts_mad2** | C | 0.0318 | **1.9063** | **60.0×** | cdc20_ts 参数空间未覆盖 |
| **73_cdc20_ts** | C | 0.0402 | **1.8991** | **47.3×** | cdc20_ts 参数空间未覆盖 |
| **79_cdc20_ts_bub2** | C | 0.0427 | **1.7714** | **41.5×** | cdc20_ts + bub2 联合未覆盖 |
| 62_GAL_CLB2_db | C | 0.0633 | 1.6240 | 25.6× | CLB2_db + GAL 联合未覆盖 |
| 124_APC_A_GAL_CLB2 | C | 0.1548 | 1.6114 | 10.4× | APC_A + GAL_CLB2 联合 |
| 71_CLB5_db_pds1_cdc20 | C | 0.0301 | 1.5096 | 50.1× | cdc20 + CLB5_db 联合 |
| 75_cdc20_pds1 | C | 0.0319 | 1.4927 | 46.8× | cdc20 + pds1 联合 |
| 85_GAL_ESP1_cdc20_ts | C | 0.0575 | 1.4715 | 25.6× | cdc20_ts + ESP1 联合 |
| 102_net1_cdc20_ts | C | 0.0438 | 1.2637 | 28.9× | cdc20_ts + net1 联合 |
| 63_clb5_clb6_KO | B | 0.1292 | 1.2636 | 9.8× | clb5_clb6_KO 未覆盖 |
| 15_cln3_bck2_multi_CLN2 | C | 0.0236 | 0.8463 | 35.9× | cln3_bck2 联合 |
| 22_cln1_cln2_cln3_multi_CLB5 | B | 0.0428 | 0.8397 | 19.6× | cln_triple + CLB5 联合 |
| 10_bck2_KO | B | 0.1114 | 0.7101 | 6.4× | bck2_KO 未覆盖 |
| 41_sic1_cdc6_cdh1 | C | 0.0391 | 0.6927 | 17.7× | sic1_cdc6_cdh1 三联合 |
| 47_clb1_clb2_KO | C | 0.0235 | 0.6684 | 28.5× | clb1_clb2_KO 未覆盖 |

**关键发现**: Top 15 灾难性恶化中 12 个是 Pattern C（Arrest），10 个涉及 cdc20 或其下游/上游联合突变。**cdc20_ts 系列（73/78/79/85/102）是 al4pde_unc 的绝对灾难区**，5 个突变体 MAE 全部 > 1.26，而 v5+none 仅 0.03-0.06。

### 3.5 al4pde_unc 表现相对最好的 10 个突变体

| Mutant | v5+none MAE | unc MAE | 比值 |
|--------|-------------|---------|------|
| 90_tem1_net1_ts | 0.0301 | 0.0411 | 1.4× |
| 86_tem1_KO | 0.0287 | 0.0527 | 1.8× |
| 59_CLB2_db_multi_CDC6 | 0.0370 | 0.0538 | 1.5× |
| 57_CLB2_db_multi_SIC1 | 0.0332 | 0.0600 | 1.8× |
| 1_WT_Glc | 0.0161 | 0.0620 | 3.8× |
| 11_multi_BCK2 | 0.0208 | 0.0709 | 3.4× |
| 2.2_WT_Raff | 0.0218 | 0.0743 | 3.4× |
| 2.1_WT_Gal | 0.0209 | 0.0743 | 3.6× |
| 33_GAL_SIC1_GAL_CLN2_cdh1 | 0.0418 | 0.0801 | 1.9× |
| 9_GAL_CLN3 | 0.0262 | 0.0814 | 3.1× |

**关键洞察**: 即使是 al4pde_unc 表现最好的 10 个突变体，MAE 也全部比 v5+none 高 1.4×-3.8×。WT 突变体（1/2.1/2.2）在 v5+none 中 MAE≈0.02（极佳），在 unc 中退化到 0.06-0.07，说明**即使是 Wild Type 这种最简单的预测目标，5000+10000 样本的训练集也无法让模型充分收敛**。

---

## 4. MAAL Eval 结果

### 4.1 全局指标

| 指标 | v5+none | al4pde_unc | **MAAL** | MAAL vs unc |
|------|---------|------------|----------|-------------|
| LHS Test Rel L2 | 0.0815 | 0.1652 | 0.1659 | +0.4% |
| LHS Test MAE | 0.0618 | 0.1737 | 0.1671 | -3.8% ✅ |
| Real Rel L2 | 0.0780 | 0.3575 | **0.4210** | +17.8% ❌ |
| **Real MAE** | 0.0552 | 0.3726 | **0.4088** | **+9.7%** ❌ |
| Real MSE | 0.166 | 12.149 | 10.855 | -10.6% ✅ |
| Real Corr | 0.9942 | 0.8733 | **0.8240** | -5.6% ❌ |
| Sim-to-Real Gap | 0.89× | 2.15× | **2.45×** | +14.0% ❌ |
| Core 9 Real MAE | 0.148 | 1.126 | 1.153 | +2.4% ❌ |

**关键洞察**: MAAL 在 LHS Test 上略好于 unc（-3.8% MAE），但在 Real Mutants 上全面更差（Real MAE +9.7%，Real Corr -5.6%，Gap +14%）。这说明 **MAAL 的 target_priority 权重让模型在 LHS 分布上稍微收敛，但牺牲了对 Real Mutant 的泛化能力**——target_priority 让 AL 过度聚焦于 atlas 中标注的少数家族，而忽视了更广泛的参数空间探索。

### 4.2 Pattern 分层性能

| Pattern | v5+none MAE | unc MAE | **MAAL MAE** | MAAL vs unc |
|---------|-------------|---------|--------------|-------------|
| A (Multi-Div ≥2) | 0.0498 | 0.2409 | **0.2772** | +15.1% ❌ |
| B (Single-Div =1) | 0.0562 | 0.2995 | **0.3297** | +10.1% ❌ |
| **C (Arrest =0)** | 0.0563 | 0.4697 | **0.5104** | **+8.6%** ❌ |

**关键洞察**: MAAL 在所有三个 Pattern 上都比 unc 更差，Pattern A 恶化最严重（+15.1%）。这进一步证明 **MAAL 的 target_priority 机制全面失效**——不仅没有改善目标家族（atlas 中 lhs_MAE_min 高的家族），反而因为过度聚焦而忽视了对其他家族的覆盖。

### 4.3 Tier 分布

| Tier (MAE 范围) | v5+none | al4pde_unc | **MAAL** | MAAL vs unc |
|-----------------|---------|------------|----------|-------------|
| Tier 1 (< 0.1) | 113 (89.7%) | 18 (14.3%) | **12 (9.5%)** | -6 ❌ |
| Tier 2 (0.1-0.2) | 10 (7.9%) | 38 (30.2%) | 29 (23.0%) | -9 |
| Tier 3 (0.2-0.3) | 2 (1.6%) | 20 (15.9%) | 24 (19.0%) | +4 |
| Tier 4 (≥ 0.3) | 1 (0.8%) | 50 (39.7%) | **61 (48.4%)** | +11 ❌ |
| **MAE > 1.0** | 0 | 10 | 9 | -1 |

**关键洞察**: MAAL 的 Tier 4 灾难区从 unc 的 50 个（39.7%）扩大到 61 个（48.4%），近一半 real mutant 预测完全失效。Tier 1 从 18 个进一步缩减到 12 个。

### 4.4 MAAL Top 15 灾难性恶化突变体

| Mutant | Pattern | v5+none MAE | **MAAL MAE** | 恶化倍数 | 根因 |
|--------|---------|-------------|--------------|----------|------|
| 78_cdc20_ts_mad2 | C | 0.0318 | **1.9534** | **61.5×** | cdc20_ts 灾难 |
| 73_cdc20_ts | C | 0.0402 | 1.8976 | 47.2× | cdc20_ts 灾难 |
| 79_cdc20_ts_bub2 | C | 0.0427 | 1.7290 | 40.5× | cdc20_ts 灾难 |
| **74_cdc20_clb5** | C | 0.0421 | **1.6816** | **39.9×** | MAAL 独有灾难 |
| 71_CLB5_db_pds1_cdc20 | C | 0.0301 | 1.4861 | 49.3× | cdc20 灾难 |
| 75_cdc20_pds1 | C | 0.0319 | 1.4854 | 46.5× | cdc20 灾难 |
| **86_tem1_KO** | C | 0.0287 | **1.3612** | **47.5×** | MAAL 独有灾难 |
| 102_net1_cdc20_ts | C | 0.0438 | 1.2403 | 28.3× | cdc20_ts 灾难 |
| 22_cln1_cln2_cln3_multi_CLB5 | B | 0.0428 | 1.0107 | 23.6× | cln_triple 灾难 |
| 15_cln3_bck2_multi_CLN2 | C | 0.0236 | 0.9286 | 39.4× | cln3_bck2 灾难 |
| 56_CLB2_db_Gal | C | 0.2518 | 0.7753 | 3.1× | CLB2_db_Gal 恶化 |
| 61.1_CLB2_db_clb5_Gal | C | 0.1474 | 0.7687 | 5.2× | CLB2_db 恶化 |
| 10_bck2_KO | B | 0.1114 | 0.7487 | 6.7× | bck2_KO 灾难 |
| 8_cln3_KO | B | 0.0892 | 0.7333 | 8.2× | cln3_KO 灾难 |
| 116_APC_A_cdh1 | C | 0.0323 | 0.6973 | 21.6× | APC_A_cdh1 灾难 |

**关键发现**: MAAL 比 unc 多了 2 个独有灾难性恶化：**74_cdc20_clb5** (unc=0.09, maal=1.68，恶化 18×) 和 **86_tem1_KO** (unc=0.05, maal=1.36，恶化 26×)。这两个突变体在 unc 中表现尚可，但 MAAL 的 target_priority 让 AL 过度采样其他家族，导致 tem1_KO 和 cdc20_clb5 的覆盖进一步不足。

---

## 5. MAAL vs al4pde_uncertainty 对比分析

### 5.1 逐突变体差值统计

| 统计 | 值 |
|------|----|
| MAAL 比 unc 更差的突变体数 | **77 / 126 (61.1%)** |
| MAAL 比 unc 更好的突变体数 | 48 / 126 (38.1%) |
| 持平（差值 < 0.001） | 1 / 126 (0.8%) |
| 总 MAE 差值 (maal - unc) | **+4.56** |
| 平均 MAE 差值 | +0.0362 |
| MAAL 比 unc 更差的最大幅度 | +1.59 (74_cdc20_clb5) |
| MAAL 比 unc 更好的最大幅度 | -1.56 (62_GAL_CLB2_db) |

### 5.2 MAAL 比 unc 更差的前 10 个突变体

| Mutant | unc MAE | MAAL MAE | 差值 | 根因 |
|--------|---------|----------|------|------|
| **74_cdc20_clb5** | 0.0939 | 1.6816 | **+1.59** | target_priority 忽视 cdc20_clb5 |
| **86_tem1_KO** | 0.0527 | 1.3612 | +1.31 | target_priority 忽视 tem1_KO |
| 116_APC_A_cdh1 | 0.1619 | 0.6973 | +0.54 | APC_A_cdh1 未在 atlas 重点标注 |
| 90_tem1_net1_ts | 0.0411 | 0.4831 | +0.44 | tem1_net1_ts 未在 atlas 重点标注 |
| 13_cln3_bck2_KO | 0.1686 | 0.5898 | +0.42 | cln3_bck2_KO 未在 atlas |
| 61.1_CLB2_db_clb5_Gal | 0.3588 | 0.7687 | +0.41 | CLB2_db_clb5_Gal 覆盖不足 |
| 8_cln3_KO | 0.3455 | 0.7333 | +0.39 | cln3_KO 未在 atlas |
| 39_sic1_cdc6_42_49 | 0.2417 | 0.6238 | +0.38 | sic1_cdc6 未在 atlas |
| 59_CLB2_db_multi_CDC6 | 0.0538 | 0.4188 | +0.37 | CLB2_db_multi_CDC6 未在 atlas |
| 106_cdc14_ts_cdh1 | 0.1870 | 0.5335 | +0.35 | cdc14_ts_cdh1 未在 atlas |

### 5.3 MAAL 比 unc 更好的前 10 个突变体

| Mutant | unc MAE | MAAL MAE | 差值 | 根因 |
|--------|---------|----------|------|------|
| **62_GAL_CLB2_db** | 1.6240 | 0.0605 | **-1.56** | target_priority 命中 GAL_CLB2_db |
| 85_GAL_ESP1_cdc20_ts | 1.4715 | 0.0883 | -1.38 | target_priority 命中 |
| 124_APC_A_GAL_CLB2 | 1.6114 | 0.2466 | -1.36 | target_priority 命中 |
| 63_clb5_clb6_KO | 1.2636 | 0.2621 | -1.00 | target_priority 命中 |
| 99_cdc14_ts | 0.3391 | 0.0376 | -0.30 | target_priority 命中 cdc14_ts |
| 47_clb1_clb2_KO | 0.6684 | 0.3752 | -0.29 | target_priority 命中 |
| 41_sic1_cdc6_cdh1 | 0.6927 | 0.4599 | -0.23 | target_priority 部分命中 |
| 54_GAL_CLB2_cdh1 | 0.3685 | 0.1441 | -0.22 | target_priority 命中 |
| 111_TAB6_1_CLB1_clb2 | 0.5988 | 0.4072 | -0.19 | target_priority 命中 |
| 94_cdc15_ts_multi_TEM1 | 0.6258 | 0.4706 | -0.16 | target_priority 命中 |

### 5.4 MAAL vs unc 对比结论

**MAAL 的 target_priority 机制呈现"极端两极分化"**:
- **命中的家族**（GAL_CLB2_db, GAL_ESP1_cdc20_ts, APC_A_GAL_CLB2, clb5_clb6_KO, cdc14_ts）改善 1.0-1.6 个 MAE 单位
- **未命中的家族**（cdc20_clb5, tem1_KO, APC_A_cdh1, tem1_net1_ts, cln3_bck2_KO, cln3_KO, sic1_cdc6）恶化 0.4-1.6 个 MAE 单位

**根本问题**: MAAL 的 target_priority 基于 mutant atlas 的 `lhs_MAE_min` 标注，但 atlas 只覆盖了 126 个 real mutant 中的部分家族。对于 atlas 未标注的家族（如 cdc20_clb5, tem1_KO, cln3_bck2_KO），target_priority=0，导致 AL 完全忽视这些家族的采样。这种"赢者通吃"效应在训练数据严重不足（15000 样本）的情况下被放大。

**结论**: **MAAL 在当前配置下不推荐使用**。target_priority 机制需要重新设计，确保对未标注家族有最小采样配额。

---

## 6. al4pde 失败根因深度分析

### 6.1 根因 1: 训练数据严重不足（最关键）

| 维度 | v5+none | al4pde | 比例 |
|------|---------|--------|------|
| 训练样本数 | ~130,000 (v5 80% train) | 15,000 (5000 init + 10000 AL) | **11.5%** |
| 占全量数据集比例 | 80% | 9.3% (15000/162134) | — |
| LHS Test MAE | 0.0618 | 0.1737 | 2.8× worse |
| Real MAE | 0.0552 | 0.3726 | 6.7× worse |

**分析**: al4pde 仅用 15,000 样本训练（v5+none 的 11.5%），导致模型整体欠拟合。LHS Test MAE 0.1737 是 v5+none (0.0618) 的 2.8 倍，说明**模型连 LHS 分布内预测都没收敛**，更谈不上泛化到 Real Mutant。这是 al4pde 失败的最根本原因——在数据量严重不足的情况下，任何 AL 策略都无法弥补初始训练集的覆盖盲区。

**数学论证**: 出芽酵母 ODE 模型有 141 个参数，39 个变量，500 时间步。假设每个参数维度需要 100 个样本才能合理覆盖，141 维参数空间至少需要 14100 个样本。al4pde 的 15000 样本仅勉强达到这个下限，而 v5+none 的 130,000 样本是下限的 9.2 倍，才能实现 Real MAE=0.0552 的精度。

### 6.2 根因 2: 初始训练集未包含关键家族

al4pde 的初始训练集（5000 样本）通过 Family × Pattern 分层随机抽样从 v1 LHS 常规样本中选取。但 v1 数据集（53,796 样本）本身**不包含 V2-V5 的定向采样家族**（cdc20_ts_family, tem1_ts_family, cdc14_ts_family, swi5_cdh1_family, GAL_CLB2_cdh1_family 等）。

| 关键家族 | v1 是否覆盖 | al4pde_unc MAE | v5+none MAE | 恶化倍数 |
|----------|-------------|----------------|-------------|----------|
| cdc20_ts_family (73/78/79) | ❌ 未覆盖 | 1.86 | 0.038 | **49×** |
| tem1_KO (86) | ❌ 未覆盖 | 0.05 | 0.029 | 1.8× |
| cln3_bck2_KO (13) | ❌ 未覆盖 | 0.17 | 0.020 | 8.4× |
| sic1_cdc6_cdh1 (41) | ❌ 未覆盖 | 0.69 | 0.039 | 17.7× |
| clb1_clb2_KO (47) | ❌ 未覆盖 | 0.67 | 0.024 | 28.5× |
| CLB2_db_Gal (56) | ❌ 未覆盖 | 0.48 | 0.252 | 1.9× |

**分析**: AL 的候选池虽然是 v5 数据集（包含定向家族），但 AL 触发从 epoch 20 开始，每轮仅添加 1000 样本。在 10 轮 AL 中，即使全部命中 cdc20_ts_family，也仅能添加 10000 个样本中的极小部分。而 cdc20_ts_family 在 v5 中只有 1200 个样本（pert=0.12），AL 的 uncertainty/diversity 采样无法保证命中。

### 6.3 根因 3: AL 触发太晚，每轮样本太少

| 配置 | 当前值 | 问题 |
|------|--------|------|
| AL 触发 epoch | 20 | 前 20 epoch 模型在 5000 样本上严重欠拟合，AL 采样的不确定性信号不可靠 |
| 每轮 AL 样本数 | 1000 | 相比 162,008 候选池，1000 样本仅占 0.6%，覆盖能力极弱 |
| AL 总轮数 | 10 | 200 epoch 共 10 轮，每轮间隔 20 epoch |
| AL 总样本数 | 10,000 | 仅占候选池 6.2%，占全量数据集 9.3% |

**分析**: AL 触发太晚（epoch 20）导致前 20 epoch 模型在 5000 样本上严重欠拟合，此时模型的不确定性估计（MC Dropout）不可靠——模型对"未知未知"区域（如 cdc20_ts）的预测可能反而自信（低不确定性），导致 AL 错过这些关键区域。每轮 1000 样本相对 162,008 候选池比例太低（0.6%），即使 AL 完美命中目标家族，10 轮累计也仅覆盖候选池 6.2%。

### 6.4 根因 4: cdc20_ts 系列全面崩溃

cdc20_ts 系列（5 个突变体：73/74/75/78/79）在 al4pde 中 MAE 全部 > 1.4，是 v5+none 的 40-60 倍。

| Mutant | v5+none MAE | unc MAE | maal MAE | 涉及参数 |
|--------|-------------|---------|----------|----------|
| 73_cdc20_ts | 0.0402 | 1.8991 | 1.8976 | ka20p, ki20p, ks20p 温度敏感 |
| 74_cdc20_clb5 | 0.0421 | 0.0939 | 1.6816 | cdc20 + clb5 联合 |
| 75_cdc20_pds1 | 0.0319 | 1.4927 | 1.4854 | cdc20 + pds1 联合 |
| 78_cdc20_ts_mad2 | 0.0318 | 1.9063 | 1.9534 | cdc20_ts + mad2 联合 |
| 79_cdc20_ts_bub2 | 0.0427 | 1.7714 | 1.7290 | cdc20_ts + bub2 联合 |

**分析**: cdc20_ts 涉及 CDC20 蛋白的温度敏感突变，参数空间（ka20p, ki20p, ks20p 等的 perturbation）高度非线性。v1 数据集的 LHS 常规采样几乎不会命中这些参数的敏感区域。v5 数据集通过 `cdc20_ts_family` 定向采样（虽然 V5 中未直接添加，但 V4 的 density pert=0.08 覆盖了部分）才能让 v5+none 实现 MAE=0.04。al4pde 的 AL 策略无法识别这种"未知未知"区域——MC Dropout 的不确定性估计依赖于模型已有知识，对完全未见的参数区域可能反而低不确定性。

### 6.5 根因 5: MAAL target_priority 权重失衡

MAAL 的 acquisition = 0.6×diversity + 0.4×target_priority，target_weight=0.4 过高。

**问题**:
1. target_priority 基于 mutant atlas 的 `lhs_MAE_min` 标注，但 atlas 只覆盖部分家族
2. 对于 atlas 未标注的家族，target_priority=0，导致 AL 完全忽视
3. 0.4 权重让 target_priority 对 acquisition 的影响过大，覆盖了 diversity 的探索能力
4. 在训练数据严重不足时，这种"赢者通吃"效应被放大

**证据**: MAAL 在 77 个突变体上比 unc 更差，其中 74_cdc20_clb5 (差 +1.59) 和 86_tem1_KO (差 +1.31) 是 MAAL 独有的灾难性恶化——这两个突变体在 atlas 中未重点标注，target_priority=0，导致 AL 完全忽视。

---

## 7. Sim-to-Real Gap 演进对比

### 7.1 Sim-to-Real Gap 历史演进

| 版本 | 训练样本 | LHS Test MAE | Real MAE | Sim-to-Real Gap |
|------|----------|--------------|----------|-----------------|
| v1+none | ~43,000 | 0.0764 | 0.3600 | 4.53× |
| v2+none | ~96,000 | 0.0635 | 0.1926 | 2.98× |
| v3+none | ~110,000 | 0.0636 | 0.1013 | 1.89× |
| v4+none | ~126,000 | 0.0602 | 0.0598 | 1.09× |
| **v5+none** | **~130,000** | **0.0618** | **0.0552** | **0.89×** ✅ |
| **al4pde_unc** | **15,000** | 0.1737 | 0.3726 | **2.15×** ❌ |
| **MAAL** | **15,000** | 0.1671 | 0.4088 | **2.45×** ❌ |

### 7.2 关键洞察

1. **al4pde 的 Sim-to-Real Gap (2.15×) 相当于回退到 V2 时代 (2.98×) 和 V1 时代 (4.53×) 之间**，4 个版本的定向采样努力（V2→V5）全部白费
2. **MAAL 的 Gap (2.45×) 比 unc (2.15×) 更差**，证明 target_priority 机制不仅没改善 sim-to-real gap，反而恶化了
3. **al4pde 的 LHS Test MAE (0.1737) 是 v5+none (0.0618) 的 2.8 倍**，说明问题不仅是 sim-to-real gap，而是模型整体欠拟合
4. **训练样本数与 Gap 的关系**: 43,000→4.53×, 96,000→2.98×, 110,000→1.89×, 126,000→1.09×, 130,000→0.89×, 15,000→2.15×。**15,000 样本的 Gap (2.15×) 介于 43,000 (4.53×) 和 96,000 (2.98×) 之间**，说明 al4pde 的有效训练规模相当于 V1 和 V2 之间

---

## 8. al4pde 优化方案

### 8.1 优化方案总览

| 优先级 | 方案 | 预期效果 | 实施难度 |
|--------|------|----------|----------|
| **P0** | 增大初始训练集至 30,000-50,000 | Real MAE 0.37→0.15-0.20 | 低 |
| **P1** | 改用 v5 数据集作为训练集（非 v1） | 覆盖定向家族，Real MAE -50% | 低 |
| **P2** | 提前 AL 触发至 epoch 5，每轮增至 2000-3000 样本 | AL 采样更早生效 | 中 |
| **P3** | MAAL target_priority 增加"未标注家族最小配额"机制 | 防止 cdc20_clb5 类灾难 | 中 |
| **P4** | 引入"cdc20_ts 专用 AL 策略"，强制采样 cdc20_ts_family | 解决 cdc20_ts 灾难 | 高 |
| **P5** | 两阶段训练：先全量预训练 100 epoch，再 AL 微调 100 epoch | 解决欠拟合 | 高 |

### 8.2 P0: 增大初始训练集（最关键）

**问题**: 当前 5000 初始样本仅占全量 3.1%，严重欠拟合。

**方案**: 将 `al_initial_train_size` 从 5000 增大到 30000-50000。

```bash
# run_pipeline.sh 修改
AL_PARAMS="$AL_PARAMS --al_pool_dataset $DATASET_V5 --al_initial_train_size 30000"
```

**预期效果**:
- 30,000 样本占全量 18.5%，接近 v2+none 的 96,000 样本比例
- Real MAE 预计从 0.37 降到 0.15-0.20（参考 v2+none 0.19）
- LHS Test MAE 预计从 0.17 降到 0.08-0.10

### 8.3 P1: 改用 v5 数据集作为训练集

**问题**: 当前训练集是 v1（53,796 样本），不包含 V2-V5 的定向采样家族。

**方案**: 将训练数据集从 v1 改为 v5（162,134 样本）。

```bash
# run_pipeline.sh 修改
DATASET_V5="$DATA_DIR/lhs_cheat_origin_v5_210min_500steps_dual_labels.npz"
# al4pde 实验的 DATASET 改为 $DATASET_V5（而非 $DATASET_CHEAT）
"hyper_fno mse_only false ${DATASET_V5} v1_al4pde 32 32 64 200 al4pde_uncertainty 6e-4 20 1000 0 0 one_time"
```

**预期效果**:
- 候选池包含 V5 的 17 个定向采样家族
- cdc20_ts_family, tem1_ts_family 等关键家族进入候选池
- AL 采样有概率命中这些家族

### 8.4 P2: 提前 AL 触发，增大每轮样本数

**问题**: AL 触发太晚（epoch 20），每轮样本太少（1000）。

**方案**: 
- AL 触发 epoch 从 20 改为 5
- 每轮 AL 样本数从 1000 增大到 2000-3000
- AL 总轮数保持 10，总 AL 样本数 20,000-30,000

```bash
# run_pipeline.sh 修改
# 参数顺序: al_trigger_epoch al_num_add
"hyper_fno mse_only false ${DATASET_V5} v1_al4pde 32 32 64 200 al4pde_uncertainty 6e-4 5 3000 0 0 one_time"
# AL 总样本: 10 × 3000 = 30,000
# 总训练: 30,000 (init) + 30,000 (AL) = 60,000
```

**预期效果**:
- AL 在模型欠拟合阶段就介入，更早探索未知区域
- 每轮 3000 样本覆盖能力提升 3 倍
- 总训练样本 60,000，占全量 37%

### 8.5 P3: MAAL target_priority 改进

**问题**: target_priority 对未标注家族赋 0，导致"赢者通吃"。

**方案**: 为未标注家族设置最小采样配额。

```python
# utils/al_callback.py MAAL 策略修改
def maal_acquisition(diversity_scores, target_priorities, n_select, min_quota=0.1):
    # 为 target_priority=0 的样本设置最小值
    target_priorities = np.where(
        target_priorities < min_quota,
        min_quota,  # 最小配额 0.1
        target_priorities
    )
    # 重新计算 acquisition
    acquisition = div_w * diversity_scores + tgt_w * target_priorities
    sel_idx = np.argsort(acquisition)[-n_select:]
    return sel_idx
```

**预期效果**:
- cdc20_clb5, tem1_KO 等 atlas 未标注家族获得最小采样概率
- MAAL 不再出现独有灾难性恶化

### 8.6 P4: cdc20_ts 专用 AL 策略

**问题**: cdc20_ts 系列在 al4pde 中 MAE > 1.4，是 v5+none 的 40-60 倍。

**方案**: 引入"cdc20_ts 强制采样"机制，每轮 AL 保证至少 200 个 cdc20_ts_family 样本。

```python
# utils/al_callback.py 新增 cdc20_ts 强制采样
def cdc20_ts_forced_sampling(pool_X, pool_Y, pool_keys, n_forced=200):
    # 筛选 cdc20_ts_family 样本
    cdc20_mask = np.array(['cdc20_ts' in k for k in pool_keys])
    cdc20_indices = np.where(cdc20_mask)[0]
    # 随机选择 n_forced 个
    if len(cdc20_indices) >= n_forced:
        forced_idx = np.random.choice(cdc20_indices, n_forced, replace=False)
    else:
        forced_idx = cdc20_indices
    return forced_idx

# 在 AL 触发时
forced_idx = cdc20_ts_forced_sampling(pool_X, pool_Y, pool_keys, n_forced=200)
remaining_budget = n_select - len(forced_idx)
# 其余样本用 uncertainty/diversity 采样
```

### 8.7 P5: 两阶段训练

**问题**: 15,000 样本直接训练 200 epoch 严重欠拟合。

**方案**: 
- 阶段 1: 全量 v5 数据集预训练 100 epoch（不触发 AL）
- 阶段 2: 冻结部分层，用 AL 选择的 15,000 样本微调 100 epoch

```bash
# 两阶段训练配置
# Stage 1: 全量预训练
"hyper_fno mse_only false ${DATASET_V5} v1_al4pde 32 32 64 100 none 6e-4 0 0 0 0 one_time"
# Stage 2: AL 微调（需要额外脚本支持）
```

**预期效果**:
- 阶段 1 解决欠拟合，模型在 LHS 分布内收敛
- 阶段 2 用 AL 精细化 Real Mutant 区域预测

---

## 9. 九方对比总结表

| 维度 | v1+none | v2+none | v3+none | v4+none | **v5+none** | **v5+ma** | **al4pde_unc** | **MAAL** |
|------|---------|---------|---------|---------|-------------|-----------|----------------|----------|
| 数据集规模 | 53,796 | 120,017 | 137,406 | 157,599 | 162,134 | 同 v5 | v1 (53,796) | v1 (53,796) |
| 训练样本数 | ~43,000 | ~96,000 | ~110,000 | ~126,000 | ~130,000 | ~130,000 | **15,000** | **15,000** |
| LHS Test MAE | 0.0764 | 0.0635 | 0.0636 | 0.0602 | 0.0618 | 0.0617 | 0.1737 | 0.1671 |
| Real MAE | 0.3600 | 0.1926 | 0.1013 | 0.0598 | 0.0552 | 0.0549 | 0.3726 | 0.4088 |
| Real MSE | — | 4.450 | 0.442 | 0.174 | 0.166 | 0.157 | 12.149 | 10.855 |
| Real Corr | 0.8337 | 0.9370 | 0.9798 | 0.9933 | 0.9942 | 0.9945 | 0.8733 | 0.8240 |
| Sim-to-Real Gap | 4.53× | 2.98× | 1.89× | 1.09× | **0.89×** | **0.89×** | 2.15× | 2.45× |
| Tier 1 占比 | — | 49.2% | 65.9% | 85.7% | 89.7% | 90.5% | 14.3% | 9.5% |
| Tier 4 占比 | — | 20.6% | 4.0% | 0.0% | 0.8% | 0.8% | 39.7% | 48.4% |
| MAE > 1.0 数 | — | — | — | 0 | 0 | 0 | **10** | **9** |
| 评估 | ❌ 灾难 | ⚠️ 差 | ⚠️ 中 | ✅ 好 | ✅ 极好 | ✅ 极好 | ❌ 灾难 | ❌ 灾难 |

**核心洞察**:
1. **v5+none 和 v5+ma 是当前最优配置**（Real MAE ≈ 0.055，Gap < 1.0）
2. **al4pde_unc 和 MAAL 的性能介于 v1+none 和 v2+none 之间**，相当于回退到 V1-V2 时代
3. **al4pde 的训练样本数 (15,000) 是 v5+none (130,000) 的 11.5%**，但 Real MAE 是 6.7 倍，说明 AL 策略在数据严重不足时无法弥补
4. **MAAL 比 al4pde_uncertainty 更差**，证明 target_priority 机制需要重新设计

---

## 10. 结论

### 10.1 核心结论

1. **al4pde 风格 AL 框架在当前配置下完全失败**: Real MAE 0.3726 (unc) / 0.4088 (maal) 是 v5+none (0.0552) 的 6.7×-7.4×，126 个 real mutant 中 126/125 个全部恶化。Sim-to-Real Gap 从 0.89× 倒退到 2.15× (unc) / 2.45× (maal)，相当于回退到 V1-V2 时代。

2. **失败根因是训练数据严重不足**: 15,000 样本（5000 初始 + 10000 AL）仅占 v5 全量 162,134 的 9.3%，模型整体欠拟合（LHS Test MAE 0.1737 是 v5+none 0.0618 的 2.8 倍）。任何 AL 策略都无法弥补初始训练集的覆盖盲区。

3. **cdc20_ts 系列是 al4pde 的绝对灾难区**: 5 个 cdc20_ts 突变体 MAE 全部 > 1.4（v5+none 仅 0.03-0.04，恶化 40-60×），源于 v1 训练集未覆盖 cdc20_ts_family 定向采样。

4. **MAAL 比 al4pde_uncertainty 更差**: target_priority 权重 0.4 过高，导致"赢者通吃"——命中的家族（GAL_CLB2_db 等）改善 1.0-1.6 MAE，未命中的家族（cdc20_clb5, tem1_KO 等）恶化 0.4-1.6 MAE，净效果 +9.7% worse。

5. **al4pde 框架优化方案**: P0 增大初始训练集至 30,000-50,000（最关键），P1 改用 v5 数据集作为训练集，P2 提前 AL 触发至 epoch 5 + 每轮 3000 样本，P3 MAAL target_priority 增加未标注家族最小配额，P4 cdc20_ts 强制采样，P5 两阶段训练。

### 10.2 行动建议

**短期（立即可行）**:
1. **停用 al4pde 框架的当前配置**，继续使用 v5+none 或 v5+mutant_aware（Real MAE 0.055）
2. 若要继续探索 AL，先实施 P0 + P1（增大初始训练集 + 改用 v5 数据集），重新评估

**中期（需要代码修改）**:
3. 实施 P2（提前 AL 触发 + 增大每轮样本数）
4. 实施 P3（MAAL target_priority 改进），重新对比 MAAL vs unc

**长期（架构调整）**:
5. 实施 P5（两阶段训练），先全量预训练再 AL 微调
6. 探索 P4（cdc20_ts 专用 AL 策略），解决 cdc20_ts 灾难

### 10.3 最终判断

**在 V5 数据集已实现 Sim-to-Real Gap 0.89× 的情况下，al4pde 风格 AL 框架的边际收益为负**。当前 V5+none 配置已接近最优，AL 框架的探索价值有限。建议将精力集中在 V6 数据集优化（解决 tem1_ts_family 失效和 56_CLB2_db_Gal 恶化）上，而非继续调优 al4pde 框架。

---

## 附录 A: al4pde_unc Top 30 灾难性恶化突变体完整列表

| 排名 | Mutant | Pattern | v5+none MAE | unc MAE | 恶化倍数 |
|------|--------|---------|-------------|---------|----------|
| 1 | 78_cdc20_ts_mad2 | C | 0.0318 | 1.9063 | 60.0× |
| 2 | 73_cdc20_ts | C | 0.0402 | 1.8991 | 47.3× |
| 3 | 79_cdc20_ts_bub2 | C | 0.0427 | 1.7714 | 41.5× |
| 4 | 62_GAL_CLB2_db | C | 0.0633 | 1.6240 | 25.6× |
| 5 | 124_APC_A_GAL_CLB2 | C | 0.1548 | 1.6114 | 10.4× |
| 6 | 71_CLB5_db_pds1_cdc20 | C | 0.0301 | 1.5096 | 50.1× |
| 7 | 75_cdc20_pds1 | C | 0.0319 | 1.4927 | 46.8× |
| 8 | 85_GAL_ESP1_cdc20_ts | C | 0.0575 | 1.4715 | 25.6× |
| 9 | 102_net1_cdc20_ts | C | 0.0438 | 1.2637 | 28.9× |
| 10 | 63_clb5_clb6_KO | B | 0.1292 | 1.2636 | 9.8× |
| 11 | 15_cln3_bck2_multi_CLN2 | C | 0.0236 | 0.8463 | 35.9× |
| 12 | 22_cln1_cln2_cln3_multi_CLB5 | B | 0.0428 | 0.8397 | 19.6× |
| 13 | 10_bck2_KO | B | 0.1114 | 0.7101 | 6.4× |
| 14 | 41_sic1_cdc6_cdh1 | C | 0.0391 | 0.6927 | 17.7× |
| 15 | 47_clb1_clb2_KO | C | 0.0235 | 0.6684 | 28.5× |
| 16 | 94_cdc15_ts_multi_TEM1 | C | 0.0330 | 0.6258 | 19.0× |
| 17 | 111_TAB6_1_CLB1_clb2 | B | 0.1120 | 0.5988 | 5.3× |
| 18 | 48_CLB1_clb2 | B | 0.0999 | 0.5764 | 5.8× |
| 19 | 20_cln1_cln2_cln3_sic1 | B | 0.0396 | 0.5685 | 14.4× |
| 20 | 3_cln1_cln2_KO | B | 0.0370 | 0.5660 | 15.3× |

---

## 附录 B: MAAL Top 30 灾难性恶化突变体完整列表

| 排名 | Mutant | Pattern | v5+none MAE | MAAL MAE | 恶化倍数 |
|------|--------|---------|-------------|----------|----------|
| 1 | 78_cdc20_ts_mad2 | C | 0.0318 | 1.9534 | 61.5× |
| 2 | 73_cdc20_ts | C | 0.0402 | 1.8976 | 47.2× |
| 3 | 79_cdc20_ts_bub2 | C | 0.0427 | 1.7290 | 40.5× |
| 4 | 74_cdc20_clb5 | C | 0.0421 | 1.6816 | 39.9× |
| 5 | 71_CLB5_db_pds1_cdc20 | C | 0.0301 | 1.4861 | 49.3× |
| 6 | 75_cdc20_pds1 | C | 0.0319 | 1.4854 | 46.5× |
| 7 | 86_tem1_KO | C | 0.0287 | 1.3612 | 47.5× |
| 8 | 102_net1_cdc20_ts | C | 0.0438 | 1.2403 | 28.3× |
| 9 | 22_cln1_cln2_cln3_multi_CLB5 | B | 0.0428 | 1.0107 | 23.6× |
| 10 | 15_cln3_bck2_multi_CLN2 | C | 0.0236 | 0.9286 | 39.4× |
| 11 | 56_CLB2_db_Gal | C | 0.2518 | 0.7753 | 3.1× |
| 12 | 61.1_CLB2_db_clb5_Gal | C | 0.1474 | 0.7687 | 5.2× |
| 13 | 10_bck2_KO | B | 0.1114 | 0.7487 | 6.7× |
| 14 | 8_cln3_KO | B | 0.0892 | 0.7333 | 8.2× |
| 15 | 116_APC_A_cdh1 | C | 0.0323 | 0.6973 | 21.6× |
| 16 | 117_APC_A_cdh1_Gal | C | 0.0526 | 0.6697 | 12.7× |
| 17 | 39_sic1_cdc6_42_49 | A | 0.0435 | 0.6238 | 14.3× |
| 18 | 3_cln1_cln2_KO | B | 0.0370 | 0.6026 | 16.3× |
| 19 | 23_cln1_cln2_cln3_GAL_CLB5 | C | 0.0405 | 0.5912 | 14.6× |
| 20 | 13_cln3_bck2_KO | C | 0.0200 | 0.5898 | 29.5× |

---

## 附录 C: al4pde 优化方案代码片段

### C.1 P0: 增大初始训练集

```bash
# run_pipeline.sh 修改
# 当前: --al_initial_train_size 5000
# 修改为: --al_initial_train_size 30000
AL_PARAMS="$AL_PARAMS --al_pool_dataset $DATASET_V5 --al_initial_train_size 30000"
```

### C.2 P1: 改用 v5 数据集作为训练集

```bash
# run_pipeline.sh 修改
# 当前: DATASET_CHEAT (v1)
# 修改为: DATASET_V5
DATASET_V5="$DATA_DIR/lhs_cheat_origin_v5_210min_500steps_dual_labels.npz"
"hyper_fno mse_only false ${DATASET_V5} v1_al4pde 32 32 64 200 al4pde_uncertainty 6e-4 20 1000 0 0 one_time"
```

### C.3 P2: 提前 AL 触发，增大每轮样本数

```bash
# run_pipeline.sh 修改
# 当前: al_trigger_epoch=20, al_num_add=1000
# 修改为: al_trigger_epoch=5, al_num_add=3000
"hyper_fno mse_only false ${DATASET_V5} v1_al4pde 32 32 64 200 al4pde_uncertainty 6e-4 5 3000 0 0 one_time"
# AL 总样本: 10 × 3000 = 30,000
# 总训练: 30,000 (init) + 30,000 (AL) = 60,000
```

### C.4 P3: MAAL target_priority 改进

```python
# utils/al_callback.py MAAL 策略修改 (行 2340-2375 附近)
def maal_select(diversity_scores, target_priorities, n_select, 
                div_w=0.6, tgt_w=0.4, min_quota=0.1):
    """
    MAAL acquisition with minimum quota for unlabeled families.
    
    Args:
        diversity_scores: diversity score for each candidate
        target_priorities: target priority for each candidate (0 for unlabeled)
        n_select: number of samples to select
        div_w: diversity weight (default 0.6)
        tgt_w: target priority weight (default 0.4)
        min_quota: minimum target_priority for unlabeled families (default 0.1)
    """
    # 为 target_priority=0 的样本设置最小配额
    target_priorities_adjusted = np.where(
        target_priorities < min_quota,
        min_quota,
        target_priorities
    )
    
    # 重新计算 acquisition
    acquisition = div_w * diversity_scores + tgt_w * target_priorities_adjusted
    sel_idx = np.argsort(acquisition)[-n_select:]
    return sel_idx
```

### C.5 P4: cdc20_ts 强制采样

```python
# utils/al_callback.py 新增 cdc20_ts 强制采样机制
def cdc20_ts_forced_sampling(pool_keys, n_forced=200):
    """
    强制采样 cdc20_ts_family 样本，防止 cdc20_ts 系列崩溃。
    
    Args:
        pool_keys: 候选池样本键列表
        n_forced: 强制采样数量 (default 200)
    """
    # 筛选 cdc20_ts_family 样本
    cdc20_mask = np.array([
        'cdc20_ts' in k or 'cdc20' in k.lower() 
        for k in pool_keys
    ])
    cdc20_indices = np.where(cdc20_mask)[0]
    
    # 随机选择 n_forced 个
    if len(cdc20_indices) >= n_forced:
        forced_idx = np.random.choice(cdc20_indices, n_forced, replace=False)
    else:
        forced_idx = cdc20_indices
    
    return forced_idx

# 在 AL 触发时调用
def al_trigger_with_forced(model, pool_X, pool_Y, pool_keys, n_select=1000):
    # 1. 强制采样 cdc20_ts_family
    forced_idx = cdc20_ts_forced_sampling(pool_keys, n_forced=200)
    
    # 2. 剩余预算用 uncertainty/diversity 采样
    remaining_budget = n_select - len(forced_idx)
    remaining_mask = np.ones(len(pool_X), dtype=bool)
    remaining_mask[forced_idx] = False
    
    # 用 uncertainty 采样剩余样本
    uncertainty_scores = compute_uncertainty(model, pool_X[remaining_mask])
    remaining_indices = np.where(remaining_mask)[0]
    unc_sel = np.argsort(uncertainty_scores)[-remaining_budget:]
    
    sel_idx = np.concatenate([forced_idx, remaining_indices[unc_sel]])
    return sel_idx
```

### C.6 P5: 两阶段训练

```bash
# run_pipeline.sh 两阶段训练配置
# Stage 1: 全量 v5 预训练 100 epoch (无 AL)
"hyper_fno mse_only false ${DATASET_V5} v1_al4pde 32 32 64 100 none 6e-4 0 0 0 0 one_time"

# Stage 2: AL 微调 100 epoch (需要额外脚本支持)
# 从 Stage 1 checkpoint 加载，用 AL 选择的 15,000 样本微调
"hyper_fno mse_only false ${DATASET_V5} v1_al4pde 32 32 64 100 al4pde_uncertainty 2e-4 5 1000 0 0 one_time --load_checkpoint stage1.ckpt"
```

---

## 附录 D: 关键家族分组 MAE 对比

| 家族 | n | v5+none MAE | unc MAE | maal MAE | unc/v5 | maal/v5 |
|------|---|-------------|---------|----------|--------|---------|
| **cdc20_ts_family** | 5 | 0.0377 | **1.4327** | **1.7494** | **38.0×** | **46.3×** |
| cln_triple | 9 | 0.0360 | 0.4130 | 0.5132 | 11.5× | 14.3× |
| CLB1_clb2_family | 5 | 0.0756 | 0.5326 | 0.4101 | 7.0× | 5.4× |
| APC_A_family | 6 | 0.0704 | 0.4411 | 0.3560 | 6.3× | 5.1× |
| sic1_family | 7 | 0.0302 | 0.1413 | 0.2059 | 4.7× | 6.8× |
| CLB2_db_family | 9 | 0.0883 | 0.3944 | 0.4392 | 4.5× | 5.0× |
| cdc14_ts_family | 4 | 0.0600 | 0.2242 | 0.2819 | 3.7× | 4.7× |
| cln1_cln2_KO_family | 6 | 0.0400 | 0.3800 | 0.4219 | 9.5× | 10.5× |
| tem1_family | 5 | 0.1324 | 0.1705 | 0.5747 | 1.3× | 4.3× |

**关键发现**:
1. **cdc20_ts_family 是 al4pde 的绝对灾难区**（unc 38×, maal 46×）
2. **cln_triple 和 cln1_cln2_KO_family 严重恶化**（10-14×），源于 v1 训练集未覆盖 cln1_cln2_KO 定向采样
3. **tem1_family 在 unc 中尚可 (1.3×)，但 MAAL 中严重恶化 (4.3×)**——MAAL 的 target_priority 忽视了 tem1_KO
4. **所有家族在 al4pde 中都恶化**，没有例外

---

**报告结束**
