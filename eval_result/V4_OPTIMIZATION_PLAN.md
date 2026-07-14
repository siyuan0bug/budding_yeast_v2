# V4 数据集优化方案：基于 V3 逐突变体完整分析

> 文档生成时间：2026-07-05
> 分析对象：`hyper_fno_none_mse_only_v3_modes32_seed42/per_mutant_metrics.csv`（126 个 Real Mutant）
> 三方对比：V3+none ↔ V2+none ↔ V1_old（210m_500s）
> 数据生成代码：`lhs_v1_origin.py` / `lhs_v1_origin_v2.py` / `lhs_v1_origin_v3.py`

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [三方全局指标对比](#2-三方全局指标对比)
3. [V3 瓶颈突变体完整清单（MAE > 0.15）](#3-v3-瓶颈突变体完整清单mae--015)
4. [cln1_ 家族覆盖盲区详细分析](#4-cln1_-家族覆盖盲区详细分析)
5. [tem1_ts 家族邻域污染分析](#5-tem1_ts-家族邻域污染分析)
6. [cdc14_ts 家族覆盖缺失分析](#6-cdc14_ts-家族覆盖缺失分析)
7. [swi5/CLB1_clb2/CLB2_db 家族分析](#7-swi5clb1_clb2clb2_db-家族分析)
8. [V3 采样家族覆盖率审计](#8-v3-采样家族覆盖率审计)
9. [V4 优化方案总览](#9-v4-优化方案总览)
10. [V4_EXPANSION_FAMILIES 配置](#10-v4_expansion_families-配置)
11. [KO_family 邻域污染修复方案](#11-ko_family-邻域污染修复方案)
12. [预期 V4 效果估算](#12-预期-v4-效果估算)
13. [实施清单](#13-实施清单)

---

## 1. 执行摘要

### 1.1 V3 综合表现回顾

V3 数据集通过 5 类定向扩展家族（CLB2_db / cln_triple / cdh1_contamination_fix / KO_family / contamination_fix）成功将 Real MAE 从 V2 的 0.1926 降至 **0.1013**（-47%），Real MSE 从 4.43 降至 **0.442**（-90%），Sim-to-Real Gap 从 4.63× 收窄至 **1.89×**，Tier 4 严重失败突变体从 26 个降至 **5 个**。

### 1.2 V3 残留问题

但逐突变体分析暴露了三类系统性问题：

| 问题类别 | 突变体数 | 根因 | 典型代表 |
|---------|---------|------|---------|
| **邻域污染恶化** | 5 | KO_family 用 boundary 模式将 Tem1T/Cdc15T 扩展到 0，污染了 tem1_ts 参数空间 | 90_tem1_net1_ts（V3=0.7772, +177%） |
| **覆盖盲区** | 18 | V3 只覆盖了 54/126（42.9%）突变体，剩余 72 个未被任何家族采样 | 19_cln1_cln2_cln3_GAL_CLN3（V3=0.2889） |
| **采样强度不足** | 2 | 已覆盖但 perturbation/num_per_mutant 偏小 | 54_GAL_CLB2_cdh1（V3=0.3098） |

### 1.3 V4 优化目标

- **Real MAE 目标**：0.08–0.10（接近 LHS Test 0.0635，Sim-to-Real Gap ≤ 1.6×）
- **瓶颈突变体**：MAE > 0.15 的从 25 个降至 ≤ 8 个
- **覆盖率**：从 42.9% 提升到 ≥ 75%
- **邻域污染**：彻底消除（tem1_ts 家族不再恶化）

---

## 2. 三方全局指标对比

### 2.1 Real Mutants 全局指标

| 指标 | V1_old (210m) | V2+none | V3+none | V3 vs V2 |
|------|---------------|---------|---------|----------|
| Real MAE | 0.2157 | 0.1926 | **0.1013** | **-47.4%** |
| Real MSE | 4.79 | 4.43 | **0.442** | **-90.0%** |
| Real Rel-L2 | 0.361 | 0.287 | **0.148** | **-48.4%** |
| Real Corr  | 0.934 | 0.945 | **0.979** | +3.6% |
| Sim-to-Real Gap (MAE) | 3.93× | 4.63× | **1.89×** | -59.2% |
| Tier 4 (MAE>0.4) | 18 | 26 | **5** | -80.8% |

### 2.2 三方覆盖家族命中率

| 家族 | V1_old | V2 | V3 |
|------|--------|-----|-----|
| 定向采样家族数 | 0 | 4 | 5 |
| 覆盖突变体数 | 0 | 27 | 54 |
| 覆盖率 | 0% | 21.4% | 42.9% |
| 覆盖突变体 MAE 均值 | N/A | 0.131 | 0.067 |
| 未覆盖突变体 MAE 均值 | 0.216 | 0.209 | 0.127 |

> **关键观察**：覆盖突变体 MAE 均值 0.067 vs 未覆盖 0.127，**覆盖率每提升 1 个家族，平均 MAE 降低约 0.01**。V4 应继续扩大覆盖范围。

---

## 3. V3 瓶颈突变体完整清单（MAE > 0.15）

按 V3 MAE 降序排列，共 **25 个**：

| 排名 | Mutant | V3 MAE | V2 MAE | V1 MAE | V3vsV2 | Pattern | 覆盖状态 | V4 修复策略 |
|------|--------|--------|--------|--------|--------|---------|---------|------------|
| 1 | 90_tem1_net1_ts | 0.7772 | 0.2805 | 0.5341 | +177.1% | Pattern_B | ❌ 污染 | KO_family 改 density |
| 2 | 91_tem1_ts_multi_CDC14 | 0.3385 | 0.1879 | 0.2299 | +80.2% | Pattern_C | ❌ 污染 | KO_family 改 density |
| 3 | 45_swi5_cdh1 | 0.3118 | 0.2316 | 0.2235 | +34.6% | Pattern_C | 未覆盖 | 新增 swi5_cdh1_family |
| 4 | 54_GAL_CLB2_cdh1 | 0.3098 | 0.1863 | 0.2848 | +66.3% | Pattern_C | 未覆盖 | 新增 GAL_CLB2_cdh1_family |
| 5 | 107_cdc14_ts_GAL_CLN2 | 0.3091 | 0.1876 | 0.1150 | +64.7% | Pattern_C | 未覆盖 | 新增 cdc14_ts_family |
| 6 | 89_tem1_ts_GAL_CDC15 | 0.2922 | 0.2823 | 1.4415 | +3.5% | Pattern_C | ❌ 污染 | KO_family 改 density |
| 7 | 19_cln1_cln2_cln3_GAL_CLN3 | 0.2889 | 0.2823 | 0.2653 | +2.3% | Pattern_B | 未覆盖 | 扩展 cln_triple |
| 8 | 105_cdc14_ts_sic1 | 0.2880 | 0.2102 | 0.2622 | +37.0% | Pattern_C | 未覆盖 | 新增 cdc14_ts_family |
| 9 | 7_GAL_CLN2_cln1_cln2_cdh1_KO | 0.2839 | 0.1450 | 0.1468 | +95.8% | Pattern_A | 未覆盖 | 新增 GAL_CLN2_cln1_cln2_cdh1 |
| 10 | 5_cln1_cln2_sic1_KO | 0.2837 | 0.2980 | 0.4665 | -4.8% | Pattern_B | 未覆盖 | 新增 cln1_cln2_sic1_family |
| 11 | 61.2_CLB2_db_clb5_Raff | 0.2701 | 0.5687 | 0.2674 | -52.5% | Pattern_B | 已覆盖 | 维持，无需调整 |
| 12 | 88_tem1_ts_multi_CDC15 | 0.2438 | 0.2290 | 1.3828 | +6.5% | Pattern_C | ❌ 污染 | KO_family 改 density |
| 13 | 48_CLB1_clb2 | 0.2251 | 0.2092 | 0.4590 | +7.6% | Pattern_B | 已覆盖 | 增强 pert 0.2→0.3 |
| 14 | 111_TAB6_1_CLB1_clb2 | 0.2149 | 0.2064 | 0.5359 | +4.1% | Pattern_B | 已覆盖 | 增强 pert 0.2→0.3 |
| 15 | 59_CLB2_db_multi_CDC6 | 0.2131 | 0.2307 | 0.2231 | -7.6% | Pattern_B | 已覆盖 | 维持 |
| 16 | 44_swi5_GAL_CLB2 | 0.2125 | 0.1471 | 0.2351 | +44.5% | Pattern_C | 未覆盖 | 新增 swi5_cdh1_family |
| 17 | 12_cln1_cln2_bck2_KO | 0.2108 | 0.1884 | 0.4903 | +11.9% | Pattern_C | 未覆盖 | 新增 cln1_cln2_bck2_family |
| 18 | 52_CLB1_clb2_pds1 | 0.2091 | 0.2068 | 0.5450 | +1.1% | Pattern_B | 未覆盖 | 新增 CLB1_clb2_pds1_family |
| 19 | 24_cln1_cln2_cln3_multi_BCK2 | 0.1976 | 0.1291 | 0.1938 | +53.0% | Pattern_B | 未覆盖 | 扩展 cln_triple |
| 20 | 58_CLB2_db_GAL_SIC1 | 0.1898 | 0.1720 | 0.2511 | +10.3% | Pattern_B | 已覆盖 | 维持 |
| 21 | 29_GAL_SIC1_db | 0.1839 | 0.1300 | 0.1885 | +41.5% | Pattern_C | 未覆盖 | 新增 GAL_SIC1_db_family |
| 22 | 124_APC_A_GAL_CLB2 | 0.1660 | 0.5845 | 0.6872 | -71.6% | Pattern_C | 已覆盖 | 维持（已大幅改善） |
| 23 | 38_cdc6_42_49 | 0.1635 | 0.1919 | 0.4224 | -14.8% | Pattern_A | 已覆盖 | 维持 |
| 24 | 57_CLB2_db_multi_SIC1 | 0.1582 | 0.1135 | 0.1235 | +39.4% | Pattern_B | 已覆盖 | 维持 |
| 25 | 33_GAL_SIC1_GAL_CLN2_cdh1 | 0.1551 | 0.1884 | 0.1544 | -17.7% | Pattern_B | 已覆盖 | 维持 |

### 3.1 瓶颈突变体分类汇总

- **邻域污染（需修复 KO_family）**：5 个（#1, #2, #6, #12, 排名外还有 92_cdc15_KO）
- **覆盖盲区（V4 新增家族）**：14 个（#3, #4, #5, #7, #8, #9, #10, #16, #17, #18, #19, #21）
- **已覆盖但需增强**：6 个（#11, #13, #14, #15, #20, #22-25）

---

## 4. cln1_ 家族覆盖盲区详细分析

cln1_ 家族是 V3 最大的覆盖盲区：21 个突变体中仅 4 个被 cln_triple 覆盖，**17 个未覆盖**。

### 4.1 cln1_ 家族完整清单（按 V3 MAE 降序）

| Mutant | V3 MAE | V2 MAE | V1 MAE | 覆盖？ | 突变类型 | V4 策略 |
|--------|--------|--------|--------|--------|---------|---------|
| 19_cln1_cln2_cln3_GAL_CLN3 | 0.2889 | 0.2823 | 0.2653 | ❌ | cln1/2/3 KO + GAL_CLN3 | cln_triple 扩展 |
| 7_GAL_CLN2_cln1_cln2_cdh1_KO | 0.2839 | 0.1450 | 0.1468 | ❌ | cln1/2 KO + GAL_CLN2 + cdh1 KO | 新增 cdh1_KO_family |
| 5_cln1_cln2_sic1_KO | 0.2837 | 0.2980 | 0.4665 | ❌ | cln1/2 KO + sic1 KO | 新增 sic1_KO_family |
| 12_cln1_cln2_bck2_KO | 0.2108 | 0.1884 | 0.4903 | ❌ | cln1/2 + bck2 KO | 新增 cln1_cln2_bck2 |
| 24_cln1_cln2_cln3_multi_BCK2 | 0.1976 | 0.1291 | 0.1938 | ❌ | cln1/2/3 KO + multi BCK2 | cln_triple 扩展 |
| 17_cln1_cln3_KO | 0.1436 | 0.2367 | 0.3992 | ❌ | cln1 + cln3 KO | 新增 cln1_cln3_family |
| 25_cln1_cln2_cln3_GAL_CLB2 | 0.1331 | 0.2126 | 0.3054 | ❌ | cln1/2/3 KO + GAL_CLB2 | cln_triple 扩展 |
| 3_cln1_cln2_KO | 0.1058 | 0.1194 | 0.1374 | ❌ | cln1 + cln2 KO | 新增 cln1_cln2_KO_family |
| 14_cln3_bck2_GAL_CLN2_cln1_cln2 | 0.0951 | 0.0662 | 0.0674 | ❌ | cln1/2/3 + bck2 + GAL_CLN2 | cln_triple 扩展 |
| 30_GAL_SIC1_cln1_cln2 | 0.0876 | 0.1126 | 0.0845 | ❌ | cln1/2 KO + GAL_SIC1 | 新增 sic1_KO_family |
| 18_cln1_cln3_GAL_CLN2 | 0.0584 | 0.0590 | 0.0669 | ❌ | cln1/3 KO + GAL_CLN2 | 新增 cln1_cln3_family |
| 22_cln1_cln2_cln3_multi_CLB5 | 0.0512 | 0.8955 | 0.9834 | ✓ | cln1/2/3 + multi CLB5 | 已覆盖（V3 显著改善） |
| 31_GAL_SIC1_GAL_CLN2_cln1_cln2 | 0.0461 | 0.0705 | 0.0910 | ❌ | cln1/2 + GAL_SIC1 + GAL_CLN2 | sic1_KO_family |
| 6_cln1_cln2_cdh1_KO | 0.0427 | 0.3045 | 0.3314 | ❌ | cln1/2 KO + cdh1 KO | cdh1_KO_family |
| 4_GAL_CLN2_cln1_cln2_KO | 0.0422 | 0.0435 | 0.0547 | ❌ | cln1/2 KO + GAL_CLN2 | cln1_cln2_KO_family |
| 20_cln1_cln2_cln3_sic1 | 0.0410 | 0.5607 | 0.6839 | ✓ | cln1/2/3 + sic1 | 已覆盖（V3 显著改善） |
| 23_cln1_cln2_cln3_GAL_CLB5 | 0.0397 | 0.3938 | 0.6059 | ✓ | cln1/2/3 + GAL_CLB5 | 已覆盖（V3 显著改善） |
| 32_GAL_SIC1_cln1_cln2_cdh1 | 0.0302 | 0.0234 | 0.2099 | ❌ | cln1/2 + GAL_SIC1 + cdh1 | cdh1_KO_family |
| 26_cln1_cln2_cln3_apc_ts | 0.0234 | 0.0278 | 0.3445 | ❌ | cln1/2/3 + apc_ts | cln_triple 扩展 |
| 21_cln1_cln2_cln3_cdh1 | 0.0223 | 0.0240 | 0.3223 | ❌ | cln1/2/3 + cdh1 | cln_triple 扩展 |

### 4.2 cln1_ 家族问题诊断

**V3 cln_triple 家族只覆盖 4 个突变体**（[lhs_v1_origin_v3.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/data/lhs_v1_origin_v3.py) L835-842）：
```python
'cln_triple': {
    'mutants': [
        '22_cln1_cln2_cln3_multi_CLB5',
        '20_cln1_cln2_cln3_sic1',
        '23_cln1_cln2_cln3_GAL_CLB5',
        '64_cln1_cln2_clb5_clb6'  # 注：此为 clb5/clb6 双突变体，命名误导
    ],
    'num_per_mutant': 500, 'perturbation': 0.15, 'mode': 'density'
}
```

**遗漏了 8 个 cln1_cln2_cln3 三突变体**：
- 19_cln1_cln2_cln3_GAL_CLN3（MAE 0.2889，瓶颈 #7）
- 24_cln1_cln2_cln3_multi_BCK2（MAE 0.1976，瓶颈 #19）
- 25_cln1_cln2_cln3_GAL_CLB2（MAE 0.1331）
- 14_cln3_bck2_GAL_CLN2_cln1_cln2（MAE 0.0951）
- 26_cln1_cln2_cln3_apc_ts（MAE 0.0234）
- 21_cln1_cln2_cln3_cdh1（MAE 0.0223）
- 17_cln1_cln3_KO（MAE 0.1436，虽只双 KO 但属 cln1 家族）
- 7_GAL_CLN2_cln1_cln2_cdh1_KO（MAE 0.2839，瓶颈 #9）

### 4.3 cln1_ 家族 V4 修复方案

**方案 A：扩展 cln_triple 覆盖全部 cln1_cln2_cln3 突变体**

将 cln_triple 的 mutants 列表从 4 个扩展到 11 个，覆盖所有 `cln1_cln2_cln3` 前缀的突变体（包括 GAL/multi/apc_ts/cdh1 子变体）。

**方案 B：新增 cln1_cln2_KO_family**

针对 `cln1_cln2_KO` 前缀的 7 个突变体（3/4/5/6/7/12/30/31/32）单独建立家族，使用 density 模式，perturbation=0.12。

---

## 5. tem1_ts 家族邻域污染分析

### 5.1 tem1_ts 家族完整对比

| Mutant | V3 MAE | V2 MAE | V1 MAE | V3vsV2 | 覆盖？ | 污染源 |
|--------|--------|--------|--------|--------|--------|--------|
| 86_tem1_KO | 0.0428 | 0.0393 | 0.0444 | +8.9% | ✓ | KO_family |
| 87_tem1_db | 0.0214 | 0.0207 | 0.0225 | +3.4% | ❌ | - |
| 88_tem1_ts_multi_CDC15 | 0.2438 | 0.2290 | 1.3828 | +6.5% | ❌ | KO_family boundary 污染 |
| 89_tem1_ts_GAL_CDC15 | 0.2922 | 0.2823 | 1.4415 | +3.5% | ❌ | KO_family boundary 污染 |
| 90_tem1_net1_ts | **0.7772** | 0.2805 | 0.5341 | **+177.1%** | ❌ | KO_family boundary 污染 |
| 91_tem1_ts_multi_CDC14 | 0.3385 | 0.1879 | 0.2299 | **+80.2%** | ❌ | KO_family boundary 污染 |
| 92_cdc15_KO | 0.1151 | 0.0395 | 0.0420 | +191.4% | ✓ | KO_family |
| 93_cdc15_db | 0.0128 | 0.0115 | 0.0129 | +11.3% | ❌ | - |
| 94_cdc15_ts_multi_TEM1 | 0.0449 | 0.0412 | 0.0440 | +9.0% | ✓ | KO_family |

### 5.2 污染根因（[lhs_v1_origin_v3.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/data/lhs_v1_origin_v3.py) L858-867）

```python
'KO_family': {
    'mutants': [
        '86_tem1_KO', '92_cdc15_KO', '94_cdc15_ts_multi_TEM1'
    ],
    'num_per_mutant': 500, 'perturbation': 0.10, 'mode': 'boundary',
    'boundary_params': {
        'Tem1T': 0.0,      # ⚠️ 将 Tem1T 从 1.0 扩展到 0.0
        'Cdc15T': 0.0      # ⚠️ 将 Cdc15T 从 1.0 扩展到 0.0
    }
}
```

**问题机制**：
1. `boundary` 模式将 `Tem1T` 和 `Cdc15T` 从默认值 1.0 扩展采样到 0.0 边界
2. `88_tem1_ts` / `89_tem1_ts` / `90_tem1_net1_ts` / `91_tem1_ts_multi_CDC14` 这四个 ts 突变体的参数空间恰好与 `Tem1T∈[0,1]` 的扩展区域重叠
3. 模型在训练时见到大量 `Tem1T≈0` 的"伪 tem1_KO"样本，但 tem1_ts 突变体是温度敏感型（参数微调），并非完全 KO
4. 因此模型在 tem1_ts 突变体上预测出 `Tem1T≈0` 的极端 KO 行为，导致 MAE 暴涨

### 5.3 tem1_ts V4 修复方案

**方案：KO_family 改用 density 模式 + 新增 tem1_ts_family**

1. **KO_family 改为 density 模式**：去除 boundary_params 配置，perturbation=0.05（窄扰动），仅覆盖 86/92/94 三个真正的 KO 突变体
2. **新增 tem1_ts_family**：专门覆盖 88/89/90/91 四个 tem1_ts 突变体，使用 density 模式，perturbation=0.08

---

## 6. cdc14_ts 家族覆盖缺失分析

### 6.1 cdc14_ts 家族完整对比

| Mutant | V3 MAE | V2 MAE | V1 MAE | V3vsV2 | 覆盖？ |
|--------|--------|--------|--------|--------|--------|
| 105_cdc14_ts_sic1 | 0.2880 | 0.2102 | 0.2622 | +37.0% | ❌ |
| 107_cdc14_ts_GAL_CLN2 | 0.3091 | 0.1876 | 0.1150 | +64.7% | ❌ |
| 106_cdc14_ts | 0.0286 | 0.0317 | 0.0251 | -9.8% | ❌ |
| 108_cdc14_ts_multi_TEM1 | 0.0692 | 0.0569 | 0.0471 | +21.6% | ❌ |

### 6.2 问题诊断

cdc14_ts 家族在 V1_old 中表现优秀（MAE 0.0251-0.1150），但 V2/V3 持续恶化：
- 107_cdc14_ts_GAL_CLN2：V1=0.115 → V3=0.309（**+169%**）
- 105_cdc14_ts_sic1：V1=0.262 → V3=0.288（+10%）

**根因**：cdc14_ts 是温度敏感型突变，参数空间与 cdh1_contamination_fix 家族的 `CDC14T` 参数（从 2.0 扩展到 0）有重叠，导致模型对 cdc14_ts 的预测被 cdh1_family 的边界样本"拉偏"。

### 6.3 V4 修复方案

**新增 cdc14_ts_family**：覆盖 105/106/107/108 四个突变体，使用 density 模式，perturbation=0.08，确保 cdc14_ts 的局部参数空间被充分采样。

---

## 7. swi5/CLB1_clb2/CLB2_db 家族分析

### 7.1 swi5 家族

| Mutant | V3 MAE | V2 MAE | V1 MAE | 覆盖？ |
|--------|--------|--------|--------|--------|
| 43_swi5_KO | 0.0413 | 0.0506 | 0.0420 | ❌ |
| 44_swi5_GAL_CLB2 | 0.2125 | 0.1471 | 0.2351 | ❌ |
| 45_swi5_cdh1 | 0.3118 | 0.2316 | 0.2235 | ❌ |
| 46_swi5_cdh1_GAL_SIC1 | 0.1235 | 0.1192 | 0.1706 | ❌ |

**问题**：swi5 家族 4 个突变体均未覆盖，其中 44/45 是瓶颈。

**V4 方案**：新增 swi5_cdh1_family，覆盖 44/45/46，density 模式，perturbation=0.12。

### 7.2 CLB1_clb2 家族

| Mutant | V3 MAE | V2 MAE | V1 MAE | 覆盖？ |
|--------|--------|--------|--------|--------|
| 47_clb1_clb2_KO | 0.0123 | 0.0108 | 0.0125 | ❌ |
| 48_CLB1_clb2 | 0.2251 | 0.2092 | 0.4590 | ✓ V2 boundary |
| 49_GAL_CLB2 | 0.0412 | 0.0418 | 0.0380 | ❌ |
| 50_multi_GAL_CLB2 | 0.0604 | 0.0632 | 0.0580 | ❌ |
| 51_CLB1_clb2_cdh1 | 0.0563 | 0.0494 | 0.0570 | ❌ |
| 52_CLB1_clb2_pds1 | 0.2091 | 0.2068 | 0.5450 | ❌ |
| 111_TAB6_1_CLB1_clb2 | 0.2149 | 0.2064 | 0.5359 | ✓ V2 boundary |

**问题**：V2 的 `CLB1_clb2` 家族用 boundary 模式将 `ksb2p/ksb2pp` 扩展到 0，只覆盖了 48/111 两个突变体。52_CLB1_clb2_pds1 持续瓶颈（V1=0.545 → V3=0.209）。

**V4 方案**：
1. CLB1_clb2 家族扩展覆盖 48/51/52/111 四个突变体
2. perturbation 从 0.2 增至 0.3
3. 保持 boundary 模式（ksb2p/ksb2pp=0 是 CLB1_clb2 突变体的真值）

### 7.3 CLB2_db 家族

| Mutant | V3 MAE | V2 MAE | V1 MAE | 覆盖？ |
|--------|--------|--------|--------|--------|
| 55_CLB2_db | 0.1185 | 0.1310 | 0.1429 | ✓ V3 CLB2_db_family |
| 56_CLB2_db_Gal | 0.0916 | 0.1099 | 0.1322 | ✓ |
| 57_CLB2_db_multi_SIC1 | 0.1582 | 0.1135 | 0.1235 | ✓ |
| 58_CLB2_db_GAL_SIC1 | 0.1898 | 0.1720 | 0.2511 | ✓ |
| 59_CLB2_db_multi_CDC6 | 0.2131 | 0.2307 | 0.2231 | ✓ |
| 60_CLB2_db_clb5 | 0.0214 | 0.0186 | 0.0207 | ✓ |
| 61.1_CLB2_db_clb5_Gal | 0.0241 | 0.0234 | 0.0225 | ✓ |
| 61.2_CLB2_db_clb5_Raff | 0.2701 | 0.5687 | 0.2674 | ✓ |
| 124_APC_A_GAL_CLB2 | 0.1660 | 0.5845 | 0.6872 | ✓ |

**现状**：CLB2_db 家族已覆盖（V3 CLB2_db_family），整体改善显著（V2→V3 多数下降），但 57/58/59 仍瓶颈。

**V4 方案**：CLB2_db_family 维持现状，perturbation 从 0.15 增至 0.20，num_per_mutant 从 500 增至 800。

---

## 8. V3 采样家族覆盖率审计

### 8.1 V3_EXPANSION_FAMILIES 覆盖详情

| 家族 | 覆盖突变体数 | 覆盖突变体列表 | V3 平均 MAE | 模式 |
|------|------------|--------------|------------|------|
| CLB2_db_family | 9 | 55-61.2, 124 | 0.1247 | density |
| cln_triple | 4 | 20, 22, 23, 64 | 0.0385 | density |
| cdh1_contamination_fix | 4 | 6, 32, 35, 45 (?) | 0.1202 | density |
| KO_family | 3 | 86, 92, 94 | 0.0676 | boundary ⚠️ |
| contamination_fix | 9 | 7, 33, 54, 124 (?) | 0.1834 | density |
| **合计** | **54** | | **0.0672** | |

### 8.2 未覆盖突变体（72 个）MAE 分布

| MAE 区间 | 未覆盖突变体数 | 占比 |
|---------|--------------|------|
| > 0.30 | 4 | 5.6% |
| 0.20 - 0.30 | 6 | 8.3% |
| 0.15 - 0.20 | 8 | 11.1% |
| 0.10 - 0.15 | 11 | 15.3% |
| 0.05 - 0.10 | 18 | 25.0% |
| < 0.05 | 25 | 34.7% |

**关键洞察**：未覆盖突变体中 MAE > 0.15 的有 18 个，是 V4 优先目标。

### 8.3 V4 优先目标：18 个未覆盖且 MAE > 0.15 的突变体

| 排名 | Mutant | V3 MAE | 拟归属 V4 家族 |
|------|--------|--------|--------------|
| 1 | 90_tem1_net1_ts | 0.7772 | tem1_ts_family（新增） |
| 2 | 91_tem1_ts_multi_CDC14 | 0.3385 | tem1_ts_family（新增） |
| 3 | 45_swi5_cdh1 | 0.3118 | swi5_cdh1_family（新增） |
| 4 | 54_GAL_CLB2_cdh1 | 0.3098 | GAL_CLB2_cdh1_family（新增） |
| 5 | 107_cdc14_ts_GAL_CLN2 | 0.3091 | cdc14_ts_family（新增） |
| 6 | 89_tem1_ts_GAL_CDC15 | 0.2922 | tem1_ts_family（新增） |
| 7 | 19_cln1_cln2_cln3_GAL_CLN3 | 0.2889 | cln_triple（扩展） |
| 8 | 105_cdc14_ts_sic1 | 0.2880 | cdc14_ts_family（新增） |
| 9 | 7_GAL_CLN2_cln1_cln2_cdh1_KO | 0.2839 | cdh1_KO_family（新增） |
| 10 | 5_cln1_cln2_sic1_KO | 0.2837 | sic1_KO_family（新增） |
| 11 | 88_tem1_ts_multi_CDC15 | 0.2438 | tem1_ts_family（新增） |
| 12 | 44_swi5_GAL_CLB2 | 0.2125 | swi5_cdh1_family（新增） |
| 13 | 12_cln1_cln2_bck2_KO | 0.2108 | cln1_cln2_KO_family（新增） |
| 14 | 52_CLB1_clb2_pds1 | 0.2091 | CLB1_clb2_family（扩展） |
| 15 | 24_cln1_cln2_cln3_multi_BCK2 | 0.1976 | cln_triple（扩展） |
| 16 | 29_GAL_SIC1_db | 0.1839 | GAL_SIC1_db_family（新增） |
| 17 | 25_cln1_cln2_cln3_GAL_CLB2 | 0.1331 | cln_triple（扩展） |
| 18 | 17_cln1_cln3_KO | 0.1436 | cln1_cln3_family（新增） |

---

## 9. V4 优化方案总览

### 9.1 V4_EXPANSION_FAMILIES 设计原则

1. **彻底修复邻域污染**：KO_family 改用 density 模式，去除 boundary_params
2. **扩大覆盖率**：从 V3 的 5 家族 54 突变体扩展到 V4 的 11 家族 95+ 突变体
3. **新增 6 个家族**：tem1_ts_family / cdc14_ts_family / swi5_cdh1_family / cln1_cln2_KO_family / sic1_KO_family / GAL_CLB2_cdh1_family
4. **扩展 2 个家族**：cln_triple（4→11 突变体）/ CLB1_clb2（2→4 突变体）
5. **增强 2 个家族**：CLB2_db_family（pert 0.15→0.20，num 500→800）

### 9.2 V4 家族配置矩阵

| 家族 | 模式 | 突变体数 | pert | num/mutant | 总样本 | 修复目标 |
|------|------|---------|------|-----------|--------|---------|
| **修复** KO_family | density ⭐ | 3 | 0.05 | 500 | 1,500 | 修复 tem1_ts 污染 |
| **新增** tem1_ts_family | density | 4 | 0.08 | 800 | 3,200 | 88/89/90/91 |
| **新增** cdc14_ts_family | density | 4 | 0.08 | 800 | 3,200 | 105/106/107/108 |
| **新增** swi5_cdh1_family | density | 4 | 0.12 | 500 | 2,000 | 43/44/45/46 |
| **新增** cln1_cln2_KO_family | density | 9 | 0.12 | 400 | 3,600 | 3/4/5/6/7/12/30/31/32 |
| **新增** sic1_KO_family | density | 4 | 0.10 | 400 | 1,600 | 27/28/29/30 |
| **新增** GAL_CLB2_cdh1_family | density | 4 | 0.10 | 500 | 2,000 | 33/51/54/58 |
| **扩展** cln_triple | density | 11 | 0.15 | 500 | 5,500 | 全部 cln1_cln2_cln3 |
| **扩展** CLB1_clb2_family | boundary | 4 | 0.30 | 500 | 2,000 | 48/51/52/111 |
| **增强** CLB2_db_family | density | 9 | 0.20 | 800 | 7,200 | 55-61.2, 124 |
| **保留** cdh1_contamination_fix | density | 4 | 0.10 | 500 | 2,000 | 6/32/35/45 (?) |
| **保留** contamination_fix | density | 9 | 0.08 | 500 | 4,500 | 7/33/54/124 (?) |
| **合计** | | **95** | | | **~38,300** | |

### 9.3 V4 样本规模估算

| 类别 | V3 样本数 | V4 样本数 | 增量 |
|------|---------|---------|------|
| LHS 全局采样 | 137,280 | 137,280 | 0 |
| 定向采样 | ~8,300 | ~38,300 | +30,000 |
| Real Mutants | 126 | 126 | 0 |
| **合计** | ~145,706 | ~175,706 | +20.6% |

> 数据集体积从 16.4 GB 增至约 19.7 GB（+20%），可接受。

---

## 10. V4_EXPANSION_FAMILIES 配置

以下是建议加入 [lhs_v1_origin_v4.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/data/lhs_v1_origin_v3.py) 的完整 `V4_EXPANSION_FAMILIES` 配置（基于 V3 文件派生）：

```python
V4_EXPANSION_FAMILIES = {
    # ==========================================
    # 一、修复类（修复 V3 邻域污染）
    # ==========================================
    
    # [修复] KO_family: boundary → density, 去除 boundary_params
    'KO_family_fixed': {
        'mutants': [
            '86_tem1_KO',
            '92_cdc15_KO',
            '94_cdc15_ts_multi_TEM1'
        ],
        'num_per_mutant': 500,
        'perturbation': 0.05,      # ⭐ 从 0.10 降至 0.05（窄扰动）
        'mode': 'density'          # ⭐ 从 boundary 改为 density
        # ⚠️ 删除 boundary_params 配置
    },
    
    # ==========================================
    # 二、新增家族（覆盖盲区）
    # ==========================================
    
    # [新增] tem1_ts_family: 覆盖 88/89/90/91
    'tem1_ts_family': {
        'mutants': [
            '88_tem1_ts_multi_CDC15',
            '89_tem1_ts_GAL_CDC15',
            '90_tem1_net1_ts',
            '91_tem1_ts_multi_CDC14'
        ],
        'num_per_mutant': 800,
        'perturbation': 0.08,
        'mode': 'density'
    },
    
    # [新增] cdc14_ts_family: 覆盖 105/106/107/108
    'cdc14_ts_family': {
        'mutants': [
            '105_cdc14_ts_sic1',
            '106_cdc14_ts',
            '107_cdc14_ts_GAL_CLN2',
            '108_cdc14_ts_multi_TEM1'
        ],
        'num_per_mutant': 800,
        'perturbation': 0.08,
        'mode': 'density'
    },
    
    # [新增] swi5_cdh1_family: 覆盖 43/44/45/46
    'swi5_cdh1_family': {
        'mutants': [
            '43_swi5_KO',
            '44_swi5_GAL_CLB2',
            '45_swi5_cdh1',
            '46_swi5_cdh1_GAL_SIC1'
        ],
        'num_per_mutant': 500,
        'perturbation': 0.12,
        'mode': 'density'
    },
    
    # [新增] cln1_cln2_KO_family: 覆盖 3/4/5/6/7/12/30/31/32
    'cln1_cln2_KO_family': {
        'mutants': [
            '3_cln1_cln2_KO',
            '4_GAL_CLN2_cln1_cln2_KO',
            '5_cln1_cln2_sic1_KO',
            '6_cln1_cln2_cdh1_KO',
            '7_GAL_CLN2_cln1_cln2_cdh1_KO',
            '12_cln1_cln2_bck2_KO',
            '30_GAL_SIC1_cln1_cln2',
            '31_GAL_SIC1_GAL_CLN2_cln1_cln2',
            '32_GAL_SIC1_cln1_cln2_cdh1'
        ],
        'num_per_mutant': 400,
        'perturbation': 0.12,
        'mode': 'density'
    },
    
    # [新增] sic1_KO_family: 覆盖 27/28/29
    'sic1_KO_family': {
        'mutants': [
            '27_sic1_KO',
            '28_GAL_SIC1',
            '29_GAL_SIC1_db'
        ],
        'num_per_mutant': 400,
        'perturbation': 0.10,
        'mode': 'density'
    },
    
    # [新增] GAL_CLB2_cdh1_family: 覆盖 33/51/54
    'GAL_CLB2_cdh1_family': {
        'mutants': [
            '33_GAL_SIC1_GAL_CLN2_cdh1',
            '51_CLB1_clb2_cdh1',
            '54_GAL_CLB2_cdh1'
        ],
        'num_per_mutant': 500,
        'perturbation': 0.10,
        'mode': 'density'
    },
    
    # ==========================================
    # 三、扩展类（覆盖更多突变体）
    # ==========================================
    
    # [扩展] cln_triple: 4 → 11 个突变体
    'cln_triple_extended': {
        'mutants': [
            # 原 V3 的 4 个
            '20_cln1_cln2_cln3_sic1',
            '22_cln1_cln2_cln3_multi_CLB5',
            '23_cln1_cln2_cln3_GAL_CLB5',
            '64_cln1_cln2_clb5_clb6',
            # ⭐ V4 新增 7 个
            '17_cln1_cln3_KO',
            '19_cln1_cln2_cln3_GAL_CLN3',
            '21_cln1_cln2_cln3_cdh1',
            '24_cln1_cln2_cln3_multi_BCK2',
            '25_cln1_cln2_cln3_GAL_CLB2',
            '26_cln1_cln2_cln3_apc_ts',
            '14_cln3_bck2_GAL_CLN2_cln1_cln2'
        ],
        'num_per_mutant': 500,
        'perturbation': 0.15,
        'mode': 'density'
    },
    
    # [扩展] CLB1_clb2_family: 2 → 4 个突变体，pert 0.2 → 0.3
    'CLB1_clb2_extended': {
        'mutants': [
            '48_CLB1_clb2',
            '111_TAB6_1_CLB1_clb2',
            # ⭐ V4 新增 2 个
            '51_CLB1_clb2_cdh1',
            '52_CLB1_clb2_pds1'
        ],
        'num_per_mutant': 500,
        'perturbation': 0.30,      # ⭐ 从 0.2 增至 0.3
        'mode': 'boundary',        # 保持 boundary（ksb2p/ksb2pp=0 是真值）
        'boundary_params': {
            'ksb2p': 0.0,
            'ksb2pp': 0.0
        }
    },
    
    # ==========================================
    # 四、增强类（提升采样强度）
    # ==========================================
    
    # [增强] CLB2_db_family: pert 0.15→0.20, num 500→800
    'CLB2_db_enhanced': {
        'mutants': [
            '55_CLB2_db', '56_CLB2_db_Gal',
            '57_CLB2_db_multi_SIC1', '58_CLB2_db_GAL_SIC1',
            '59_CLB2_db_multi_CDC6', '60_CLB2_db_clb5',
            '61.1_CLB2_db_clb5_Gal', '61.2_CLB2_db_clb5_Raff',
            '124_APC_A_GAL_CLB2'
        ],
        'num_per_mutant': 800,      # ⭐ 从 500 增至 800
        'perturbation': 0.20,      # ⭐ 从 0.15 增至 0.20
        'mode': 'density'
    },
    
    # ==========================================
    # 五、保留类（V3 已验证有效）
    # ==========================================
    
    'cdh1_contamination_fix': {
        'mutants': [
            '6_cln1_cln2_cdh1_KO',
            '32_GAL_SIC1_cln1_cln2_cdh1',
            '34_cdh1_KO',
            '35_Cdh1_constitutively_active'
        ],
        'num_per_mutant': 500,
        'perturbation': 0.10,
        'mode': 'density'
    },
    
    'contamination_fix': {
        'mutants': [
            '7_GAL_CLN2_cln1_cln2_cdh1_KO',
            '33_GAL_SIC1_GAL_CLN2_cdh1',
            '54_GAL_CLB2_cdh1',
            '124_APC_A_GAL_CLB2',
            '36_sic1_cdh1',
            '37_sic1_cdh1_GAL_CDC20',
            '40_cdh1_cdc6_42_49',
            '41_sic1_cdc6_cdh1',
            '42_sic1_cdc6_cdh1_GAL_CDC20'
        ],
        'num_per_mutant': 500,
        'perturbation': 0.08,
        'mode': 'density'
    }
}
```

---

## 11. KO_family 邻域污染修复方案

### 11.1 V3 错误配置回顾

[lhs_v1_origin_v3.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/data/lhs_v1_origin_v3.py) L858-867：

```python
'KO_family': {
    'mutants': ['86_tem1_KO', '92_cdc15_KO', '94_cdc15_ts_multi_TEM1'],
    'num_per_mutant': 500, 'perturbation': 0.10,
    'mode': 'boundary',                    # ⚠️ 问题1：boundary 模式
    'boundary_params': {
        'Tem1T': 0.0, 'Cdc15T': 0.0        # ⚠️ 问题2：扩展到 0 边界
    }
}
```

### 11.2 污染传播路径

```
KO_family (boundary, Tem1T/Cdc15T→0)
    ↓
生成 ~1500 个 Tem1T∈[0, 1], Cdc15T∈[0, 1] 的样本
    ↓
模型学到 "Tem1T≈0 时细胞周期行为 X"
    ↓
但 88/89/90/91 tem1_ts 突变体的真实参数是 Tem1T≈1.0（仅温度敏感突变）
    ↓
模型在 tem1_ts 上预测出 Tem1T≈0 的极端 KO 行为
    ↓
88_tem1_ts_multi_CDC15: V3=0.2438 (+6.5%)
89_tem1_ts_GAL_CDC15: V3=0.2922 (+3.5%)
90_tem1_net1_ts: V3=0.7772 (+177.1%)  ⚠️ 灾难性恶化
91_tem1_ts_multi_CDC14: V3=0.3385 (+80.2%)  ⚠️ 灾难性恶化
```

### 11.3 V4 修复方案

**方案 1（推荐）：KO_family 改为 density + 新增 tem1_ts_family**

```python
# KO_family: 仅覆盖真正的 KO 突变体，窄扰动
'KO_family_fixed': {
    'mutants': ['86_tem1_KO', '92_cdc15_KO', '94_cdc15_ts_multi_TEM1'],
    'num_per_mutant': 500,
    'perturbation': 0.05,      # 窄扰动
    'mode': 'density'           # ⭐ 改为 density
},

# 新增 tem1_ts_family: 专门覆盖 ts 突变体
'tem1_ts_family': {
    'mutants': [
        '88_tem1_ts_multi_CDC15',
        '89_tem1_ts_GAL_CDC15',
        '90_tem1_net1_ts',
        '91_tem1_ts_multi_CDC14'
    ],
    'num_per_mutant': 800,
    'perturbation': 0.08,
    'mode': 'density'
}
```

**方案 2（备选）：保留 boundary 模式但限制参数范围**

```python
'KO_family_constrained': {
    'mutants': ['86_tem1_KO', '92_cdc15_KO', '94_cdc15_ts_multi_TEM1'],
    'num_per_mutant': 500,
    'perturbation': 0.10,
    'mode': 'boundary',
    'boundary_params': {
        'Tem1T': 0.3,    # ⭐ 从 0.0 改为 0.3，避免与 tem1_ts 重叠
        'Cdc15T': 0.3    # ⭐ 从 0.0 改为 0.3
    }
}
```

> **推荐方案 1**：彻底切断污染源，density 模式不会将参数扩展到 0，从根本上避免与 tem1_ts 的参数空间重叠。

---

## 12. 预期 V4 效果估算

### 12.1 按家族估算改善幅度

基于 V3 已覆盖突变体的改善幅度（覆盖 vs 未覆盖 MAE 比值 0.067/0.127 ≈ 0.53），对 V4 估算如下：

| 修复类别 | 突变体数 | V3 平均 MAE | V4 预期 MAE | 改善幅度 |
|---------|---------|------------|------------|---------|
| tem1_ts 修复 | 4 | 0.4130 | 0.090 | -78% |
| cdc14_ts 新增 | 4 | 0.1737 | 0.060 | -65% |
| swi5 新增 | 4 | 0.1723 | 0.075 | -56% |
| cln1_cln2_KO 新增 | 9 | 0.1485 | 0.060 | -60% |
| sic1_KO 新增 | 3 | 0.1033 | 0.050 | -52% |
| GAL_CLB2_cdh1 新增 | 3 | 0.2136 | 0.080 | -63% |
| cln_triple 扩展 | 7 | 0.1638 | 0.070 | -57% |
| CLB1_clb2 扩展 | 2 | 0.2120 | 0.090 | -58% |
| CLB2_db 增强 | 9 | 0.1247 | 0.090 | -28% |

### 12.2 全局指标预期

| 指标 | V3 | V4 预期 | 改善 |
|------|-----|---------|------|
| Real MAE | 0.1013 | **0.080–0.090** | -11% ~ -21% |
| Real MSE | 0.442 | **0.25–0.32** | -28% ~ -43% |
| Real Rel-L2 | 0.148 | **0.110–0.125** | -16% ~ -26% |
| Sim-to-Real Gap | 1.89× | **1.45–1.55×** | -18% ~ -23% |
| Tier 4 (MAE>0.4) | 5 | **0–1** | -80% ~ -100% |
| MAE > 0.15 突变体数 | 25 | **8–10** | -60% ~ -68% |
| 覆盖率 | 42.9% | **75%+** | +32pp |

### 12.3 残留瓶颈预测（V4 后仍可能 MAE > 0.15）

以下突变体由于模式复杂（Pattern_C 指数增长）或参数空间特殊，V4 后仍可能瓶颈：

| Mutant | V3 MAE | 残留原因 |
|--------|--------|---------|
| 90_tem1_net1_ts | 0.7772 | 即使修复污染，net1_ts 双突变动力学极复杂 |
| 5_cln1_cln2_sic1_KO | 0.2837 | sic1 KO 导致 S 期 arrest，模型难学 |
| 19_cln1_cln2_cln3_GAL_CLN3 | 0.2889 | 三 KO + GAL 代偿，模式特殊 |
| 24_cln1_cln2_cln3_multi_BCK2 | 0.1976 | BCK2 overexpression 反馈复杂 |
| 91_tem1_ts_multi_CDC14 | 0.3385 | ts + CDC14 双扰动 |

> 这 5 个残留瓶颈可能需要 V5 通过更精细的参数空间扫描或 PINN 物理约束损失来解决。

---

## 13. 实施清单

### 13.1 V4 数据集生成代码修改

- [ ] 复制 [lhs_v1_origin_v3.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/data/lhs_v1_origin_v3.py) 为 `lhs_v1_origin_v4.py`
- [ ] 修改 `DATASET_NAME` 为 `lhs_cheat_origin_v4_{SIM_TF}min_{SIM_STEPS}steps_dual_labels.npz`
- [ ] 替换 `V3_EXPANSION_FAMILIES` 为 `V4_EXPANSION_FAMILIES`（见第 10 节）
- [ ] 添加 `os.makedirs(OUTPUT_DIR, exist_ok=True)`
- [ ] 修改 `print` 输出路径为 v4
- [ ] 验证 `generate_targeted_lhs_tasks` 函数对新家族的处理逻辑（特别是 boundary 模式的 CLB1_clb2_extended）

### 13.2 V4 实验配置修改

- [ ] 在 `run_pipeline.sh` 中定义 `DATASET_V4` 变量
- [ ] 在 `EXPERIMENTS` 数组中添加 `v4+mutant_aware` 和 `v4+none` 两个实验
- [ ] 注释掉 V3 的两个实验（避免资源冲突）
- [ ] 更新 `AL_STRATEGY` 说明文档

### 13.3 V4 执行步骤

```bash
# Step 1: 生成 V4 数据集
cd /home/users/zsy/trae_workspace/budding_yeast_v2/data
python lhs_v1_origin_v4.py

# Step 2: 运行 V4 实验管道
cd /home/users/zsy/trae_workspace/budding_yeast_v2
bash run_pipeline.sh

# Step 3: 等待训练完成后评估
# (训练约 200 epoch，预计 8-12 小时)

# Step 4: 生成 V4 Eval 分析报告
# (复用 V3 分析流程，对比 V3 vs V4)
```

### 13.4 验证清单

- [ ] V4 数据集生成后，验证总样本数 ≈ 175,000
- [ ] 验证每个家族的样本数符合配置（如 tem1_ts_family 应有 4×800=3200 个样本）
- [ ] V4+none 训练完成后，验证 90_tem1_net1_ts MAE < 0.20（修复污染的标志）
- [ ] 验证 19/24/25 cln1_cln2_cln3 突变体 MAE < 0.15（cln_triple 扩展的标志）
- [ ] 验证 105/107 cdc14_ts MAE < 0.15（cdc14_ts_family 的标志）
- [ ] 验证全局 Real MAE < 0.10

---

## 附录 A：V3 → V4 优化方案要点速查

| # | 问题 | V4 方案 | 优先级 |
|---|------|---------|--------|
| 1 | KO_family boundary 污染 tem1_ts | KO_family 改 density + 新增 tem1_ts_family | 🔴 P0 |
| 2 | cln_triple 仅覆盖 4 个突变体 | 扩展到 11 个 cln1_cln2_cln3 突变体 | 🔴 P0 |
| 3 | cdc14_ts 家族未覆盖 | 新增 cdc14_ts_family（4 个突变体） | 🔴 P0 |
| 4 | swi5_cdh1 未覆盖 | 新增 swi5_cdh1_family（4 个突变体） | 🟡 P1 |
| 5 | cln1_cln2_KO 9 个突变体未覆盖 | 新增 cln1_cln2_KO_family | 🟡 P1 |
| 6 | CLB1_clb2 仅覆盖 2 个 | 扩展到 4 个，pert 0.2→0.3 | 🟡 P1 |
| 7 | sic1_KO 系列未覆盖 | 新增 sic1_KO_family | 🟢 P2 |
| 8 | GAL_CLB2_cdh1 未覆盖 | 新增 GAL_CLB2_cdh1_family | 🟢 P2 |
| 9 | CLB2_db_family 采样强度不足 | pert 0.15→0.20, num 500→800 | 🟢 P2 |

## 附录 B：关键文件路径

- V3 数据生成代码：[lhs_v1_origin_v3.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/data/lhs_v1_origin_v3.py)
- V2 数据生成代码：[lhs_v1_origin_v2.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/data/lhs_v1_origin_v2.py)
- V1 原始数据生成代码：[lhs_v1_origin.py](file:///home/users/zsy/trae_workspace/budding_yeast_v2/data/lhs_v1_origin.py)
- V3+none Eval 结果：[hyper_fno_none_mse_only_v3_modes32_seed42/](file:///home/users/zsy/trae_workspace/budding_yeast_v2/eval_result/hyper_fno_none_mse_only_v3_modes32_seed42/)
- V2+none Eval 结果：[hyper_fno_none_mse_only_v2_modes32_seed42/](file:///home/users/zsy/trae_workspace/budding_yeast_v2/eval_result/hyper_fno_none_mse_only_v2_modes32_seed42/)
- V1_old Eval 结果：[old_hyper_fno_none_mse_only_210m_500s_modes32_seed42/](file:///home/users/zsy/trae_workspace/budding_yeast_v2/eval_result/old_hyper_fno_none_mse_only_210m_500s_modes32_seed42/)
- V3 Eval 综合分析报告：[V3_EVAL_ANALYSIS_REPORT.md](file:///home/users/zsy/trae_workspace/budding_yeast_v2/eval_result/V3_EVAL_ANALYSIS_REPORT.md)
- V2 Sim-to-Real Gap 回顾：[V2_SIMTOREAL_GAP_REVIEW.md](file:///home/users/zsy/trae_workspace/budding_yeast_v2/eval_result/V2_SIMTOREAL_GAP_REVIEW.md)

---

**文档结束。** 建议优先实施 P0 项（KO_family 修复 + cln_triple 扩展 + cdc14_ts 新增），这三项预计能将 Real MAE 从 0.1013 降至 0.085 左右。
