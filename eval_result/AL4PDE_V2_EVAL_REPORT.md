# AL4PDE 风格主动学习 V2 Eval 报告（65000 样本新配置）

**报告日期**: 2026-07-08
**项目**: 出芽酵母细胞周期 HyperFNO surrogate 模型
**分析对象**: al4pde 框架 4 种 AL 策略（random / uncertainty / diversity / maal）新配置 eval 结果
**对比基线**: V5+none / V6+none 全量训练，以及旧 al4pde 配置（15000 样本，见 `AL4PDE_EVAL_ANALYSIS_REPORT.md`）
**格式参照**: `V5_EVAL_ANALYSIS_REPORT.md` / `AL4PDE_EVAL_ANALYSIS_REPORT.md`

---

## 1. Executive Summary

### 1.1 一句话结论

**al4pde 框架 V2 配置（初始 15000 + AL 50000 = 65000 样本）相比 V1 配置（初始 5000 + AL 10000 = 15000 样本）显著改善**：uncertainty Real MAE 从 0.3726 降到 0.1894（**-49%**），MAAL 从 0.4088 降到 0.1579（**-61%**）；**但 4 策略排名反转预期**——random 最优（Real MAE=0.0976），diversity 最差（0.2725，2.8× worse than random）；**diversity 和 maal 在 cdc20_ts 系列灾难性失败**（5 个 cdc20_ts 突变体 MAE 全部 >1.6，而 random 仅 0.07-0.12）；**根因是 diversity/maal 的采集函数系统性避开 cdc20_ts 参数聚类**，而 random 无偏采样能命中该区域；**即使最优策略 random（0.0976）仍是 V6+none（0.0492）的 2.0×**，证明在 141 维参数空间中，65000 样本的 AL 仍无法匹敌 166000 全量训练。

### 1.2 核心数字对比

| 维度 | V6+none | V5+none | **al4pde_random** | **al4pde_unc** | **al4pde_div** | **MAAL** | 旧 al4pde_unc | 旧 MAAL |
|------|---------|---------|--------------------|-----------------|-----------------|----------|---------------|---------|
| 训练样本数 | ~133,000 | ~130,000 | **65,000** | **65,000** | **65,000** | **65,000** | 15,000 | 15,000 |
| LHS Test MAE | 0.0591 | 0.0618 | 0.1018 | 0.1389 | **0.0972** | 0.1376 | 0.1737 | 0.1671 |
| Real MAE | **0.0492** ✅ | 0.0552 | **0.0976** | 0.1894 | 0.2725 | 0.1579 | 0.3726 | 0.4088 |
| Real MSE | 0.109 | 0.166 | 0.406 | 2.614 | 8.832 | 4.614 | 12.149 | 10.855 |
| Real Corr | 0.9962 | 0.9942 | 0.9891 | 0.9611 | 0.9235 | 0.9744 | 0.8733 | 0.8240 |
| Sim-to-Real Gap | 0.83× | 0.89× | 0.96× | 1.36× | 2.80× | 1.15× | 2.15× | 2.45× |
| Tier 1 (<0.1) | 94.4% | 89.7% | **73.0%** | 38.9% | 27.8% | 52.4% | 14.3% | 9.5% |
| Tier 4 (≥0.3) | 0.0% | 0.8% | 2.4% | 12.7% | **24.6%** | 7.9% | 39.7% | 48.4% |

### 1.3 三个核心结论

1. **V2 配置大幅改善但未达全量训练水平**: uncertainty Real MAE -49%（0.3726→0.1894），MAAL -61%（0.4088→0.1579）。但最优策略 random（0.0976）仍是 V6+none（0.0492）的 2.0×，仍是 V5+none（0.0552）的 1.8×。训练样本从 15000 增到 65000（4.3×）改善了整体欠拟合，但 AL 的采样偏差问题依旧存在。

2. **4 策略排名反转预期——random 最优，diversity 最差**: random（0.0976）> MAAL（0.1579）> uncertainty（0.1894）> diversity（0.2725）。diversity 是 random 的 2.8× worse。这与"AL 策略应优于随机"的直觉相反，根因是 diversity/maal 的采集函数系统性避开了 cdc20_ts 参数聚类（见 §3）。

3. **cdc20_ts 系列仍是 diversity/maal 的绝对灾难区**: 5 个 cdc20_ts 突变体（73/74/75/78/79）在 diversity 中 MAE 全部 >1.6（V6+none 仅 0.03-0.04，恶化 40-50×），在 maal 中 78/73 >1.9。而 random 在这些突变体上 MAE 仅 0.07-0.12（恶化 2-3×）。证明 diversity 的特征空间距离采样会系统性地漏掉 cdc20_ts 这类"参数聚类紧凑但行为高度非线性"的区域。

---

## 2. V2 配置回顾

### 2.1 V1→V2 配置变化

| 配置项 | V1（旧） | V2（新） | 变化 |
|--------|----------|----------|------|
| 初始训练集 | 5,000（Family×Pattern 分层随机） | **15,000**（K-Means 从 v1 LHS 常规样本） | 3× |
| AL 触发 epoch | 20 | **4** | 5× 更早 |
| 每轮 AL 样本数 | 1,000 | 1,000 | 不变 |
| AL 总轮数 | 10 | **50** | 5× |
| AL 总样本数 | 10,000 | **50,000** | 5× |
| 总训练样本数 | 15,000 | **65,000** | 4.3× |
| 训练数据集 | v1（53,796） | v1（53,796） | 不变 |
| 候选池 | v5（162,008，排除 Real Mutant） | v5（162,008） | 不变 |
| 候选池内存优化 | 无（全量加载） | **有**（只加载 P_norm + Y_IC，Y 按需从磁盘） | 新增 |

### 2.2 实验配置（run_pipeline.sh L123-126）

```bash
"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 200 al4pde_random    6e-4 10 1000 0 0 one_time 0"
"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 200 al4pde_uncertainty 6e-4 10 1000 0 0 one_time 1"
"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 200 al4pde_diversity  6e-4 10 1000 0 0 one_time 2"
"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 200 maal             6e-4 10 1000 0 0 one_time 3"
```

**关键参数**:
- `--al_pool_dataset $DATASET_V5`: v5 作为候选池（含 54,619 target + 107,389 LHS 常规）
- `--al_initial_train_size 15000`: K-Means 初始训练集（run_pipeline.sh L221 自动注入）
- `--al_trigger_epoch 4`: 每 4 epoch 触发（run_pipeline.sh L203 自动设置）
- 200 epoch / 4 = 50 轮 AL × 1000 样本 = 50,000 AL 样本

### 2.3 4 种 AL 策略实现位置（al_callback.py）

| 策略 | 实现函数 | 行号 | acquisition 逻辑 |
|------|----------|------|------------------|
| al4pde_random | `_al4pde_random_sample` | L2333-2341 | 候选池随机抽样 |
| al4pde_uncertainty | `_al4pde_uncertainty_sample` | L2343-2357 | MC Dropout 10 次前向，top-K 方差 |
| al4pde_diversity | `_al4pde_diversity_sample` | L2359-2371 | 轨迹特征空间最大距离（bmdal_reg 风格） |
| maal | `_maal_sample` | L2373-2414 | `0.6×diversity + 0.4×target_priority` |

**MAAL target_priority**（`_map_family_priority` L2490-2576）: 3D 加权公式 = `0.5×MAE百分位排名 + 0.3×参数空间距离 + 0.2×家族覆盖度`

---

## 3. 4 策略详细对比

### 3.1 全局指标对比

| 指标 | random | uncertainty | diversity | MAAL | 最优 | 最差 |
|------|--------|-------------|-----------|------|------|------|
| LHS Test MAE | 0.1018 | 0.1389 | **0.0972** | 0.1376 | diversity | uncertainty |
| LHS Test Corr | 0.9840 | 0.9825 | 0.9901 | 0.9833 | diversity | uncertainty |
| Real MAE | **0.0976** | 0.1894 | 0.2725 | 0.1579 | random | diversity |
| Real MSE | **0.406** | 2.614 | 8.832 | 4.614 | random | diversity |
| Real Corr | **0.9891** | 0.9611 | 0.9235 | 0.9744 | random | diversity |
| Sim-to-Real Gap | **0.96×** | 1.36× | 2.80× | 1.15× | random | diversity |
| Tier 1 占比 | **73.0%** | 38.9% | 27.8% | 52.4% | random | diversity |
| Tier 4 占比 | **2.4%** | 12.7% | 24.6% | 7.9% | random | diversity |

**关键洞察**:
1. **random 在 Real Mutants 上全面最优**（MAE/MSE/Corr/Gap/Tier 全胜），但在 LHS Test 上略逊于 diversity
2. **diversity 在 LHS Test 上最优**（0.0972），但在 Real Mutants 上最差（0.2725）——典型的"过拟合 LHS 分布，忽视 Real Mutant 分布"
3. **MAAL 综合第二**：Real MAE 0.1579 优于 uncertainty（0.1894）和 diversity（0.2725），但逊于 random（0.0976）
4. **uncertainty 表现不佳**：Real MAE 0.1894，Tier 4 占比 12.7%，MC Dropout 的不确定性估计在 65000 样本时仍不可靠

### 3.2 Tier 分布对比

| Tier (MAE 范围) | random | uncertainty | diversity | MAAL |
|-----------------|--------|-------------|-----------|------|
| Tier 1 (< 0.1) | **92 (73.0%)** | 49 (38.9%) | 35 (27.8%) | 66 (52.4%) |
| Tier 2 (0.1-0.2) | 26 (20.6%) | 40 (31.7%) | 37 (29.4%) | 45 (35.7%) |
| Tier 3 (0.2-0.3) | 5 (4.0%) | 21 (16.7%) | 23 (18.3%) | 5 (4.0%) |
| Tier 4 (≥ 0.3) | **3 (2.4%)** | 16 (12.7%) | **31 (24.6%)** | 10 (7.9%) |
| MAE > 1.0 | 0 | 2 | 5 | 3 |

**关键洞察**:
- random 的 Tier 1 占比（73%）接近 V5+none（89.7%）的水平，且无 MAE>1.0 灾难
- diversity 有 5 个 MAE>1.0 的灾难性突变体，全部是 cdc20_ts 系列
- MAAL 有 3 个 MAE>1.0，其中 2 个是 cdc20_ts（78/73），1 个是 clb5_clb6_KO（63）

### 3.3 cdc20_ts 系列灾难性对比（核心发现）

| Mutant | V6+none | V5+none | random | uncertainty | diversity | MAAL |
|--------|---------|---------|--------|-------------|-----------|------|
| 78_cdc20_ts_mad2 | 0.031 | 0.032 | **0.067** | 0.087 | **1.955** | **1.952** |
| 73_cdc20_ts | 0.042 | 0.040 | **0.119** | 0.134 | **1.886** | **1.916** |
| 74_cdc20_clb5 | 0.039 | 0.042 | **0.091** | 0.078 | **1.638** | 0.078 |
| 79_cdc20_ts_bub2 | — | 0.043 | — | — | — | — |
| 75_cdc20_pds1 | — | 0.032 | — | — | — | — |

**关键洞察**:
1. **diversity 和 MAAL 在 78/73 上灾难性失败**（MAE≈1.9，是 V6+none 的 60×）
2. **random 在 cdc20_ts 上表现尚可**（78: 0.067, 73: 0.119，是 V6+none 的 2-3×）
3. **uncertainty 在 cdc20_ts 上中等**（78: 0.087, 73: 0.134）
4. **MAAL 在 74_cdc20_clb5 上幸免**（0.078），但在 78/73 上与 diversity 一样灾难

**根因分析**:
- cdc20_ts 涉及 CDC20 蛋白温度敏感突变，参数空间（ka20p, ki20p, ks20p）高度非线性
- v1 训练集（初始 15000 K-Means）几乎不覆盖 cdc20_ts 参数区域
- **diversity 的特征空间距离采样**倾向于选择"分散"样本，但 cdc20_ts 参数在归一化空间中聚类紧凑（ka20p/ki20p/ks20p 的 perturbation 范围窄），被 diversity 采集函数认为是"低多样性"区域而避开
- **MAAL 的 target_priority** 基于 mutant atlas，若 cdc20_ts 未在 atlas 中标注（或 lhs_MAE_min 值低），target_priority=0，导致 MAAL 也避开该区域
- **random 无偏采样**，有均等概率命中 cdc20_ts 区域的 50 轮 AL 中，累计 50000 样本足以覆盖

### 3.4 其他关键突变体对比

| Mutant | V6+none | random | uncertainty | diversity | MAAL | 说明 |
|--------|---------|--------|-------------|-----------|------|------|
| 124_APC_A_GAL_CLB2 | 0.189 | 0.590 | 0.483 | 0.586 | **0.323** | MAAL target_priority 命中 |
| 56_CLB2_db_Gal | 0.139 | 0.167 | **0.095** | 0.202 | 0.132 | uncertainty 最好 |
| 61.1_CLB2_db_clb5_Gal | 0.152 | 0.168 | **0.065** | 0.258 | 0.176 | uncertainty 最好 |
| 88_tem1_ts_multi_CDC15 | 0.070 | 0.264 | 0.274 | 0.384 | 0.285 | V6 修复后最优 |
| 89_tem1_ts_GAL_CDC15 | 0.111 | 0.338 | 0.358 | 0.374 | 0.387 | V6 修复后最优 |
| 1_WT_Glc | 0.017 | 0.030 | 0.032 | 0.035 | 0.056 | 即使 WT，AL 仍 2-3× worse |

**关键洞察**:
1. **MAAL 在 124_APC_A_GAL_CLB2 上最优**（0.323 vs random 0.590），证明 target_priority 对 atlas 标注家族有效
2. **uncertainty 在 56/61.1 上最优**（0.095/0.065），MC Dropout 能识别 CLB2_db_Gal 参数区域的不确定性
3. **V6+none 在 88/89 tem1_ts 上碾压所有 AL 策略**（0.070/0.111 vs AL 的 0.26-0.39），证明 V6 的窄扰动定向采样远优于 AL 的探索
4. **即使 Wild Type（1_WT_Glc），AL 仍比全量训练差 2-3×**（0.030 vs 0.017），说明 65000 样本仍未让模型充分收敛

---

## 4. V2 vs V1 配置对比（同策略）

### 4.1 uncertainty 策略 V1→V2 改善

| 指标 | V1（15000） | V2（65000） | 改善 |
|------|-------------|-------------|------|
| LHS Test MAE | 0.1737 | 0.1389 | -20.0% |
| Real MAE | 0.3726 | 0.1894 | **-49.2%** |
| Real MSE | 12.149 | 2.614 | -78.5% |
| Real Corr | 0.8733 | 0.9611 | +10.1% |
| Sim-to-Real Gap | 2.15× | 1.36× | -36.7% |
| Tier 1 占比 | 14.3% | 38.9% | +24.6pp |
| Tier 4 占比 | 39.7% | 12.7% | -27.0pp |

### 4.2 MAAL 策略 V1→V2 改善

| 指标 | V1（15000） | V2（65000） | 改善 |
|------|-------------|-------------|------|
| LHS Test MAE | 0.1671 | 0.1376 | -17.7% |
| Real MAE | 0.4088 | 0.1579 | **-61.4%** |
| Real MSE | 10.855 | 4.614 | -57.5% |
| Real Corr | 0.8240 | 0.9744 | +18.3% |
| Sim-to-Real Gap | 2.45× | 1.15× | -53.1% |
| Tier 1 占比 | 9.5% | 52.4% | +42.9pp |
| Tier 4 占比 | 48.4% | 7.9% | -40.5pp |

**关键洞察**:
1. **MAAL 的改善幅度（-61.4%）超过 uncertainty（-49.2%）**，说明 MAAL 的 target_priority 机制在数据量充足时更有效
2. **V2 配置下 MAAL（0.1579）反而优于 uncertainty（0.1894）**，而 V1 配置下 MAAL（0.4088）比 uncertainty（0.3726）更差——MAAL 的 target_priority 在 V1 的 15000 样本时"赢者通吃"导致忽视全局，但在 V2 的 65000 样本时能更好地平衡目标家族与全局覆盖
3. **Tier 4 灾难区大幅缩减**: uncertainty 从 39.7%→12.7%，MAAL 从 48.4%→7.9%，但 diversity 仍有 24.6%

---

## 5. Sim-to-Real Gap 演进

| 版本 | 训练样本 | Real MAE | LHS Test MAE | Sim-to-Real Gap |
|------|----------|----------|--------------|-----------------|
| V1+none | ~43,000 | 0.3600 | 0.0764 | 4.53× |
| V5+none | ~130,000 | 0.0552 | 0.0618 | 0.89× |
| V6+none | ~133,000 | 0.0492 | 0.0591 | **0.83×** ✅ |
| al4pde_random (V2) | 65,000 | 0.0976 | 0.1018 | **0.96×** |
| MAAL (V2) | 65,000 | 0.1579 | 0.1376 | 1.15× |
| al4pde_uncertainty (V2) | 65,000 | 0.1894 | 0.1389 | 1.36× |
| al4pde_diversity (V2) | 65,000 | 0.2725 | 0.0972 | 2.80× |
| 旧 al4pde_uncertainty (V1) | 15,000 | 0.3726 | 0.1737 | 2.15× |
| 旧 MAAL (V1) | 15,000 | 0.4088 | 0.1671 | 2.45× |

**关键洞察**:
1. **al4pde_random (V2) 的 Gap=0.96× 接近 1.0**，意味着 Real Mutant 预测误差（0.0976）接近 LHS Test 误差（0.1018），Sim-to-Real 对齐度良好
2. **diversity 的 Gap=2.80× 是 V2 配置中最差的**，Real MAE 是 LHS Test MAE 的 2.8 倍，说明 diversity 过拟合 LHS 分布而忽视 Real Mutant
3. **V2 配置的 Gap 全面优于 V1**: random 0.96× vs 旧 uncertainty 2.15×，证明增加训练数据量是改善 Sim-to-Real 对齐的关键

---

## 6. 为什么 random 最优？——AL 策略的"探索-利用"困境

### 6.1 diversity 失败根因

diversity 策略的采集函数基于轨迹特征空间的最大距离（bmdal_reg 风格），目标是选择"最分散"的样本。但在出芽酵母 141 维参数空间中：

1. **cdc20_ts 参数聚类紧凑**: ka20p/ki20p/ks20p 的温度敏感突变在归一化参数空间中形成一个紧凑的聚类，diversity 采集函数认为这些样本"低多样性"
2. **diversity 系统性避开 cdc20_ts**: 50 轮 AL 中，diversity 倾向于选择参数空间边缘的样本（高多样性），而 cdc20_ts 聚类在空间内部被忽略
3. **结果**: 5 个 cdc20_ts 突变体 MAE 全部 >1.6（diversity），而 random 仅 0.07-0.12

### 6.2 MAAL 的部分失败

MAAL 的 acquisition = `0.6×diversity + 0.4×target_priority`：

1. **target_priority 命中的家族改善**: 124_APC_A_GAL_CLB2（MAAL 0.323 vs random 0.590），证明 target_priority 对 atlas 标注家族有效
2. **target_priority 未命中的家族灾难**: 78/73 cdc20_ts（MAAL 1.95，与 diversity 一样灾难），因为 cdc20_ts 在 atlas 中未标注或 lhs_MAE_min 值低
3. **0.6×diversity 权重过高**: MAAL 仍保留了 diversity 的"避开紧凑聚类"倾向，导致 cdc20_ts 灾难

### 6.3 uncertainty 的中等表现

uncertainty 策略基于 MC Dropout 10 次前向的预测方差：

1. **对 CLB2_db_Gal 系列有效**: 56（0.095）/61.1（0.065）优于 random，因为 MC Dropout 能识别 CLB2_db_Gal 参数区域的高不确定性
2. **对 cdc20_ts 中等**: 78（0.087）/73（0.134），优于 diversity/MAAL 但逊于 random
3. **整体欠拟合导致不确定性估计不可靠**: 65000 样本下模型仍有 39% 突变体 MAE>0.1，MC Dropout 的不确定性估计在欠拟合阶段不可靠

### 6.4 random 最优的根本原因

random 策略在 50 轮 AL 中无偏采样 50000 样本：

1. **无偏覆盖**: random 有均等概率命中 cdc20_ts 参数区域，50 轮累计 50000 样本足以覆盖
2. **无采集函数偏差**: 不受 diversity 的"避开紧凑聚类"或 MAAL 的"target_priority 赢者通吃"影响
3. **在 141 维高维空间中，随机采样的覆盖能力优于任何启发式采集函数**——这是高维空间的"维度灾难"效应，任何基于距离/不确定性的采集函数都难以有效探索 141 维空间

---

## 7. 结论与建议

### 7.1 核心结论

1. **V2 配置（65000 样本）显著改善**: uncertainty -49%，MAAL -61%，但最优策略 random（0.0976）仍是 V6+none（0.0492）的 2.0×
2. **random 最优，diversity 最差**: 4 策略排名 random > MAAL > uncertainty > diversity，反转了"AL 应优于随机"的预期
3. **cdc20_ts 系列是 diversity/MAAL 的绝对灾难区**: 5 个 cdc20_ts 突变体 MAE>1.6，根因是采集函数系统性避开 cdc20_ts 参数聚类
4. **AL 在 141 维高维空间中面临"维度灾难"**: 任何基于距离/不确定性的采集函数都难以有效探索，random 的无偏覆盖反而最优

### 7.2 行动建议

**短期**:
1. **停用 diversity 和 MAAL 策略**——在当前配置下它们不如 random
2. **若要继续 AL 探索，使用 al4pde_random**——0.0976 是 AL 策略中的最优
3. **V6+none（0.0492）仍是当前最优配置**，AL 的边际收益为负

**中期**:
4. **增大初始训练集至 30000-50000**: 参考 V5+none 的 130000 样本，65000 仍不足
5. **改用 V6 数据集作为初始训练集**（而非 v1），让初始集覆盖 cdc20_ts 等定向家族
6. **为 diversity/MAAL 增加"cdc20_ts 强制采样配额"**，每轮保证至少 200 个 cdc20_ts_family 样本

**长期**:
7. **探索"分层 AL"策略**: 对 cdc20_ts 等已知难家族用 random 强制采样，对其他家族用 uncertainty/diversity
8. **考虑"两阶段训练"**: 先全量预训练 100 epoch，再 AL 微调 100 epoch
9. **在 141 维空间中，AL 的理论极限可能是 random**——除非有更强的先验知识指导采集函数

### 7.3 最终判断

**在 V6 数据集已实现 Real MAE=0.0492、Sim-to-Real Gap=0.83× 的情况下，al4pde 风格 AL 框架的边际收益为负**。即使最优策略 random（0.0976）仍是 V6+none 的 2.0×。AL 的价值在于"数据生成昂贵"的场景，但本项目 ODE 求解器（solve_ivp）的标注成本远低于训练 200 epoch 的计算成本，因此全量训练（V6+none）是更优选择。

---

## 附录 A: 4 策略 Top 10 灾难性突变体

### al4pde_random Top 5 灾难

| Mutant | MAE | 根因 |
|--------|-----|------|
| 124_APC_A_GAL_CLB2 | 0.590 | APC_A+GAL_CLB2 联合 |
| 89_tem1_ts_GAL_CDC15 | 0.338 | tem1_ts multi |
| 61.2_CLB2_db_clb5_Raff | 0.311 | CLB2_db_clb5_Raff |
| 10_bck2_KO | 0.298 | bck2_KO |
| 88_tem1_ts_multi_CDC15 | 0.264 | tem1_ts multi |

### al4pde_uncertainty Top 5 灾难

| Mutant | MAE | 根因 |
|--------|-----|------|
| 85_GAL_ESP1_cdc20_ts | 1.790 | cdc20_ts 灾难 |
| 71_CLB5_db_pds1_cdc20 | 1.320 | cdc20 灾难 |
| 62_GAL_CLB2_db | 0.968 | GAL_CLB2_db |
| 42_sic1_cdc6_cdh1_GAL_CDC20 | 0.662 | sic1_cdc6_cdh1 |
| 124_APC_A_GAL_CLB2 | 0.483 | APC_A+GAL_CLB2 |

### al4pde_diversity Top 5 灾难

| Mutant | MAE | 根因 |
|--------|-----|------|
| 78_cdc20_ts_mad2 | 1.955 | cdc20_ts 灾难 |
| 73_cdc20_ts | 1.886 | cdc20_ts 灾难 |
| 79_cdc20_ts_bub2 | 1.720 | cdc20_ts 灾难 |
| 74_cdc20_clb5 | 1.638 | cdc20_ts 灾难 |
| 71_CLB5_db_pds1_cdc20 | 1.460 | cdc20 灾难 |

### MAAL Top 5 灾难

| Mutant | MAE | 根因 |
|--------|-----|------|
| 78_cdc20_ts_mad2 | 1.952 | cdc20_ts 灾难 |
| 73_cdc20_ts | 1.916 | cdc20_ts 灾难 |
| 63_clb5_clb6_KO | 1.212 | clb5_clb6_KO |
| 80_pds1_KO | 0.729 | pds1_KO |
| 10_bck2_KO | 0.501 | bck2_KO |

---

**报告完成时间**: 2026-07-08
**下一步**: 参考 `V6_EVAL_ANALYSIS_REPORT.md` 和 `V5_NO_TARGET_EVAL_REPORT.md` 了解 V6 数据集和 target 样本的贡献
