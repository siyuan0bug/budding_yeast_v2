# 归一化与反归一化方案

## 一、总览

| 数据                  | 归一化算法               | 映射范围               | 统计量来源     |
| ------------------- | ------------------- | ------------------ | --------- |
| Y 轨迹（39 变量 × 500 步） | Z-score             | \[-10, 10] (clamp) | 全部 LHS 样本 |
| Params（141 维参数）     | Min-Max             | \[0, 1]            | 全部 LHS 样本 |
| 时间 t（500 步）         | 无（已是 \[0, 1]）       | \[0, 1]            | —         |
| IC 初始条件             | 取 Y\_norm\[:, :, 0] | 同 Y                | —         |

**核心原则**：

1. 每种变量和每种参数各自独立归一化，不同变量/参数之间互不干扰
2. Y 使用 Z-score：loss-评估对齐度最高（Spearman=0.48 vs log\_mm=-0.13）
3. Params 使用 Min-Max：保持 LHS 均匀性，处理固定值/离散参数更好

***

## 二、Y 轨迹 Z-score 归一化

### 2.1 归一化公式

```
mean_y = Y[lhs].mean(dim=(0, 2), keepdim=True)    # (1, 39, 1)
std_y  = Y[lhs].std(dim=(0, 2), keepdim=True) + 1e-6  # (1, 39, 1)
Y_norm = (Y - mean_y) / std_y
Y_norm = nan_to_num(Y_norm, nan=0.0, posinf=10.0, neginf=-10.0)
```

每个变量独立计算 mean 和 std，跨样本+时间维度聚合。

### 2.2 为什么 Y 用 Z-score 而不是 minmax/log\_mm

经过三种方案的对比实验：

| 指标                 | Z-score (最优) | Params mm + Y log\_mm | Params mm + Y 混合 |
| ------------------ | :----------: | :-------------------: | :--------------: |
| all\_test Rel L2   |   **0.124**  |         0.216         |       0.217      |
| all\_test Corr     |   **0.980**  |         0.950         |       0.946      |
| all\_test MAE      |   **0.083**  |         0.140         |       0.142      |
| real\_mutants Corr |   **0.834**  |         0.771         |       0.721      |

**根本原因：Loss-评估对齐**

训练 loss 在归一化空间计算，评估指标在原始空间计算。两者的对齐程度决定模型优化方向是否与评估目标一致。

1. **Z-score 的隐式权重与评估权重正相关** (Spearman=0.48)
   - zscore 后每个变量方差≈1，MSE loss 中各变量贡献取决于原始方差 σ²
   - 原始方差大的变量（ORI, CDC20T, CLB2, SIC1）在 loss 中权重也大
   - 这与 MAE/MSE 评估中大值变量的贡献方向一致
2. **log\_mm 的隐式权重与评估权重负相关** (Spearman=-0.13)
   - log\_mm 放大小值变量、压缩大值变量
   - 模型花大量精力优化 F5P 等小值变量（相对误差小但绝对误差微不足道）
   - 忽略 ORI 等大值变量（绝对误差才是评估重点）
3. **minmax 的极端放大问题** (Spearman=0.36)
   - 对值域跨度大的变量（如 ORI: y\_max-y\_min≈8990），归一化空间 0.01 误差放大到 89.9
   - 比 zscore 的 6.8 大 13 倍，导致模型过度关注极端变量

### 2.3 反归一化公式

```python
def denorm_y(Y_norm, norm_routing, y_min, y_max, y_log_min, y_log_max, y_mu, y_sigma, eps=1e-8,
             mean_y=None, std_y=None):
    return Y_norm * std_y + mean_y
```

### 2.4 负值惩罚

Z-score 空间中，原始 Y=0 对应 zscore = -mean\_y / std\_y。负值惩罚阈值设为此值：

```python
bound = -mean_y / std_y  # 原始空间 Y=0 对应的 zscore 值
negative_penalty = relu(bound - pred_norm) ** 2
```

***

## 三、Params 归一化（Min-Max）

### 3.1 为什么选 Min-Max

1. **LHS 采样是均匀设计**：Min-Max 是线性变换，保持均匀性不变；Z-score 会扭曲均匀分布
2. **尖峰 + 均匀展布分布**：大部分样本参数固定（尖峰），少数被 LHS 采样（均匀展布），Z-score 的 σ 被尖峰压缩导致不稳定
3. **有界输出**：映射到 \[0, 1]，Z-score 无界
4. **离散参数友好**：Bool 和低离散参数映射到固定点

### 3.2 归一化公式

```
p_min = Params[lhs].min(dim=0, keepdim=True)   # 每参数独立
p_max = Params[lhs].max(dim=0, keepdim=True)
P_norm = (Params - p_min) / (p_max - p_min + 1e-8)
P_norm = clamp(P_norm, 0.0, 1.0)
```

### 3.3 反归一化公式

```
Params = P_norm * (p_max - p_min) + p_min
```

### 3.4 特殊参数的归一化行为

#### 固定值参数（91 个，占 65%）

归一化后映射到 0.0，正确（这些参数对所有样本提供相同信息）

#### 二值参数（3 个，仅有 2 个唯一取值）

归一化后均映射到 {0.0, 1.0}

| idx | 参数名          | 原始取值             | 说明                 |
| --- | ------------ | ---------------- | ------------------ |
| 5   | `ESP1T`      | {1.0, 3.0}       | ESP1 总蛋白浓度，突变时 1→3 |
| 46  | `init_CDH1`  | {0.0, 0.9304992} | CDH1 初始浓度，突变时置 0   |
| 47  | `init_CDH1T` | {0.0, 1.0}       | CDH1T 初始浓度，突变时置 0  |

注：这些参数在数据类型上都是 float，但因只有 2 个唯一取值，Min-Max 归一化后恰好映射到 {0.0, 1.0}。

#### 低离散参数（1 个，3-10 个唯一取值）

| idx | 参数名   | 原始取值                 | 归一化后              |
| --- | ----- | -------------------- | ----------------- |
| 140 | `mdt` | {90.0, 150.0, 160.0} | {0.0, 0.857, 1.0} |

***

## 四、模型输入构造

```
IC = Y_norm[:, :, 0:1].expand(-1, -1, 500)   # 初始条件，广播到 500 步
X = stack([IC, t_grid], dim=2)                 # shape: (N, 39, 2, 500)
```

X 的第 2 维（dim=2）有两个通道：

- `X[:, :, 0, :]` = IC（归一化后的初始浓度）
- `X[:, :, 1, :]` = t\_grid（0→1 的时间网格）

条件输入：`P_norm`，shape `(N, 141)`

***

## 五、Real Mutants 的归一化

Real Mutants 使用**与 LHS 完全相同的统计量**做归一化，不参与统计量计算，避免信息泄露。

***

## 六、代码中的关键位置

| 文件                           | 功能                      | 关键变量                                |
| ---------------------------- | ----------------------- | ----------------------------------- |
| `data/yeast_datamodule.py`   | 计算统计量 + 归一化 + denorm\_y | `mean_y`, `std_y`, `p_min`, `p_max` |
| `models/yeast_lit_module.py` | 传递统计量给 loss             | `on_train_start` 中赋值                |
| `utils/losses.py`            | 负值惩罚阈值                  | `bound = -mean_y / std_y`           |
| `train.py`                   | 画图时反归一化                 | 调用 `denorm_y(mean_y=, std_y=)`      |
| `eval.py`                    | 评估时反归一化                 | 调用 `denorm_y(mean_y=, std_y=)`      |
| `utils/al_callback.py`       | AL 新增样本参数归一化 + Y 反归一化   | Params Min-Max 归一化，Y Z-score 反归一化   |

***

## 七、方案演进记录

### v1: 全 Z-score（原始方案）

- Y 和 Params 都用 Z-score
- 效果最好（all\_test MAE 0.083, real\_mutants Corr 0.834）

### v2: Params Min-Max + Y 混合 minmax/log\_mm

- Params 改为 Min-Max（保持 LHS 均匀性）
- Y 按变量特性选择 minmax 或 log\_mm
- 效果下降（all\_test MAE 0.142, real\_mutants Corr 0.721）
- 原因：log\_mm 的 loss-评估对齐差，优化方向偏离评估目标

### v3: Params Min-Max + Y Z-score（当前方案）

- Params 使用 Min-Max（保持 LHS 均匀性，处理固定值/离散参数更好）
- Y 使用 Z-score（loss-评估对齐度最高）
- 代码已实现，验证通过：
  - 固定值参数（91个）归一化后为 0.0
  - 二值参数（3个：ESP1T, init\_CDH1, init\_CDH1T）归一化后为 {0.0, 1.0}
  - 低离散参数（1个：mdt）归一化后为 {0.0, 0.857, 1.0}
  - Params\_norm 范围 \[0.0, 1.0]，均值 0.149
  - AL 采样时参数已正确归一化后输入模型

