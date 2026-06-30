# PINN Residual 损失函数详解 (v2)

本项目实现了 **PINN (Physics-Informed Neural Network) 残差损失 v2**，将 ODE 物理约束作为软正则化项加入训练，引导模型学习符合物理定律的动力学轨迹。

## 目录

- [1. v2 重写动机](#1-v2-重写动机)
- [2. 核心思想](#2-核心思想)
- [3. 损失组成](#3-损失组成)
- [4. ODE 残差损失 L_residual](#4-ode-残差损失-l_residual)
- [5. 跳跃条件损失 L_jump (v2)](#5-跳跃条件损失-l_jump-v2)
- [6. 负值惩罚 L_neg](#6-负值惩罚-l_neg)
- [7. Event 掩码检测 (v2: 基于真值)](#7-event-掩码检测-v2-基于真值)
- [8. Curriculum Learning](#8-curriculum-learning)
- [9. 权重配置与调参](#9-权重配置与调参)
- [10. v1 → v2 改进对比](#10-v1--v2-改进对比)
- [11. 与其他损失函数对比](#11-与其他损失函数对比)
- [12. 完整训练流程示例](#12-完整训练流程示例)

---

## 1. v2 重写动机

**v1 实验结果**：pinn_residual 全面差于 mse_only baseline。

| 指标 | mse_only | pinn_residual v1 | 差距 |
|-----|----------|-------------------|------|
| all_test RelL2 | 0.128 | 0.234 | +83% |
| all_test MAE | 0.083 | 0.223 | +168% |
| lhs_test RelL2 | 0.097 | 0.221 | +128% |
| real_mutants RelL2 | 0.428 | 0.516 | +20% |

即使 `lambda_phys=0.01` 已经很低，v1 仍然全面恶化。根因是**四个根本性设计缺陷**：

1. **Event 掩码基于预测而非真值**（最致命）：早期模型预测差，掩码检测出错，残差在错误位置施加约束，梯度爆炸破坏特征学习
2. **负值惩罚权重过大**：`lambda_penalty=10.0` 远大于 MSE(1.0)，压制主任务
3. **跳跃条件方向约束失效**：假设 event 类型取最小违反量，预测方向错误时可能恰好满足某条规则
4. **物理约束与数据拟合冲突**：ODE 在 event 点不成立，即使掩码处理，边界处残差仍异常

---

## 2. 核心思想

传统 MSE 损失只约束"预测接近标签"，但不保证预测轨迹**符合物理定律**。PINN 残差损失额外加入：

> 如果模型预测轨迹 û(t) 代入 ODE 方程 `du/dt = f(u, p)` 后残差很小，说明预测**满足物理约束**。

**v2 关键改进**：
- Event 掩码用**真值**检测（v1 用预测）
- 负值惩罚权重降低（v1: 10.0 → v2: 1.0）
- 跳跃条件用**真值方向**约束（v1 假设 event 类型）
- Curriculum learning：前 30 轮纯 MSE，之后逐步加入物理约束

---

## 3. 损失组成

**总损失公式**（[losses.py:forward](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/losses.py)）：

```
L_total = L_data(MSE) + λ_pen · L_neg + λ_phys_eff · L_residual + λ_jump_eff · L_jump
```

其中 `λ_phys_eff` 和 `λ_jump_eff` 由 Curriculum learning 调度（见第 8 节）。

| 分量 | 含义 | 默认权重 | 量级 |
|------|------|---------|------|
| `L_data` | MSE 数据损失 | 1.0 | 归一化空间 |
| `L_neg` | 负值惩罚 | `λ_pen=1.0` (v2) | 归一化空间 |
| `L_residual` | ODE 残差损失 | `λ_phys=0.01` | 归一化空间 |
| `L_jump` | 跳跃条件损失 | `λ_jump=0.001` | 归一化空间 |

**实现**（[losses.py:forward](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/losses.py)）：
```python
# 1. 数据损失 (MSE) —— 始终是主任务
loss_mse = self.mse(pred_norm, target_norm).mean(dim=(1, 2))

# 2. 负值惩罚（v2 权重降低到 1.0）
negative_penalty = (torch.relu(self.bound_norm - pred_norm) ** 2).mean(dim=(1, 2))

# 基础损失 = MSE + 负值惩罚
base_loss = (loss_mse + lam_pen * negative_penalty) * batch_weights
total_loss = base_loss.mean()

# 3. Curriculum learning: warmup 期纯 MSE，之后逐步加入物理约束
lam_phys_eff, lam_jump_eff = self._get_physics_weight()

if lam_phys_eff > 0 or lam_jump_eff > 0:
    # 4. Event 掩码用**真值**检测（v1 用预测，早期掩码错误）
    mask = detect_event_mask_torch(target_real, ...)

    # 5. ODE 残差损失（连续区间）
    loss_residual = self._compute_ode_residual_loss(pred_real, p_dict, mask)

    # 6. 跳跃条件损失（event 点，用真值方向约束）
    loss_jump = self._compute_jump_loss(pred_real, target_real, mask)

    total_loss += lam_phys_eff * loss_residual + lam_jump_eff * loss_jump
```

---

## 4. ODE 残差损失 L_residual

### 4.1 数学定义

```
r(t) = dû/dt - f(û(t), p)           # ODE 残差（真实空间）
r_norm(t) = r(t) / std_y             # 归一化到与 MSE 同量级
L_residual = mean_t( ||r_norm(t)||² ) · weight(t)
```

### 4.2 计算步骤

**Step 1: 中心差分计算 dû/dt**：
```python
du_dt = torch.zeros_like(pred_real)
du_dt[:, :, 1:-1] = (pred_real[:, :, 2:] - pred_real[:, :, :-2]) / (2 * dt)
du_dt[:, :, 0] = (pred_real[:, :, 1] - pred_real[:, :, 0]) / dt
du_dt[:, :, -1] = (pred_real[:, :, -1] - pred_real[:, :, -2]) / dt
```
- `dt = t_max / (T - 1) = 210 / 499 ≈ 0.42 min`

**Step 2: 向量化计算 f(u, p)**：
```python
# 将 (B, 39, T) -> (B*T, 39) 一次性调用 ODE 右端项
y_all = pred_real[:, :, t_indices].transpose(1, 2).reshape(B * n_t, V)
f_all = eqns_torch(y_all, p_dict_expanded, t_all)  # (B*T, 39)
f_u[:, :, t_indices] = f_all.reshape(B, n_t, V).transpose(1, 2)
```
- `residual_subsample=10`：每 10 步采样一次，降低计算量

**Step 3: 归一化残差**：
```python
residual_norm = residual / self.std_y  # 归一化到与 MSE 同量级
residual_sq = residual_norm ** 2
```

**Step 4: 软掩码加权**：
```python
weight = torch.ones_like(mask, dtype=pred_real.dtype)
weight[~mask] = self.event_weight  # event 区间权重=0.1, 连续区间=1.0
weight = weight.unsqueeze(1).expand_as(residual_sq)
loss_residual = (residual_sq * weight).sum() / (weight.sum() + 1e-8)
```

### 4.3 为什么归一化到 std_y 空间

如果残差在真实空间计算，量级远大于 MSE，模型会为了最小化残差而预测**ODE 平衡点附近的常数轨迹**（残差最小），导致 MSE 无法收敛。

归一化到 `std_y` 空间后，残差与 MSE 同量级，物理约束作为弱正则化，不会压制主任务。

---

## 5. 跳跃条件损失 L_jump (v2)

### 5.1 v2 改进

**v1 问题**：假设 event 类型，对 4 种 event 规则取最小违反量。预测方向错误时可能恰好满足某条规则，约束失效。

**v2 改进**：用**真值方向**约束预测。在 event 区间，如果真值下降，预测不应上升；如果真值上升，预测不应下降。

### 5.2 数学定义

```
d_pred = (û(t+1) - û(t)) / std_y       # 预测变化（归一化）
d_target = (y(t+1) - y(t)) / std_y      # 真值变化（归一化）
target_dir = sign(d_target)              # 真值方向: +1上升, -1下降, 0平稳
viol = relu(-target_dir · d_pred)        # 方向违反: 预测与真值方向相反
L_jump = mean_event(viol²)              # 在 event 区间内取平均
```

### 5.3 计算步骤

```python
# 计算相邻点变化（归一化空间）
d_pred = (pred_real[:, :, 1:] - pred_real[:, :, :-1]) / self.std_y
d_target = (target_real[:, :, 1:] - target_real[:, :, :-1]) / self.std_y

# 真值方向符号
target_dir = torch.sign(d_target)

# 方向违反：预测变化与真值方向相反
# 如果真值下降 (dir=-1)，预测不应上升 → viol = relu(d_pred)
# 如果真值上升 (dir=+1)，预测不应下降 → viol = relu(-d_pred)
viol = torch.relu(-target_dir * d_pred)
viol_sq = viol ** 2

# 在 event 区间内应用约束
event_mask = (~mask[:, :-1]) | (~mask[:, 1:])
weight = event_mask.unsqueeze(1).expand_as(viol_sq).float()
loss_jump = (viol_sq * weight).sum() / (weight.sum() + 1e-8)
```

### 5.4 v2 优势

1. **不需要假设 event 类型**：直接用真值方向，避免 v1 的"最小违反量"假设
2. **约束更严格**：预测必须与真值同向，而非"至少满足某条规则"
3. **物理意义明确**：event 点的状态重置有明确方向，真值方向就是正确方向

---

## 6. 负值惩罚 L_neg

**目的**：惩罚预测出现负值（生物浓度不能为负）。

**实现**：
```python
# bound_norm = -mean_y / std_y  (归一化空间下的 0 值)
negative_penalty = (torch.relu(self.bound_norm - pred_norm) ** 2).mean(dim=(1, 2))
```

- `bound_norm`：归一化空间下的 0 值（因为 `(0 - mean_y) / std_y = -mean_y / std_y`）
- `torch.relu(bound_norm - pred_norm)`：预测低于 0 的部分
- 平方惩罚：强化对大幅负值的惩罚

**v2 权重降低**：`lambda_penalty` 从 v1 的 10.0 降到 v2 的 1.0，避免负值惩罚压制 MSE 主任务。

---

## 7. Event 掩码检测 (v2: 基于真值)

### 7.1 v2 关键改进

**v1 问题**：掩码基于**预测**检测，早期模型预测差导致掩码错误：
- 真值 event 点没被屏蔽 → 残差巨大，梯度爆炸
- 非 event 点被误屏蔽 → 残差信息丢失

**v2 修复**：掩码基于**真值**检测，掩码始终正确：
```python
# v1 (错误): mask = detect_event_mask_torch(pred_real, ...)
# v2 (正确): mask = detect_event_mask_torch(target_real, ...)
```

### 7.2 双重检测策略

**实现**（[physics_eqns_torch.py:detect_event_mask_torch](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/physics_eqns_torch.py)）：

1. **事件条件检测**：检查 4 个 event 的触发条件是否接近满足
   - event1: `|CLB2 - 0.3| < tol` 且 CLB2 下降 → 分裂
   - event2: `|CLB2+CLB5 - 0.2| < tol` 且下降 → S期退出
   - event3: `|ORI - 1.0| < tol` 且上升 → DNA复制
   - event4: `|SPN - 1.0| < tol` 且上升 → 纺锤体通过

2. **硬跳变检测**：受 event 影响变量的相邻点差分 > 阈值
   - MASS: > 0.5, CLB2/CLB5: > 0.3, ORI: > 0.5, BUD/SPN: > 0.3
   - Vi20/lte1/BUB2: > 0.3

### 7.3 窗口扩展

检测到 event 点后，在 ±`event_window`（默认 4 步 ≈ 1.7 min）窗口内屏蔽残差，因为线性插值会把突变"涂抹"到附近几个时间步。

**输出**：
- `mask=True`：连续区间，计算 ODE 残差
- `mask=False`：event 区间，计算跳跃条件

---

## 8. Curriculum Learning

### 8.1 动机

物理约束在模型预测较差时（早期训练）不仅无益，反而有害：
- 早期预测差 → ODE 残差大且无意义 → 梯度方向错误
- 早期 event 掩码（即使基于真值）与预测不匹配 → 约束失效

### 8.2 实现

**v2 Curriculum 调度**（[losses.py:_get_physics_weight](file:///home/users/zsy/trae_workspace/budding_yeast_v2/utils/losses.py)）：
```python
def _get_physics_weight(self):
    """前 warmup_epochs 轮纯 MSE，之后线性增长到目标权重"""
    if self._current_epoch < self.warmup_epochs:  # 默认 30
        return 0.0, 0.0
    progress = min(1.0, (self._current_epoch - self.warmup_epochs) / 30.0)
    return self.lambda_phys * progress, self.lambda_jump * progress
```

### 8.3 权重调度时间线

| 训练阶段 | epoch | λ_phys_eff | λ_jump_eff | 策略 |
|---------|-------|-----------|-----------|------|
| Warmup | 0-30 | 0 | 0 | 纯 MSE，让模型学会基本轨迹 |
| Ramp | 30-60 | 0→0.01 | 0→0.001 | 线性增加物理约束 |
| Full | 60+ | 0.01 | 0.001 | 完整物理约束 |

### 8.4 lit_module 集成

**[yeast_lit_module.py:training_step](file:///home/users/zsy/trae_workspace/budding_yeast_v2/models/yeast_lit_module.py)**：
```python
if self.hparams.loss_type == 'pinn_residual':
    # Curriculum learning 由 loss 类内部管理
    self.criterion.set_epoch(self.current_epoch)
    loss = self.criterion(out_full, y, p, weights, lam_pen, lam_sm)
```

每个 epoch 开始时调用 `set_epoch`，loss 类内部根据当前 epoch 计算有效物理权重。

---

## 9. 权重配置与调参

### 9.1 命令行参数

**run_pipeline.sh 配置**（[run_pipeline.sh](file:///home/users/zsy/trae_workspace/budding_yeast_v2/run_pipeline.sh)）：
```bash
if [ "$LOSS_TYPE" = "pinn_residual" ]; then
    PINN_PARAMS="--lambda_phys 0.01 --lambda_jump 0.001 --lambda_penalty 1.0 --residual_subsample 10 --event_weight 0.1"
fi
```

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--lambda_phys` | 0.01 | ODE 残差权重（弱正则） |
| `--lambda_jump` | 0.001 | 跳跃条件权重（弱正则） |
| `--lambda_penalty` | 1.0 (v2) | 负值惩罚权重（v1=10.0） |
| `--residual_subsample` | 10 | 残差计算的时间步子采样率 |
| `--event_weight` | 0.1 | event 区间的残差权重（软掩码） |

### 9.2 权重选择建议

| 场景 | λ_phys | λ_jump | λ_pen | 说明 |
|------|--------|--------|-------|------|
| 纯 MSE 基线 | 0 | 0 | 0 | 等价于 `mse_only` |
| v2 推荐 | 0.01 | 0.001 | 1.0 | 物理约束作辅助，MSE 主导 |
| 中等物理约束 | 0.1 | 0.01 | 1.0 | 物理约束明显，需监控 MSE |
| 强物理约束（不推荐） | 1.0 | 0.1 | 1.0 | 物理压制 MSE，模型预测平衡点常数 |

### 9.3 关键调参经验

**λ_phys 过大的危害**：
- `λ_phys=1.0`：val_loss=0.930（baseline=0.037，差 25 倍）
- 模型为最小化残差，预测**常数轨迹**（dû/dt=0，残差=-f(u,p)，在平衡点 f(u,p)=0，残差=0）

**λ_penalty 过大的危害**（v2 修复）：
- `λ_penalty=10.0`：负值惩罚压制 MSE，模型过度关注避免负值而牺牲精度
- `λ_penalty=1.0`：负值惩罚与 MSE 同量级，作弱正则

---

## 10. v1 → v2 改进对比

| 维度 | v1 | v2 |
|-----|----|----|
| Event 掩码 | 基于预测 `detect_event_mask(pred)` | 基于真值 `detect_event_mask(target)` |
| 负值惩罚权重 | 10.0 | 1.0 |
| 跳跃条件 | 假设 event 类型，取最小违反量 | 用真值方向约束 |
| Curriculum | lit_module 临时调整 lambda | loss 类内部 set_epoch + _get_physics_weight |
| Warmup | 20% epochs (fraction) | 30 epochs (absolute) + 30 epochs ramp |
| 实验结果 | RelL2 0.234 (差于 baseline 0.128) | 待验证 |

---

## 11. 与其他损失函数对比

| 损失类型 | 物理约束 | 需要标签 | 计算成本 | 适用场景 |
|---------|---------|---------|---------|---------|
| `mse_only` | 无 | 是 | 1× | 基线 |
| `mse_negpen` | 负值惩罚 | 是 | 1× | 避免负值 |
| `mse_smooth` | 频谱平滑 | 是 | 2× | 平滑轨迹 |
| `t_smooth` | 跳跃+平滑 | 是 | 3× | 精细 event 处理 |
| `physics_informed` | 频谱物理 | 是 | 2× | 频域约束 |
| **`pinn_residual` v2** | **ODE 残差 + 真值方向跳跃** | **是** | **5-10×** | **强物理约束** |

**注**：`pinn_residual` 仍需要标签（MSE 项），但物理约束项（L_residual + L_jump）**不需要标签**，可作为半监督学习的基础（未来扩展方向）。

---

## 12. 完整训练流程示例

### 12.1 数据流

```
DataModule
  ├─ train_dataset_lhs: (X, P, Y) 归一化
  │    ├─ X: (B, 39, 2, T) IC + time grid
  │    ├─ P: (B, 141) 归一化参数
  │    └─ Y: (B, 39, T) 归一化轨迹标签
  ├─ mean_y, std_y: (1, 39, 1) 状态归一化统计量
  └─ p_mean, p_std: (1, 141) 参数归一化统计量
         │
         ▼
PINNResidualLoss.__init__(mean_y, std_y, p_mean, p_std, ...)
  ├─ register_buffer('mean_y', mean_y.reshape(1, 39, 1))
  ├─ register_buffer('std_y', std_y.reshape(1, 39, 1))
  ├─ register_buffer('p_mean', p_mean.reshape(1, 141))
  ├─ register_buffer('p_std', p_std.reshape(1, 141))
  └─ register_buffer('bound_norm', -mean_y / std_y)  # 归一化空间下的 0 值
```

### 12.2 单次训练步骤

```python
# yeast_lit_module.py: training_step
def training_step(self, batch, batch_idx):
    x_ic_time, p, y, weights = batch  # 归一化数据

    # 1. 模型前向
    out_full = self.forward_ic_time(x_ic_time, p)  # (B, 39, T) 归一化预测

    # 2. 设置当前 epoch（Curriculum learning）
    if self.hparams.loss_type == 'pinn_residual':
        self.criterion.set_epoch(self.current_epoch)

    # 3. 计算损失
    loss = self.criterion(out_full, y, p, weights, lam_pen, lam_sm)

    # 4. 反向传播
    self.log('train_loss', loss)
    return loss
```

### 12.3 损失前向传播详解

```python
# PINNResidualLoss.forward
def forward(self, pred_norm, target_norm, p_norm, batch_weights, ...):
    # ===== 阶段 1: 基础损失（始终计算）=====

    # 1.1 MSE 数据损失（归一化空间）
    loss_mse = self.mse(pred_norm, target_norm).mean(dim=(1, 2))  # (B,)

    # 1.2 负值惩罚（归一化空间）
    # bound_norm = -mean_y / std_y，是归一化空间下的 0 值
    negative_penalty = (torch.relu(self.bound_norm - pred_norm) ** 2).mean(dim=(1, 2))

    # 1.3 基础损失
    base_loss = (loss_mse + lam_pen * negative_penalty) * batch_weights
    total_loss = base_loss.mean()

    # ===== 阶段 2: 物理约束（Curriculum 控制开启）=====

    lam_phys_eff, lam_jump_eff = self._get_physics_weight()

    if lam_phys_eff > 0 or lam_jump_eff > 0:
        # 2.1 反归一化到真实空间（物理方程在真实空间成立）
        pred_real = pred_norm * self.std_y + self.mean_y    # (B, 39, T)
        target_real = target_norm * self.std_y + self.mean_y
        p_real = p_norm * self.p_std + self.p_mean          # (B, 141)
        p_dict = params_tensor_to_dict(p_real, extra)       # 参数字典

        # 2.2 Event 掩码（基于真值，无梯度）
        with torch.no_grad():
            mask = detect_event_mask_torch(target_real, ...)
            # mask=True: 连续区间（ODE 成立）
            # mask=False: event 区间（ODE 不成立）

        # 2.3 ODE 残差损失（连续区间）
        if lam_phys_eff > 0:
            loss_residual = self._compute_ode_residual_loss(pred_real, p_dict, mask)
            total_loss += lam_phys_eff * loss_residual

        # 2.4 跳跃条件损失（event 区间，真值方向约束）
        if lam_jump_eff > 0:
            loss_jump = self._compute_jump_loss(pred_real, target_real, mask)
            total_loss += lam_jump_eff * loss_jump

    return total_loss
```

### 12.4 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 物理方程在哪个空间计算 | 真实空间 | ODE 方程 `du/dt = f(u, p)` 在真实空间成立 |
| 残差归一化到哪个空间 | 归一化空间（除以 std_y） | 与 MSE 同量级，避免物理损失压制 MSE |
| Event 掩码基于什么 | 真值 | 早期预测差，基于预测的掩码会错误 |
| 跳跃条件约束什么 | 真值方向 | 不需要假设 event 类型，约束更严格 |
| 物理约束何时开启 | epoch ≥ 30 | Curriculum learning，早期纯 MSE |
| 物理约束权重多大 | λ_phys=0.01, λ_jump=0.001 | 弱正则，物理约束作辅助而非主导 |

### 12.5 训练日志监控

损失类内部记录分量（`_last_loss_components`）：

| 日志键 | 含义 | 监控要点 |
|--------|------|---------|
| `mse` | MSE 损失 | 应稳步下降，不应被物理损失压制 |
| `neg_pen` | 负值惩罚 | 应保持低值，过大说明预测出现负值 |
| `phys_weight` | 当前 ODE 残差权重 | warmup 期为 0，之后线性增长到 0.01 |
| `jump_weight` | 当前跳跃条件权重 | warmup 期为 0，之后线性增长到 0.001 |

**异常信号**：
- `mse` 在 warmup 后突然上升 → 物理约束权重过大，降低 `lambda_phys`
- `neg_pen` 持续高 → 模型预测负值，检查 `lambda_penalty` 是否足够
- `phys_weight` 始终为 0 → 检查 `warmup_epochs` 配置是否正确
