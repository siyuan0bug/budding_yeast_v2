# 出芽酵母细胞周期项目损失函数总结

> 本文档对 [utils/losses.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py) 中注册的全部损失函数进行系统总结，重点深入 ODE 残差损失、H1 频域损失、PINO 组合损失等难点，并记录 v1/v2 演进中的关键教训。
>
> 配套代码：
> - [utils/losses.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py) — 损失函数实现
> - [utils/physics_eqns_torch.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/physics_eqns_torch.py) — ODE 方程的可微 PyTorch 实现
> - [data/lhs_v1_origin.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data/lhs_v1_origin.py) — ODE 真值引擎（numpy 版）

---

## 目录

1. [概述与损失注册表](#1-概述与损失注册表)
2. [统一接口约定](#2-统一接口约定)
3. [损失函数演进图](#3-损失函数演进图)
4. [基础损失：MSEOnlyLoss](#4-基础损失mseonlyloss)
5. [负值惩罚：MSENegPenLoss](#5-负值惩罚msenegpenloss)
6. [频谱平滑：MSESmoothLoss](#6-频谱平滑msesmoothloss)
7. [组合损失：PhysicsInformedLoss](#7-组合损失physicsinformedloss)
8. [膨胀 Sobolev：DilatedSobolevLoss](#8-膨胀-sobolevdilatedsobolevloss)
9. [PINN 残差损失：PINNResidualLoss（重点）](#9-pinn-残差损失pinnresidualloss重点)
   - 9.1 [核心思想：ODE 残差](#91-核心思想ode-残差)
   - 9.2 [f(u, p) 长什么样](#92-fu-p-长什么样)
   - 9.3 [dû/dt 的有限差分计算](#93-dûdt-的有限差分计算)
   - 9.4 [残差归一化与数值稳定性](#94-残差归一化与数值稳定性)
   - 9.5 [Event 软掩码（关键）](#95-event-软掩码关键)
   - 9.6 [跳跃条件损失 L_jump](#96-跳跃条件损失-l_jump)
   - 9.7 [Curriculum learning（课程学习）](#97-curriculum-learning课程学习)
   - 9.8 [整体前向流程](#98-整体前向流程)
10. [H1 Sobolev 损失：H1Loss（重点）](#10-h1-sobolev-损失h1loss重点)
    - 10.1 [数学定义](#101-数学定义)
    - 10.2 [频域实现与 norm='ortho'](#102-频域实现与-normortho)
    - 10.3 [为什么 k² 权重](#103-为什么-k-权重)
11. [PINO 损失：PINOLoss（重点）](#11-pino-损失pinoloss重点)
    - 11.1 [设计思想](#111-设计思想)
    - 11.2 [公式组成](#112-公式组成)
    - 11.3 [H1 频域约束的实现细节](#113-h1-频域约束的实现细节)
    - 11.4 [与 PINN/H1 的对比](#114-与-pinnh1-的对比)
12. [外部 Curriculum：get_curriculum_weights](#12-外部-curriculumget_curriculum_weights)
13. [关键参数配置参考](#13-关键参数配置参考)
14. [v1/v2 教训汇总](#14-v1v2-教训汇总)
15. [难点总结与陷阱](#15-难点总结与陷阱)

---

## 1. 概述与损失注册表

项目在 [losses.py:549-558](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L549-L558) 注册了 8 种损失函数，由简到繁演进：

| loss_type | 类 | 核心组成 | 主要用途 |
|-----------|----|----|----|
| `mse_only` | `MSEOnlyLoss` | 纯 MSE | 基线，最常用 |
| `mse_negpen` | `MSENegPenLoss` | MSE + 负值惩罚 | 防止预测负浓度 |
| `mse_smooth` | `MSESmoothLoss` | MSE + 频谱平滑 | 抑制高频噪声 |
| `physics_informed` | `PhysicsInformedLoss` | MSE + 负值惩罚 + 频谱平滑 | 早期物理约束尝试 |
| `t_smooth` | `DilatedSobolevLoss` | MSE + 膨胀跳跃检测 + 1/2 阶平滑 | 处理 event 跳变 |
| `pinn_residual` | `PINNResidualLoss` | MSE + 负值惩罚 + **ODE 残差** + 跳跃条件 | 物理约束核心 |
| `h1` | `H1Loss` | 时域 L2 + **频域梯度 L2** | Sobolev 范数约束 |
| `pino` | `PINOLoss` | PINN + **H1 频域梯度**（组合） | 最复杂损失 |

注册表代码：
```python
LOSS_REGISTRY = {
    'physics_informed': PhysicsInformedLoss,
    'mse_only': MSEOnlyLoss,
    'mse_negpen': MSENegPenLoss,
    'mse_smooth': MSESmoothLoss,
    't_smooth': DilatedSobolevLoss,
    'pinn_residual': PINNResidualLoss,
    'h1': H1Loss,
    'pino': PINOLoss,
}
```

---

## 2. 统一接口约定

所有损失函数遵循统一签名（[losses.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py) 各类的 `forward`）：

```python
def forward(self, pred_norm, target_norm, batch_weights, lambda_penalty, lambda_smooth):
    ...
    return total_loss  # 标量
```

**关键约定**：

| 参数 | 形状 | 说明 |
|------|------|------|
| `pred_norm` | `(B, 39, T)` | 归一化空间预测轨迹，T=500 |
| `target_norm` | `(B, 39, T)` | 归一化空间真值轨迹 |
| `batch_weights` | `(B,)` | 批次权重（用于 AL 加权） |
| `lambda_penalty` | 标量 | 负值惩罚权重（外部传入，可覆盖默认） |
| `lambda_smooth` | 标量 | 频谱平滑权重（外部传入） |

**PINN/PINO 特殊签名**（[L313](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L313), [L514](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L514)）：多一个 `p_norm` 参数：
```python
def forward(self, pred_norm, target_norm, p_norm, batch_weights, lambda_penalty, lambda_smooth):
```

**所有计算在归一化空间进行**（除了 ODE 残差需要反归一化到真实空间，见 [第 9 节](#9-pinn-残差损失pinnresidualloss重点)）。

---

## 3. 损失函数演进图

```
MSEOnlyLoss (基线)
   │
   ├──+ 负值惩罚 ──→ MSENegPenLoss
   │
   ├──+ 频谱平滑 ──→ MSESmoothLoss
   │
   └──+ 负值惩罚 + 频谱平滑 ──→ PhysicsInformedLoss
                                   │
                                   │ (event 处理升级)
                                   ▼
                        DilatedSobolevLoss (膨胀跳跃检测)

PINNResidualLoss (ODE 残差 + 跳跃条件) ◄── 物理约束方向
   │
   └── 继承 ──→ PINOLoss
                  ▲
                  │ (新增 H1 频域约束)
                  │
H1Loss (时域 L2 + 频域梯度 L2) ◄── Sobolev 方向
```

**两条技术路线**：
- **物理约束路线**：MSE → PINN（ODE 残差）→ PINO（+ H1 频域）
- **Sobolev 路线**：MSE → H1（时域+频域）
- **PINO = 物理约束路线 + Sobolev 路线的融合**

---

## 4. 基础损失：MSEOnlyLoss

**文件**：[losses.py:29-34](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L29-L34)

```python
class MSEOnlyLoss(nn.Module):
    def __init__(self, mean_y=None, std_y=None):
        super().__init__()

    def forward(self, pred_norm, target_norm, batch_weights, lambda_penalty, lambda_smooth):
        return F.mse_loss(pred_norm, target_norm)
```

最简单的基线，纯 MSE。`mean_y/std_y` 参数仅为接口兼容，不实际使用。

**为什么是默认基线？** 项目初期对比实验显示，复杂损失（physics_informed、t_smooth）在 Real Mutant MAE 上不如纯 MSE。MSE 在归一化空间计算，各变量方差≈1，隐式权重与大值变量（ORI, CDC20T, CLB2）的评估贡献正相关，与评估指标对齐度最高。

---

## 5. 负值惩罚：MSENegPenLoss

**文件**：[losses.py:37-47](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L37-L47)

```python
class MSENegPenLoss(nn.Module):
    def __init__(self, mean_y, std_y):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.bound_norm = -mean_y / std_y  # 归一化空间的零浓度边界

    def forward(self, pred_norm, target_norm, batch_weights, lambda_penalty, lambda_smooth):
        loss_mse = self.mse(pred_norm, target_norm).mean(dim=(1, 2))
        negative_penalty = (torch.relu(self.bound_norm - pred_norm) ** 2).mean(dim=(1, 2))
        total_loss = loss_mse + lambda_penalty * negative_penalty
        return (total_loss * batch_weights).mean()
```

**核心逻辑**：
- 浓度不可能为负，但模型可能预测负值
- `bound_norm = -mean_y / std_y`：把真实空间的 0 浓度转到归一化空间（因为 `y_norm = (y - mean_y) / std_y`，当 `y=0` 时 `y_norm = -mean_y/std_y`）
- `torch.relu(bound_norm - pred_norm)`：只有当预测值**低于**零浓度边界时才产生惩罚
- 平方惩罚确保梯度连续

**为什么需要？** ODE 方程中很多项的分母含浓度（如 `SIC1T = SIC1+C2+C5+...`），预测负值会让分母接近 0，下游 PINN 残差计算产生 NaN。

---

## 6. 频谱平滑：MSESmoothLoss

**文件**：[losses.py:50-63](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L50-L63)

```python
class MSESmoothLoss(nn.Module):
    def forward(self, pred_norm, target_norm, batch_weights, lambda_penalty, lambda_smooth):
        loss_mse = self.mse(pred_norm, target_norm).mean(dim=(1, 2))
        pred_fft = torch.fft.rfft(pred_norm, dim=-1)
        target_fft = torch.fft.rfft(target_norm, dim=-1)
        k = torch.fft.rfftfreq(pred_norm.size(-1), device=pred_norm.device)
        diff_fft = pred_fft - target_fft
        loss_spectral_smooth = torch.mean((torch.abs(diff_fft) ** 2) * (k ** 2).unsqueeze(0).unsqueeze(0), dim=(1, 2))
        total_loss = loss_mse + lambda_smooth * loss_spectral_smooth
        return (total_loss * batch_weights).mean()
```

**核心逻辑**：
- 对预测和真值做 FFT，计算频域差的能量
- `(k ** 2)` 权重：**高频分量（大 k）受更大惩罚**，等效于约束导数（因为 `d/dt ↔ iω`，导数能量 ∝ ω²）
- 这是 H1 频域梯度的雏形（但没加 `norm='ortho'`，见 [第 10.2 节](#102-频域实现与-normortho)）

**注意**：此处的频域平滑**没有 event 掩码**，在 event 跳变点会产生 Gibbs 现象（高频振荡），可能压制真实跳变。这是后来 DilatedSobolevLoss 改进的动力。

---

## 7. 组合损失：PhysicsInformedLoss

**文件**：[losses.py:11-26](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L11-L26)

```python
class PhysicsInformedLoss(nn.Module):
    def forward(self, pred_norm, target_norm, batch_weights, lambda_penalty, lambda_smooth):
        loss_mse = self.mse(pred_norm, target_norm).mean(dim=(1, 2))
        negative_penalty = (torch.relu(self.bound_norm - pred_norm) ** 2).mean(dim=(1, 2))
        pred_fft = torch.fft.rfft(pred_norm, dim=-1)
        target_fft = torch.fft.rfft(target_norm, dim=-1)
        k = torch.fft.rfftfreq(pred_norm.size(-1), device=pred_norm.device)
        diff_fft = pred_fft - target_fft
        loss_spectral_smooth = torch.mean((torch.abs(diff_fft) ** 2) * (k ** 2).unsqueeze(0).unsqueeze(0), dim=(1, 2))
        total_loss = loss_mse + (lambda_penalty * negative_penalty) + (lambda_smooth * loss_spectral_smooth)
        return (total_loss * batch_weights).mean()
```

**组成**：MSE + 负值惩罚 + 频谱平滑。是 MSENegPenLoss 和 MSESmoothLoss 的组合。

**权重**：由外部 `get_curriculum_weights`（[第 12 节](#12-外部-curriculumget_curriculum_weights)）提供 `lambda_penalty` 和 `lambda_smooth`，实现 curriculum learning（前 15 轮纯 MSE，15-60 轮线性增长到 10.0/0.5，60 轮后保持）。

---

## 8. 膨胀 Sobolev：DilatedSobolevLoss

**文件**：[losses.py:66-110](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L66-L110)

这是对 `MSESmoothLoss` 频谱平滑的改进，专门处理 event 跳变点。

**核心创新：膨胀跳跃检测**

```python
# 检测真值的跳变点（一阶差分大）
diff_target = target_norm[:, :, 1:] - target_norm[:, :, :-1]
is_jump_exact = (torch.abs(diff_target) > 0.2).float()  # 跳变掩码

# 膨胀：用卷积把跳变点扩展成窗口
is_jump_flat = is_jump_exact.reshape(B * V, 1, T_minus_1)
is_jump_dilated = F.conv1d(is_jump_flat, kernel, padding=padding)
is_jump_dilated = (is_jump_dilated > 0).float().reshape(B, V, T_minus_1)

is_smooth = 1.0 - is_jump_dilated  # 平滑区掩码
```

**损失组成**（[L108](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L108)）：
```python
total_loss = 2.0 * loss_mse + 5.0 * loss_jump + 0.5 * loss_smooth_1st + 1.0 * loss_smooth_2nd + 10.0 * negative_penalty
```

| 项 | 权重 | 作用 |
|----|------|------|
| `loss_mse` | 2.0 | 数据拟合 |
| `loss_jump` | 5.0 | 跳变点处的预测误差（高权重，强制拟合跳变） |
| `loss_smooth_1st` | 0.5 | 平滑区一阶导数误差 |
| `loss_smooth_2nd` | 1.0 | 平滑区二阶导数误差 |
| `negative_penalty` | 10.0 | 负值惩罚 |

**关键思想**：跳变点用 `is_jump_exact` 高权重约束（拟合跳变），平滑区用 `is_smooth` 约束导数平滑。这解决了频谱平滑在 event 点 Gibbs 现象的问题。

---

## 9. PINN 残差损失：PINNResidualLoss（重点）

**文件**：[losses.py:113-375](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L113-L375)

这是项目中最复杂的物理约束损失，也是 PINO 的基础。

### 9.1 核心思想：ODE 残差

出芽酵母的 39 个变量遵循 ODE 方程组：

$$\frac{du}{dt} = f(u, p)$$

- `u` = 39 个状态变量（MASS, CLN2, CLB2, ...）
- `p` = 141 维参数（速率常数等）
- `f(u, p)` = ODE 右端函数，[physics_eqns_torch.py:90](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/physics_eqns_torch.py#L90) 的 `eqns_torch`

**模型预测出轨迹 `û(t)`**。如果这条轨迹**真的符合物理规律**，代入 ODE 应满足：

$$\frac{d\hat{u}}{dt} = f(\hat{u}, p)$$

**残差**就是两者的差：

$$r(t) = \frac{d\hat{u}}{dt} - f(\hat{u}, p)$$

残差越小，预测轨迹越符合物理定律。残差损失 = `‖r(t)‖²`。

**为什么用 ODE 残差？** MSE 只约束"预测值接近真值"，但真值有限。ODE 残差额外约束"预测轨迹的演化规律符合物理方程"，这是**先验知识**，能让模型在数据稀疏区域也遵守物理规律，提升泛化能力（特别是 Sim-to-Real 迁移）。

### 9.2 f(u, p) 长什么样

`eqns_torch` 在 [physics_eqns_torch.py:90-201](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/physics_eqns_torch.py#L90-L201)，完全对齐 numpy 版 `eqns`，但用 torch 运算（可微）。

**简单例子：MASS（[L164](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/physics_eqns_torch.py#L164)）**

```python
dydt[:, 0] = kg * MASS  # 指数增长
```

其中 `kg = ln(2)/mdt`（[L120](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/physics_eqns_torch.py#L120)），`mdt` 是质量倍增时间参数。MASS 的 ODE 是简单指数增长 `dM/dt = kg·M`，event1 触发时 MASS 乘以 `mass_factor` 衰减。

**复杂例子：CLB2（[L166](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/physics_eqns_torch.py#L166)）**

```python
dydt[:, 2] = (p_dict['ksb2p'] + p_dict['ksb2pp'] * MCM1) * MASS \
           + (p_dict['kd3c1'] * C2P + p_dict['kd3f6'] * F2P) \
           + (p_dict['kdib2'] * C2 + p_dict['kdif2'] * F2) \
           - (Vdb2 + p_dict['kasb2'] * SIC1 + p_dict['kasf2'] * CDC6) * CLB2
```

涉及：
- MCM1 转录调控（通过 Goldbeter-Koshland 函数）
- SIC1/CDC6 结合/解离
- CDC20/CDH1 降解（Vdb2）
- 多个非线性项

**Goldbeter-Koshland 函数（[L33-45](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/physics_eqns_torch.py#L33-L45)）**

```python
def _gk_torch(Va, Vi, Ja, Ji):
    BB = Vi - Va + Ja * Vi + Ji * Va
    discriminant = BB ** 2 - 4 * (Vi - Va) * Ji * Va
    discriminant = torch.clamp(discriminant, min=0.0)  # 数值稳定
    return 2 * Ji * Va / (BB + torch.sqrt(discriminant) + 1e-12)
```

模拟激酶-磷酸酶的饱和动力学，是**分母可能→0 的根源**（当 `BB + sqrt(discriminant)` 接近 0 时），这是 ODE 残差数值不稳定的源头。

### 9.3 dû/dt 的有限差分计算

**文件**：[losses.py:217-221](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L217-L221)

```python
dt = self.t_max / (T - 1)  # T=500, t_max=210 → dt≈0.42 min
du_dt = torch.zeros_like(pred_real)
# 中心差分（内部点，二阶精度）
du_dt[:, :, 1:-1] = (pred_real[:, :, 2:] - pred_real[:, :, :-2]) / (2 * dt)
# 前向差分（起点，一阶精度）
du_dt[:, :, 0] = (pred_real[:, :, 1] - pred_real[:, :, 0]) / dt
# 后向差分（终点，一阶精度）
du_dt[:, :, -1] = (pred_real[:, :, -1] - pred_real[:, :, -2]) / dt
```

**为什么用有限差分而非 autograd？**

PINO 论文用频域方法计算导数（通过 FFT 的微分性质 `d/dt ↔ iω`），但 budding yeast 轨迹含 event 跳跃，FFT 导数在跳跃点产生 Gibbs 现象（高频振荡），故 `du/dt` 仍用有限差分。

[physics_eqns_torch.py:271-307](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/physics_eqns_torch.py#L271-L307) 的 `compute_ode_residual_autograd` 函数注释说明：完整 autograd 需要模型对 t 可微，但当前模型接口是 `forward(ic_time_grid, params)`，时间 t 作为输入通道拼接，不易做 autograd 对 t 求导，故用有限差分近似。

### 9.4 残差归一化与数值稳定性

**文件**：[losses.py:223-255](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L223-L255)

#### 向量化计算 f(u, p)

```python
f_u = torch.zeros_like(pred_real)
t_indices = list(range(0, T, self.residual_subsample))  # residual_subsample=10 → 每10步算1次
t_grid = torch.linspace(0, self.t_max, T, device=pred_real.device)

n_t = len(t_indices)
y_all = pred_real[:, :, t_indices]  # (B, 39, n_t)
y_all = torch.clamp(y_all, min=0.0)  # 浓度非负约束（防 ODE 分母→0）
y_all = y_all.transpose(1, 2).reshape(B * n_t, V)  # (B*n_t, 39) 展平批量+时间

# 参数扩展到 (B*n_t,)
p_dict_expanded = {}
for k, v in p_dict.items():
    if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == B:
        p_dict_expanded[k] = v.unsqueeze(1).expand(B, n_t).reshape(B * n_t)
    else:
        p_dict_expanded[k] = v

t_all = t_grid[t_indices].unsqueeze(0).expand(B, n_t).reshape(B * n_t)
f_all = eqns_torch(y_all, p_dict_expanded, t_all)  # 一次性算所有时间点的 f(u,p)
f_all = torch.nan_to_num(f_all, nan=0.0, posinf=1e4, neginf=-1e4)  # ODE 输出清洗
f_u[:, :, t_indices] = f_all.reshape(B, n_t, V).transpose(1, 2)
```

**关键优化**：
- **子采样**（`residual_subsample=10`）：500 步 → 50 步，降低 10 倍计算量
- **向量化**：把 `(B, 39, n_t)` reshape 成 `(B*n_t, 39)` 一次性调用 `eqns_torch`，比循环快得多
- **`clamp(min=0.0)`**：ODE 方程分母含聚合变量（如 `SIC1T = SIC1+C2+C5+SIC1P+C2P+C5P`），预测负值让分母→0 产生 NaN

#### 残差归一化与三重防护

```python
residual = du_dt - f_u                    # 真实空间残差 (B, 39, T)
residual_norm = residual / self.std_y     # 归一化到归一化空间（与 MSE 可比）
# 三重数值防护
residual_norm = torch.nan_to_num(residual_norm, nan=0.0, posinf=50.0, neginf=-50.0)
residual_sq = torch.clamp(residual_norm ** 2, max=100.0)  # 封顶防爆
```

**为什么要归一化？** `du_dt` 和 `f(u,p)` 都在真实值空间计算（量级 0.001~10），但 MSE 在归一化空间计算（量级≈1）。为与 MSE 相加，残差要除以 `std_y` 转到同一量级。

**三重数值防护**：
1. `clamp(min=0.0)`：输入 ODE 前防负值
2. `nan_to_num(f_all, ...)`：ODE 输出清洗（Goldbeter-Koshland 分母→0 时）
3. `nan_to_num(residual_norm, ...)` + `clamp(residual_sq, max=100.0)`：残差清洗+封顶

**为什么必须 `nan_to_num` 在 `clamp` 之前？** `torch.clamp(NaN, max=100)` 仍是 NaN！必须先用 `nan_to_num` 转为有限值。这是 [L253-255](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L253-L255) 注释强调的关键陷阱。

### 9.5 Event 软掩码（关键）

**文件**：[losses.py:257-267](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L257-L267) + [physics_eqns_torch.py:310-380](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/physics_eqns_torch.py#L310-L380)

#### 为什么需要 event mask？

出芽酵母有 4 种 terminal events（分裂时变量硬重置）：

| Event | 触发条件 | 重置动作 | 对应代码 |
|-------|---------|---------|---------|
| event1 | CLB2 下降穿过 0.3（分裂） | BUD→0, SPN→0, MASS 衰减, lte1→0.1 | [physics_eqns_torch.py:230-242](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/physics_eqns_torch.py#L230-L242) |
| event2 | CLB2+CLB5 下降穿过 0.2（S期退出） | ORI→0 | [L244-248](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/physics_eqns_torch.py#L244-L248) |
| event3 | ORI 上升穿过 1.0（DNA复制） | Vi20 激活, BUB2 激活 | [L250-257](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/physics_eqns_torch.py#L250-L257) |
| event4 | SPN 上升穿过 1.0（纺锤体通过） | Vi20→0.01, lte1→1.0, BUB2→init | [L259-266](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/physics_eqns_torch.py#L259-L266) |

**问题**：event 触发时变量瞬间跳变（如 BUD 从 1.0 突变到 0），这是**离散事件**，不满足连续 ODE。如果在这些点算残差 `du_dt - f(u,p)`，`du_dt` 会是巨大数值（突变导数），残差爆炸，梯度淹没 MSE。

#### event 检测逻辑

[physics_eqns_torch.py:327-378](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/physics_eqns_torch.py#L327-L378)：

```python
# 1. 检测 4 个 event 触发条件
d_clb2 = torch.cat([clb2[:, :1], clb2[:, 1:] - clb2[:, :-1]], dim=1)
event1 = (torch.abs(clb2 - 0.3) < event_tol) & (d_clb2 < 0)  # CLB2≈0.3 且下降
# ... event2/3/4 类似

# 2. 检测 9 个变量的异常跳变（基于 event 重置规则的典型幅度）
jump_thresholds = {
    nMASS: 0.5,    # MASS 衰减
    nCLB2: 0.3,    # CLB2 降解
    nCLB5: 0.3,    # CLB5 降解
    nORI: 0.5,     # ORI 归零
    nBUD: 0.3,     # BUD 归零
    nSPN: 0.3,     # SPN 归零
    nVi20: 0.3,    # Vi20 激活/归零
    nlte1: 0.3,    # lte1 上升
    nBUB2: 0.3,    # BUB2 激活
}
for v, thresh in jump_thresholds.items():
    d = torch.abs(torch.cat([u[:, v:v+1, 0], u[:, v, 1:] - u[:, v, :-1]], dim=1))
    abnormal = d > thresh
    all_events = all_events | abnormal

# 3. 扩展掩码窗口（event_window=4，向前后各扩 4 步）
for b in range(B):
    event_indices = np.where(all_events_np[b])[0]
    for t in event_indices:
        t_start = max(0, t - event_window)
        t_end = min(T, t + event_window + 1)
        mask[b, t_start:t_end] = False  # 标记为 event 区间
```

**返回 `mask`**：`True=连续区间`（可算残差），`False=event 区间`（需软掩码）。

**为什么扩展窗口？** ODE 仿真后做时间插值到 500 步，event 突变会被"涂抹"到小区间，不是单点跳变。窗口扩展确保涂抹区域都被屏蔽。

#### 软掩码应用

[losses.py:257-267](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L257-L267)：

```python
weight = torch.ones_like(mask, dtype=pred_real.dtype)
weight[~mask] = self.event_weight  # event 区间权重=0.1, 连续区间权重=1.0
weight = weight.unsqueeze(1).expand_as(residual_sq)  # (B, T) → (B, 39, T)

# 子采样掩码（只在 residual_subsample 步算残差）
subsample_mask = torch.zeros(T, dtype=torch.bool, device=pred_real.device)
subsample_mask[t_indices] = True
weight = weight * subsample_mask.unsqueeze(0).unsqueeze(0).float()

loss_residual = (residual_sq * weight).sum() / (weight.sum() + 1e-8)
```

**软掩码 vs 硬掩码**：硬掩码（权重=0）会完全放弃 event 区间的约束，软掩码（权重=0.1）保留弱约束，防止模型在 event 区间完全自由。

#### v1 教训（[losses.py:118-121](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L118-L121)）

v1 用**预测轨迹**检测 event mask，早期模型预测很差，mask 检测错误，导致在真正的 event 点用权重 1.0 算残差，梯度爆炸。v2 改用**真值轨迹**检测 mask（[L351-355](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L351-L355)），稳定了。

### 9.6 跳跃条件损失 L_jump

**文件**：[losses.py:270-311](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L270-L311)

ODE 残差在 event 区间被软掩码了，那 event 点怎么约束？用跳跃条件损失：

```python
def _compute_jump_loss(self, pred_real, target_real, mask):
    # 相邻点变化（归一化空间）
    d_pred = (pred_real[:, :, 1:] - pred_real[:, :, :-1]) / self.std_y
    d_target = (target_real[:, :, 1:] - target_real[:, :, :-1]) / self.std_y

    # 真值方向符号：+1 上升, -1 下降, 0 平稳
    target_dir = torch.sign(d_target)

    # 方向违反：预测变化与真值方向相反
    # 如果真值下降 (dir=-1)，预测不应上升 → viol = relu(d_pred)
    # 如果真值上升 (dir=+1)，预测不应下降 → viol = relu(-d_pred)
    viol = torch.relu(-target_dir * d_pred)
    viol_sq = viol ** 2

    # 在 event 区间内应用约束
    event_mask = (~mask[:, :-1]) | (~mask[:, 1:])  # (B, T-1)
    weight = event_mask.unsqueeze(1).expand_as(viol_sq).float()

    if weight.sum() > 0:
        loss_jump = (viol_sq * weight).sum() / (weight.sum() + 1e-8)
    else:
        loss_jump = torch.tensor(0.0, device=pred_real.device)
    return loss_jump
```

**逻辑**：不要求预测精确匹配跳跃幅度（太难），只要求预测的**变化方向**与真值一致。真值下降时预测不能上升，反之亦然。

**v1 教训**：v1 用 event 类型假设方向（如假设 event1 时 CLB2 必下降），但实际真值方向可能因参数不同而异，约束失效。v2 改用真值实际方向 `torch.sign(d_target)`。

### 9.7 Curriculum learning（课程学习）

**文件**：[losses.py:187-195](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L187-L195)

```python
def _get_physics_weight(self):
    """前 warmup_epochs 轮纯 MSE，之后线性增长到目标权重"""
    if self._current_epoch < self.warmup_epochs:  # 默认 30 轮
        return 0.0, 0.0
    progress = min(1.0, (self._current_epoch - self.warmup_epochs) / 30.0)
    return self.lambda_phys * progress, self.lambda_jump * progress
```

- **前 30 轮**：纯 MSE（模型先学会基本拟合）
- **30-60 轮**：ODE 残差 + 跳跃条件线性增长到目标权重
- **60 轮后**：所有约束满权重生效

**为什么需要 curriculum？** 早期模型预测很差，`du_dt`（有限差分）和 `f(u,p)`（含 Goldbeter-Koshland 分母）都很不可靠，残差梯度方向错误，会破坏 MSE 训练。先让模型学会基本拟合，再加物理约束。

### 9.8 整体前向流程

**文件**：[losses.py:313-375](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L313-L375)

```python
def forward(self, pred_norm, target_norm, p_norm, batch_weights, lambda_penalty, lambda_smooth):
    # 1. 数据损失 (MSE) —— 始终是主任务
    loss_mse = self.mse(pred_norm, target_norm).mean(dim=(1, 2))

    # 2. 负值惩罚
    negative_penalty = (torch.relu(self.bound_norm - pred_norm) ** 2).mean(dim=(1, 2))

    # 基础损失 = MSE + 负值惩罚
    base_loss = (loss_mse + lam_pen * negative_penalty) * batch_weights
    total_loss = base_loss.mean()

    # 3. Curriculum: warmup 期纯 MSE，之后逐步加入物理约束
    lam_phys_eff, lam_jump_eff = self._get_physics_weight()

    if lam_phys_eff > 0 or lam_jump_eff > 0:
        # 反归一化到真实空间
        pred_real = self._denormalize_y(pred_norm)
        target_real = self._denormalize_y(target_norm)
        p_dict = self._denormalize_p(p_norm)

        # 4. Event 掩码用真值检测
        with torch.no_grad():
            mask = detect_event_mask_torch(target_real, ...)

        # 5. ODE 残差损失（连续区间）
        if lam_phys_eff > 0:
            loss_residual = self._compute_ode_residual_loss(pred_real, p_dict, mask)
            total_loss = total_loss + lam_phys_eff * loss_residual

        # 6. 跳跃条件损失（event 点，方向约束）
        if lam_jump_eff > 0:
            loss_jump = self._compute_jump_loss(pred_real, target_real, mask)
            total_loss = total_loss + lam_jump_eff * loss_jump

    return total_loss
```

**整体流程图**：

```
模型预测 û(t) (B,39,500) 归一化空间
      │
      ├─ MSE + 负值惩罚 → base_loss (始终计算)
      │
      ├─ Curriculum 检查 (epoch < 30 ? 跳过物理约束)
      │
      ├─ 反归一化 → û_real (真实空间)
      │
      ├─ 有限差分 → dû/dt (真实空间)
      │
      ├─ eqns_torch(û_real, p) → f(û,p) (真实空间)
      │     └─ clamp(min=0) 防负值, nan_to_num 清洗
      │
      ├─ 残差 r = dû/dt - f(û,p)
      │
      ├─ 归一化: r_norm = r / std_y
      ├─ 数值清洗: nan_to_num + clamp(max=100)
      │
      ├─ event mask 软掩码 (真值检测)
      │     └─ 连续区间权重=1.0, event区间=0.1
      │
      ├─ 跳跃条件: 真值方向约束 (event区间)
      │
      └─ 加权求和 → L_total = L_mse + λ_pen·L_neg + λ_phys·L_phys + λ_jump·L_jump
                       (1.0)       (1.0)          (0.01)          (0.001)
```

---

## 10. H1 Sobolev 损失：H1Loss（重点）

**文件**：[losses.py:378-440](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L378-L440)

### 10.1 数学定义

H1 Sobolev 范数：

$$\|u-v\|_{H1}^2 = \alpha \cdot \|u-v\|_{L2}^2 + \beta \cdot \left\|\frac{du}{dt} - \frac{dv}{dt}\right\|_{L2}^2$$

- **时域 L2（α项）**：`MSE(pred, target)`，约束原始值
- **频域梯度 L2（β项）**：`Σ_k k²·|FFT(pred-target)_k|²`，约束导数/高频分量

来源：HyperNOs 项目的 `H1relLoss_1D`，适配为 budding yeast 时间序列场景。

### 10.2 频域实现与 norm='ortho'

```python
class H1Loss(nn.Module):
    def __init__(self, mean_y=None, std_y=None, alpha=1.0, beta=1.0):
        self.alpha = alpha  # 时域 L2 权重
        self.beta = beta    # 频域梯度 L2 权重

    def forward(self, pred_norm, target_norm, batch_weights, lambda_penalty, lambda_smooth):
        B, V, T = pred_norm.shape

        # 时域 L2 损失 (原始值 MSE)
        diff = pred_norm - target_norm  # (B, V, T)
        loss_time = diff.pow(2).mean(dim=(1, 2))  # (B,)

        # 频域梯度 L2 损失 (导数 MSE, 通过 FFT 计算)
        diff_fft = torch.fft.rfft(diff, dim=-1, norm='ortho')  # (B, V, T//2+1) complex
        k = torch.fft.rfftfreq(T, device=pred_norm.device)  # (T//2+1,) 归一化频率 [0, 0.5]
        loss_freq = (torch.abs(diff_fft) ** 2 * (k ** 2).unsqueeze(0).unsqueeze(0)).mean(dim=(1, 2))

        # H1 = α·时域L2 + β·频域梯度L2
        total_loss = self.alpha * loss_time + self.beta * loss_freq
        return (total_loss * batch_weights).mean()
```

**`norm='ortho'` 的关键作用**：

```python
diff_fft = torch.fft.rfft(diff, dim=-1, norm='ortho')  # 能量守恒
```

无 `norm` 时 `|FFT|²` 正比于 `T²`，T=500 时频域损失比 MSE 大约 500 倍，会导致梯度爆炸，频域项压制时域 MSE。`norm='ortho'` 确保 FFT 能量守恒（Parseval 定理），让频域损失与 MSE 在可比量级。

**对比 MSESmoothLoss**：[L57-61](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L57-L61) 的 `MSESmoothLoss` 没有 `norm='ortho'`，频域损失量级远大于 MSE，需要外部 `lambda_smooth` 调节。`H1Loss` 加了 `norm='ortho'`，时域和频域量级可比，`alpha=beta=1.0` 即可。

### 10.3 为什么 k² 权重

频域梯度 L2 的物理含义：

$$\left\|\frac{du}{dt} - \frac{dv}{dt}\right\|^2 \leftrightarrow \sum_k \omega^2 \cdot |\hat{u}_k - \hat{v}_k|^2 \quad \text{(Parseval 定理)}$$

因为 `d/dt ↔ iω`，导数在频域是乘以 `iω`，模² = `ω²`。所以频域梯度 L2 = `Σ_k ω²·|FFT(diff)_k|²`，**高频分量（大 ω/k）受更大惩罚**，等效于约束预测的导数误差。

`k = rfftfreq(T)` 返回归一化频率 `[0, 0.5]`，`k²` 权重让高频分量被强力约束，抑制预测轨迹的高频振荡/噪声。

---

## 11. PINO 损失：PINOLoss（重点）

**文件**：[losses.py:443-546](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L443-L546)

### 11.1 设计思想

PINO = **P**hysics-**I**nformed **N**eural **O**perator，来源 PINO 论文（Li et al., ICLR 2021）。

核心思想：把物理约束（ODE 残差）和频域约束（H1 Sobolev）一起加到数据损失上，让神经算子不仅拟合数据，还满足物理规律。

### 11.2 公式组成

$$L_{pino} = L_{data} + \lambda_{phys} \cdot L_{phys} + \lambda_{jump} \cdot L_{jump} + \lambda_{h1} \cdot L_{h1}$$

| 项 | 含义 | 来源 |
|----|------|------|
| `L_data` | MSE 数据损失（时域 L2） | 继承 PINNResidualLoss |
| `L_phys` | ODE 残差损失 `‖dû/dt - f(û,p)‖²` | 继承 PINNResidualLoss |
| `L_jump` | 跳跃条件损失（event 点方向约束） | 继承 PINNResidualLoss |
| `L_h1` | H1 频域梯度损失 `Σ_k k²·|FFT(pred-target)_k|²` | **PINO 新增** |

### 11.3 H1 频域约束的实现细节

```python
class PINOLoss(PINNResidualLoss):
    def __init__(self, ..., lambda_h1=0.1, h1_beta=1.0):
        super().__init__(...)  # 继承 PINN 的所有参数
        self.lambda_h1 = lambda_h1
        self.h1_beta = h1_beta

    def _compute_h1_freq_loss(self, pred_norm, target_norm):
        """H1 频域梯度损失 (仅频域梯度部分, 时域 L2 由 MSE 覆盖)"""
        B, V, T = pred_norm.shape
        diff = pred_norm - target_norm
        # norm='ortho' 确保 FFT 能量守恒
        diff_fft = torch.fft.rfft(diff, dim=-1, norm='ortho')
        k = torch.fft.rfftfreq(T, device=pred_norm.device)
        # 频域梯度 L2: |FFT(diff)|² × k²
        loss_freq = (torch.abs(diff_fft) ** 2 * (k ** 2).unsqueeze(0).unsqueeze(0)).mean(dim=(1, 2))
        return loss_freq.mean()

    def _get_h1_weight(self):
        """Curriculum: H1 约束与物理约束同步启动"""
        if self._current_epoch < self.warmup_epochs:
            return 0.0
        progress = min(1.0, (self._current_epoch - self.warmup_epochs) / 30.0)
        return self.lambda_h1 * progress

    def forward(self, pred_norm, target_norm, p_norm, batch_weights, ...):
        # 父类处理: MSE + 负值惩罚 + ODE残差 + 跳跃条件
        total_loss = super().forward(pred_norm, target_norm, p_norm, batch_weights, ...)

        # 新增: H1 频域约束
        lam_h1_eff = self._get_h1_weight()
        loss_h1 = torch.tensor(0.0, device=pred_norm.device)
        if lam_h1_eff > 0:
            loss_h1 = self._compute_h1_freq_loss(pred_norm, target_norm)
            total_loss = total_loss + lam_h1_eff * loss_h1

        return total_loss
```

**关键设计决策**：

1. **H1 只包含频域梯度部分**：时域 L2 已由 MSE 覆盖，`_compute_h1_freq_loss` 只算 `Σ_k k²·|FFT(diff)|²`，不重复算时域 L2（[L490](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L490) 注释说明）

2. **H1 不用 event mask**：FFT 是全局变换，无法局部屏蔽。通过较小的 `lambda_h1=0.1` 权重避免压制 event 跳跃（[L496](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L496) 注释说明）。这是 H1 与 ODE 残差的重要区别——ODE 残差用 event 软掩码，H1 不用。

3. **H1 与物理约束同步 curriculum**：都在 warmup 30 轮后线性启动，避免早期破坏训练。

4. **du/dt 仍用有限差分**：PINO 论文用频域方法计算导数，但 budding yeast 轨迹含 event 跳跃，FFT 导数在跳跃点产生 Gibbs 现象，故 `du/dt` 仍用有限差分（[L460](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L460) 注释说明）。PINO 的频域思想体现在 H1 约束上，而非导数计算。

### 11.4 与 PINN/H1 的对比

| 特性 | PINNResidualLoss | H1Loss | **PINOLoss** |
|------|------------------|--------|-------------|
| MSE 数据损失 | ✓ | ✓（时域 L2） | ✓ |
| ODE 残差约束 | ✓ | ✗ | ✓ |
| 跳跃条件约束 | ✓ | ✗ | ✓ |
| H1 频域梯度 | ✗ | ✓ | ✓ |
| Curriculum | warmup 30 轮 | 无 | warmup 30 轮（H1 同步） |
| event mask | ✓（软掩码） | ✗ | ✓（残差用，H1 不用） |
| `norm='ortho'` | N/A | ✓ | ✓ |

**PINO = PINN 的物理约束 + H1 的频域约束**，是项目中最复杂的损失函数。

---

## 12. 外部 Curriculum：get_curriculum_weights

**文件**：[losses.py:561-570](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L561-L570)

```python
def get_curriculum_weights(epoch, max_epochs=120):
    if epoch < 15:
        return 0.0, 0.0  # (lambda_penalty, lambda_smooth)
    elif epoch < 60:
        progress = (epoch - 15) / 45.0
        lam_pen = 10.0 * progress
        lam_sm = 0.5 * progress
        return lam_pen, lam_sm
    else:
        return 10.0, 0.5
```

**适用范围**：
- `PhysicsInformedLoss` / `mse_negpen` / `mse_smooth` 用此外部 curriculum
- `pinn_residual` / `pino` **不用**此函数（内部自管 `_get_physics_weight`）

**调度**：
- 前 15 轮：(0, 0) 纯 MSE
- 15-60 轮：线性增长到 (10.0, 0.5)
- 60 轮后：保持 (10.0, 0.5)

---

## 13. 关键参数配置参考

### 13.1 PINN/PINO 参数（run_pipeline.sh [L266](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/run_pipeline.sh#L266)）

```bash
# pinn_residual 和 pino 共用
--lambda_phys 0.01      # ODE 残差权重（弱正则，v1=1.0 灾难）
--lambda_jump 0.001     # 跳跃条件权重（更弱）
--lambda_penalty 1.0    # 负值惩罚权重（v1=10.0 压制 MSE）
--residual_subsample 10 # 每 10 步算 1 次残差（降算力）
--event_weight 0.1      # event 区间残差权重（软掩码）

# pino 专属（run_pipeline.sh L270）
--lambda_h1 0.1         # H1 频域约束权重
--h1_beta 1.0           # H1 频域梯度 L2 权重（时域 L2 由 MSE 覆盖）
```

### 13.2 参数含义对照

| 参数 | 默认值 | 作用 | 位置 |
|------|--------|------|------|
| `lambda_phys` | 0.01 | ODE 残差权重 | [L141](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L141) |
| `lambda_jump` | 0.001 | 跳跃条件权重 | [L142](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L142) |
| `lambda_penalty` | 1.0 | 负值惩罚权重 | [L143](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L143) |
| `lambda_h1` | 0.1 | H1 频域约束权重 | [L473](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L473) |
| `h1_beta` | 1.0 | H1 频域梯度 L2 权重 | [L474](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L474) |
| `event_window` | 4 | event 掩码窗口半径 | [L144](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L144) |
| `event_tol` | 0.08 | 事件触发条件容差 | [L145](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L145) |
| `jump_ratio` | 0.3 | 跳变检测阈值 | [L146](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L146) |
| `residual_subsample` | 1 | 残差计算时间步子采样率 | [L147](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L147) |
| `event_weight` | 0.1 | event 区间残差权重 | [L148](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L148) |
| `warmup_epochs` | 30 | 物理约束 warmup 轮数 | [L149](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L149) |

### 13.3 权重设计哲学

**物理约束是弱正则**（`lambda_phys=0.01`），主任务始终是 MSE。这源于 v1 教训：v1 用 `lambda_phys=1.0` 导致 val_loss 从 0.037 暴增到 0.930（差 25 倍），物理约束压制了数据拟合。

**权重层级**：
```
MSE (1.0) > 负值惩罚 (1.0) > ODE 残差 (0.01) > 跳跃条件 (0.001) > H1 频域 (0.1)
```
- MSE 和负值惩罚是基础（量级 1.0）
- ODE 残差是弱物理正则（0.01）
- 跳跃条件更弱（0.001，因为 event 点约束本身较弱）
- H1 频域约束中等（0.1，因为 `norm='ortho'` 后量级与 MSE 可比）

---

## 14. v1/v2 教训汇总

PINN 损失经历了 v1 → v2 的重大修复，记录在 [losses.py:113-129](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L113-L129) 的 docstring 中。

### v1 的三个致命问题

| 问题 | v1 设定 | 后果 | v2 修复 |
|------|---------|------|---------|
| ODE 残差权重过大 | `lambda_phys=1.0` | val_loss=0.930（baseline=0.037，差 25 倍），物理约束压制 MSE | `lambda_phys=0.01`（降 100 倍） |
| 负值惩罚权重过大 | `lambda_penalty=10.0` | 负值惩罚压制 MSE 主任务 | `lambda_penalty=1.0`（降 10 倍） |
| event mask 基于预测 | 用预测轨迹检测 event | 早期模型预测差，mask 错误，在真 event 点用权重 1.0 算残差，梯度爆炸 | 改用**真值轨迹**检测 mask |

### v2 的额外改进

| 改进 | v1 | v2 |
|------|----|----|
| 跳跃条件约束 | 假设 event 类型取最小违反量（约束失效） | 用真值变化方向约束 `relu(-sign(d_target)·d_pred)²` |
| 调度策略 | 无 curriculum，全程物理约束 | Curriculum learning：前 30 轮纯 MSE，之后线性增长 |
| 数值防护 | 无 | 三重防护：`clamp(min=0)` + `nan_to_num` + `clamp(max=100)` |

### 教训总结

1. **物理约束权重必须远小于 MSE**：物理约束是正则项，不是主任务。`lambda_phys=0.01` 是经验最优。
2. **event mask 必须用真值检测**：预测早期不可靠，用预测检测 mask 会导致梯度爆炸。
3. **必须 curriculum learning**：早期模型预测差，物理约束梯度方向错误，会破坏 MSE 训练。
4. **必须数值防护**：ODE 方程含 Goldbeter-Koshland 分母，预测负值或极端值会产生 NaN，传染整个网络。

---

## 15. 难点总结与陷阱

### 15.1 ODE 残差的难点

1. **f(u, p) 的数值稳定性**：Goldbeter-Koshland 函数分母可能→0，产生 NaN/Inf。必须三重防护（输入 clamp + 输出 nan_to_num + 残差 clamp）。

2. **event 跳变点处理**：离散事件不满足连续 ODE，在 event 点算残差会爆炸。必须用真值检测的 event mask 软掩码。

3. **归一化空间转换**：ODE 残差在真实空间计算（量级 0.001~10），但 MSE 在归一化空间计算（量级≈1）。必须除以 `std_y` 归一化到同一量级。

4. **计算量**：500 步 × 39 变量 × 批量，对每步算 `eqns_torch` 很慢。用 `residual_subsample=10` 子采样 + 向量化 reshape 优化。

5. **du/dt 的精度**：用有限差分（非 autograd），中心差分二阶精度。event 附近的差分不可靠，靠 mask 屏蔽。

### 15.2 频域损失的难点

1. **`norm='ortho'` 必须加**：否则 `|FFT|²` 正比于 `T²`，T=500 时频域损失比 MSE 大 500 倍，梯度爆炸。这是 `MSESmoothLoss`（无 norm）和 `H1Loss`（有 norm）的关键区别。

2. **k² 权重的物理含义**：`Σ_k k²·|FFT(diff)|²` 等效于约束导数误差（Parseval 定理），高频分量受更大惩罚。

3. **FFT 无法局部屏蔽**：event 跳变点会产生 Gibbs 现象（高频振荡），但 FFT 是全局变换无法局部 mask。通过较小 `lambda_h1` 权重避免压制真实跳变。

4. **H1 只算频域梯度**：时域 L2 已由 MSE 覆盖，H1 不重复算，避免双重计数。

### 15.3 PINO 组合的难点

1. **物理约束 vs 频域约束的兼容**：ODE 残差用 event mask，H1 不用。两者都在 event 附近有不同处理策略。

2. **du/dt 用有限差分而非频域**：PINO 论文用频域导数，但 budding yeast 的 event 跳跃让频域导数失效（Gibbs 现象）。故 du/dt 仍用有限差分，PINO 的频域思想只体现在 H1 约束上。

3. **Curriculum 同步**：ODE 残差和 H1 都在 warmup 30 轮后启动，避免早期破坏训练。

### 15.4 通用陷阱

1. **`torch.clamp(NaN, max=100)` 仍是 NaN**：必须先用 `nan_to_num` 转为有限值，再 clamp。这是 [L253-255](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py#L253-L255) 注释强调的关键陷阱。

2. **归一化统计量来源**：`mean_y/std_y/p_mean/p_std` 只用 LHS 样本计算（避免稀有 Pattern 缺席导致偏差）。`PINNResidualLoss` 必须在 `on_train_start` 用 datamodule 真实统计量重建，不能用 `__init__` 的占位实例。

3. **参数键顺序**：`PARAM_KEYS = sorted([k for k in _defaults.keys() if k != 'is_mutant_104'])`，141 维参数向量顺序必须与 `np.savez` 时完全一致，否则模型预测与 ODE 真值错位。

4. **batch_weights 与 AL**：`batch_weights` 用于 AL 加权采样，非 AL 训练时全为 1。损失计算时 `(total_loss * batch_weights).mean()` 保留这一项。

---

## 附：损失函数选择建议

| 场景 | 推荐损失 | 理由 |
|------|---------|------|
| 基线对比 | `mse_only` | 最简单，与评估指标对齐度最高 |
| 防负值 | `mse_negpen` | 浓度非负的物理约束 |
| 抑制高频噪声 | `h1` | Sobolev 约束，`norm='ortho'` 量级可控 |
| 物理约束 | `pinn_residual` | ODE 残差 + event 处理，v2 稳定 |
| 最强约束 | `pino` | PINN + H1 组合，最复杂 |
| event 跳变处理 | `t_smooth` | 膨胀跳跃检测，但权重需调 |

**当前项目状态**：`mse_only` 是默认基线，`pino`/`h1` 实验在 [run_pipeline.sh:140-141](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/run_pipeline.sh#L140-L141) 被注释（v1 效果不好，v2 修复后待重新验证）。
