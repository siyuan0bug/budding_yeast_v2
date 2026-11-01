> **2026-10-31 AL物理响应评分**：新增`ours_sensitivity`，用当前canonical 39维ODE的Torch适配/JVP检查合法突变方向上的模型响应；整数拷贝用有限割线，屏蔽事件附近。它是采集评分，不加入训练loss或修改solver。[实现、公式及验证](AL_MECHANISM_IMPLEMENTATION_2026-10-31.md)。

# 损失函数库与生物物理约束规范 (Loss Functions & Physics Constraints)

> **2026-10-21 无pattern目标处理**：Fission原始pattern=None不再默认编码成B，内部ID=-1。
> HybridBifurcate/HybridAll的分类交叉熵只用有效目标；全批无目标返回可微零值，轨迹损失仍覆盖全部样本。
> 有效ABC的编码和损失语义保持不变，同时修复批大小1/列向量目标的形状处理。
> 这些是缺失标签处理，不是把Fission生物标签或Budding六分类改成分类目标。见 [兼容验证](FISSION_EXTENSION.md)。


> **2026-10-20 Fission物理后端**：从canonical方程AST生成可微RHS，处理mass相对尺度和
> 由完整目标轨迹事件历史重建的隐含kS；单点RDNA不能可靠推断cut株阶段。
> PINN/PINO/Hybrid PINO/HybridAll按系统分派，四种归一化的活跃物理损失和梯度已验证。
> 可微重置、事件掩码与限制见 [Fission扩展](FISSION_EXTENSION.md)。

> **2026-10-19 修复**：PINOLoss、HybridPINOCollocationLoss、HybridAllLoss 显式接收并转发
> `norm_type/s_c`，解决训练启动 TypeError；四类物理损失的解码与 DataModule/HANO 共用同一变换。
> asinh_scale 缺少 s_c 时直接报错，不再把 asinh 后的数当浓度。未重写损失公式、权重或评估指标。
> checkpoint 同时恢复物理损失使用的预处理。见 [运行契约修复](RUNTIME_CONTRACT_FIX_2026-10-19.md)。

本文档系统梳理了 `budding_yeast_v2` 神经算子框架中实现的 **17 种损失函数**、生物物理约束（ODE 残差与事件跳跃）、39 通道全量微观梯度自适应平衡（GradNorm）以及课程学习权重衰减策略。

> **2026-10-18 实现核验**：训练 loss 默认在 `Zscore(log1p(Y))` 空间计算，
> 其 relative term 使用 `sqrt(sum(z²)+0.1²)` 并截断到 10；物理空间评估 relL2
> 使用 `sqrt(max(sum(Y²),0.1²*T))`，两者不是同一指标。
> TripleDynamic 的 DTW 项只覆盖 9 个指定通道的 `0,20,...,480` 点，不含 ORI/Vi20
> 或最后 19 点；其他 MSE/导数项仍覆盖这些误差。独立递推核查未发现 Soft-DTW 算法错误，
> 不能将负训练 loss 本身当作失败证据。复现、适用边界及坏例时域定位见
> [当前证据审计 §2.3](PURE_RANDOM_BADCASE_AUDIT_2026-10-18.md#23-训练损失的具体监督范围与评估-rell2-不一致)。
> 本次没有改动生产 loss 或证明新的重训收益。

---

## 一、损失函数注册表 (Loss Registry Overview)

所有损失函数均在 [utils/losses.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/losses.py) 中统一定义并通过 `LOSS_REGISTRY` 注册，可通过 `train.py --loss <loss_name>` 命令行参数无缝切换：

| 损失函数标识 (`--loss`) | 类名 | 核心机制 | 适用场景与特性 |
| :--- | :--- | :--- | :--- |
| **`hybrid_mse_rel`** (默认推荐) | `HybridMSERelLoss` | $\text{MSE} + \lambda_{\text{rel}} \cdot \text{RelL2}$ | 兼顾振荡波形相关系数与微量细胞周期蛋白公平性 |
| **`gradnorm_hybrid`** | `GradNormHybridLoss` | 39 通道自适应梯度平衡 | 单次 Autograd 技巧，动态均衡 39 个生化变量的梯度贡献 |
| **`rel_l2`** | `RelativeL2Loss` | 安全相对 $L_2$ 损失 | 针对微量变量（如 Clb1/2, Sic1）加权，带能量平滑项 $\epsilon$ |
| **`mse_only`** | `MSEOnlyLoss` | 标准均方误差 | 经典回归基准，容易被高能量宏观变量（BUD/MASS）主导 |
| **`hybrid_bifurcate`** | `HybridBifurcateLoss` | Hybrid + 表型交叉熵 + 阻滞平坦约束 | 抑制假分岔漂移，强化野生型振荡 vs 突变体阻滞分类 |
| **`hybrid_pino_colloc`** | `HybridPINOCollocationLoss` | 配点法 ODE 残差 + 细胞分裂事件跳跃惩罚 | 结合生化反应速率 ODE 约束与解耦事件掩码 |
| **`hybrid_all`** | `HybridAllLoss` | 方案 A + C 全融合旗舰损失 | 数据损失 + 表型分类 + 阻滞约束 + PINO 物理残差 |
| **`physics_informed`** | `PhysicsInformedLoss` | MSE + 生物非负惩罚 + FFT 谱平滑 | 早期物理约束，抑制负浓度与高频抖动 |
| **`mse_negpen`** | `MSENegPenLoss` | MSE + 非负性惩罚 | 确保反归一化后物理浓度非负 |
| **`mse_smooth`** | `MSESmoothLoss` | MSE + FFT 频域高频平滑项 | 压制反向传播产生的高频伪影 |
| **`t_smooth`** | `DilatedSobolevLoss` | 空时膨胀 Sobolev 范数 | 惩罚一阶时序差分梯度误差 |
| **`pinn_residual`** | `PINNResidualLoss` | 连续时序导数与 ODE 残差 | 针对可微插值时序进行自动微分残差计算 |
| **`h1`** | `H1Loss` | Sobolev $H^1$ 范数损失 | 同时约束函数值及其一阶导数 |
| **`pino`** | `PINOLoss` | 经典 PINO 算子与物理方程损失 | 算子学习与物理方程联合优化 |
| **`phase_derivative`** | `PhaseDerivativeLoss` | 相空间轨迹一阶导数约束 | 强化相平面环面拓扑形态重构 |
| **`soft_dtw`** | `SoftDTWLoss` | 可微动态时间规整 | 解决时序相位漂移与周期未对齐问题 |
| **`triple_dynamic`** / `hybrid_phase_dtw` | `TripleDynamicLoss` | MSE + SoftDTW + 相空间导数 | 综合波形、时延鲁棒性与相空间轨迹保真度 |

---

## 二、核心损失数学原理与工程实现

### 1. 安全相对 $L_2$ 损失 (`RelativeL2Loss`)
在 39 个酵母细胞周期生化变量中，宏观形态变量（如 `MASS`, `BUD` 浓度最高达 10.0+）与关键细胞周期调控因子（如游离 `Clb2`, `Sic1` 浓度通常小于 0.05）存在跨越 **3 个数量级** 的量纲差异。若直接采用 MSE，微量蛋白将发生严重欠拟合。

**数学公式：**
$$\mathcal{L}_{\text{rel}} = \frac{1}{B \cdot V} \sum_{b=1}^{B} \sum_{v=1}^{V} \min \left( \frac{\| \hat{Y}_{b,v} - Y_{b,v} \|_2}{\sqrt{\| Y_{b,v} \|_2^2 + \epsilon^2}}, \; \text{clamp}_{\max} \right)$$
- **$\epsilon = 0.1$**：保证分母平滑且恒正，防止真值为 0 时除零产生 NaN。
- **$\text{clamp}_{\max} = 10.0$**：截断离群极值，彻底消除梯度爆炸。

---

### 2. 39 通道全量微观梯度自适应平衡 (`GradNormHybridLoss`)
为了彻底解决微量蛋白与宏观变量的训练梯度不平衡问题，框架实现了 39 通道解耦 GradNorm 算法。

**关键技术创新：单次 Autograd 极速求导技巧 (Single-Call Decoupled Norm Trick)**
传统多任务 GradNorm 对 $N$ 个任务需要 $N$ 次完整的 `torch.autograd.grad` 反向图遍历，计算开销巨大。然而在我们的场景下，通道损失 $\mathcal{L}_v$ 仅由输出层通道 $v$ 决定（在最后一层权重 $W_{\text{last}}$ 上正交解耦）：
$$\frac{\partial \sum_{v=1}^{39} \mathcal{L}_v}{\partial W_{\text{last}}^{(v)}} = \frac{\partial \mathcal{L}_v}{\partial W_{\text{last}}^{(v)}}$$
因此，**仅需 1 次 `autograd.grad` 即可同时解析出全部 39 个通道各自的精确 $L_2$ 梯度范数 $G_v(t)$**，单步计算开销从 80ms 骤降至 `<2ms`！

**算法步骤：**
1. 计算 39 通道损失相对初始损失的逆训练进度：
   $$\tilde{\mathcal{L}}_v(t) = \frac{\mathcal{L}_v(t)}{\mathcal{L}_v(0)}, \quad r_v(t) = \frac{\tilde{\mathcal{L}}_v(t)}{\frac{1}{39}\sum_{i=1}^{39} \tilde{\mathcal{L}}_i(t)}$$
2. 设定目标梯度强度 $G_{\text{target}, v}(t) = \bar{G}(t) \cdot [r_v(t)]^\alpha$。
3. 闭式 EMA 平滑更新通道权重（保证 Lightning DDP 混合精度安全，无需二次反向传播）：
   $$\Delta w_v = \eta \cdot (G_v(t) - G_{\text{target}, v}(t))$$
   $$w_v(t) \leftarrow \text{Clip}\left( (1-\gamma) w_v(t-1) + \gamma (w_v(t-1) - \Delta w_v), \; w_{\min}, \; w_{\max} \right)$$
   $$w_v(t) \leftarrow \frac{w_v(t)}{\frac{1}{39} \sum_{i=1}^{39} w_i(t)}$$

---

### 3. PINO 配点法生物物理 ODE 残差与事件跳跃 (`HybridPINOCollocationLoss`)
酵母细胞周期包含 39 个常微分方程组成的非线性动力学系统（详见 [utils/physics_eqns_torch.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/physics_eqns_torch.py)），并伴随由离散检查点触发的 **不连续跳跃事件 (Event Jumps)**：
- 芽颈收缩与出芽触发（BUD 重置为 0 / 1）
- 纺锤体极体（SPB）分离
- 细胞分裂质量均分（`MASS` 减半）

**物理残差计算与事件解耦掩码：**
1. **时间导数离散化**：采用 5 阶内节点有限差分近似求取 $\frac{d\hat{Y}}{dt}$。
2. **生化反应通量计算**：利用 GPU 并行向量化 PyTorch 计算 39 个微分变量的理论反应流速 $f_v(\hat{Y}(t), p)$。
3. **事件掩码解耦 (`detect_event_mask_torch`)**：在细胞分裂事件发生的窗口内（$\pm 4$ 时间步），将物理残差权重置为 0，防止对非连续跳跃点强行进行微分惩罚；
4. **Huber 鲁棒物理残差**：
   $$\mathcal{L}_{\text{phys}} = \frac{1}{|\Omega|} \sum_{(b, v, t) \in \Omega} \text{Huber}_\delta \left( \frac{d\hat{Y}_{b,v}}{dt}(t) - f_v(\hat{Y}(t), p_b) \right)$$
5. **跳跃幅度保真度惩罚 (`_compute_jump_loss`)**：
   $$\mathcal{L}_{\text{jump}} = \frac{1}{|\Omega_{\text{jump}}|} \sum_{(b, v, t) \in \Omega_{\text{jump}}} \| (\hat{Y}(t^+) - \hat{Y}(t^-)) - (Y(t^+) - Y(t^-)) \|^2$$

---

### 4. 表型分岔分类与阻滞稳态平坦度约束 (`HybridBifurcateLoss`)
为解决“假分岔漂移”问题（即模型将本应阻滞在 G1 期的致死突变体预测为微弱持续振荡）：
1. **表型分类交叉熵**：利用辅助分类头预测 3 类动力学表型（0: 正常振荡 Wildtype, 1: 延迟振荡 Delayed, 2: 周期阻滞 Arrest）：
   $$\mathcal{L}_{\text{cls}} = \text{CrossEntropy}(\text{Logits}_{\text{cls}}, \text{Target}_{\text{cls}})$$
2. **阻滞稳态平坦度惩罚 (Arrest Flatness Penalty)**：
   对标签为阻滞状态（Arrest）的样本，惩罚其周期后半程（$t \ge T/2$）的时序一阶差分，强制收敛至定常稳态：
   $$\mathcal{L}_{\text{arrest}} = \frac{1}{B_{\text{arrest}} \cdot V \cdot (T/2)} \sum_{b \in \text{Arrest}} \sum_{v=1}^{V} \sum_{t=T/2}^{T-1} (\hat{Y}_{b,v}(t+1) - \hat{Y}_{b,v}(t))^2$$

---

## 三、课程学习与动态权重退火 (Curriculum Weighting)

在模型训练初期，网络输出波动较大，此时直接施加高强度的物理约束或频域平滑项容易导致优化陷入局部极小值。系统提供了自动课程退火调度器 `get_curriculum_weights`：

```python
def get_curriculum_weights(epoch, max_epochs=120):
    if epoch < 15:
        # 预热期：纯数据驱动拟合基础振荡波形
        return 0.0, 0.0
    elif epoch < 60:
        # 退火期：线性注入非负性物理惩罚与谱平滑
        progress = (epoch - 15) / 45.0
        lam_pen = 10.0 * progress
        lam_sm = 0.5 * progress
        return lam_pen, lam_sm
    else:
        # 稳态期：物理约束全量施加
        return 10.0, 0.5
```

---

## 四、训练实战调用配置速查

### 1. 生产基准训练 (`hybrid_mse_rel`)
```bash
python train.py \
  --model biom_refiner_plus \
  --loss hybrid_mse_rel \
  --lambda_rel 0.2 \
  --batch_size 256 \
  --lr 8.8e-4 \
  --max_epochs 100 \
  --modes 24
```

### 2. 开启 39 通道 GradNorm 自适应平衡 (`gradnorm_hybrid`)
```bash
python train.py \
  --model biom_refiner_plus \
  --loss gradnorm_hybrid \
  --lambda_rel 0.2 \
  --max_epochs 100
```

### 3. 联合 PINO 生物物理残差与跳跃事件约束 (`hybrid_pino_colloc`)
```bash
python train.py \
  --model biom_refiner_plus \
  --loss hybrid_pino_colloc \
  --lambda_rel 0.2 \
  --lambda_phys 0.01 \
  --lambda_jump 0.001 \
  --warmup_epochs 30
```

### 4. 联合动力学表型分类与阻滞约束 (`hybrid_bifurcate`)
```bash
python train.py \
  --model biom_refiner_plus \
  --loss hybrid_bifurcate \
  --lambda_rel 0.2 \
  --lambda_pattern 0.1 \
  --lambda_arrest 0.05
```


## 2026-10-21：标准 SC-FNO 灵敏度损失

注册名 `sc_sensitivity`，类 `SensitivityLoss`：
`L = MSE(Y_pred,Y) + sensitivity_weight * mean_k MSE(dY_pred/dP_k, S_k)`。
这里 Y、P 及 S 必须处于同一归一化坐标，且求参数偏导时固定全部初值。
模型仍是标准 FNO，不增加参数调制。JVP 对每个选定参数计算完整轨迹方向导数，并保留网络参数的高阶梯度；
权重必须为正且有限。验证/测试只计算轨迹 MSE，不使用测试灵敏度。没有参考监督时训练报错。

CLI：`--model sc_fno --loss_type sc_sensitivity --sensitivity_path <reference.npz> --sensitivity_weight 1.0 --al_strategy none`。
sidecar 的形状、数据哈希、归一化指纹及事件边界要求见 [标准基线说明](BASELINE_SELECTION_AUDIT.md)。
人工解析函数测试验证了 Jacobian 数值和训练参数梯度；SC-FNO 骨干与相同权重的 FNO 输出完全一致。
