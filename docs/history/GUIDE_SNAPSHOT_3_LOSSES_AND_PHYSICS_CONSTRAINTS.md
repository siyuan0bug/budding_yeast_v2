> 历史内容快照，不代表全部规则仍然有效；当前说明以docs根目录对应指南为准。

> **2026-11-14 单一训练实现**：新旧流程共用`models/yeast_lit_module.py`的training_step、AdamW和warmup+余弦调度；不再维护FNO专用手写梯度循环。新完整池训练因此采用与现有主流程一致的调度端点和梯度裁剪1.0；不能将整理前诊断权重视为逐位相同的训练复现。loss注册与重要性权重计算保持公共实现，验证使用共享`trajectory_relative_errors`。生物ODE、初值与mutation规则未在本次整理中改变。

> **2026-11-14 损失迁移与带权筛选**：原loss统一迁至`utils/training/losses.py`，旧import仍可用。MSE、RelL2和Hybrid新增逐样本计算；带权训练采用`mean_i(w_i×loss_i)`，不按每批权重和再次归一化；全1权重走原计算路径。公共DataModule将weights作为第五个训练张量，YeastLitModule显式消费；不支持权重的loss拒绝非均匀weights。原生clustering-sensitivity的m个有放回抽签聚合为k个不同ID时，用`w_i=k*c_i/(m*n*p_i)`保持其全池风险估计；WOR适配不使用这组权重。数值/梯度回归核对重复展开与聚合等价。新协议采用原Hybrid主损失，不添加ours专用训练损失。[实施状态](BUDDING_SAMPLING_BENCHMARK_PLAN_2026-11-13.md#13-2026-11-14-实施交付与边界)。

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

所有损失函数均在 [utils/training/losses.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/training/losses.py) 中统一定义并通过 `LOSS_REGISTRY` 注册，可通过 `train.py --loss <loss_name>` 命令行参数无缝切换：

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
酵母细胞周期包含 39 个常微分方程组成的非线性动力学系统（详见 [utils/physics/budding.py](file:///home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/utils/physics/budding.py)），并伴随由离散检查点触发的 **不连续跳跃事件 (Event Jumps)**：
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

2026-11-15：用户统一未来基线训练为现有rel_l2；eps=.1、比值cap10、归一化空间计算。val/test归一化loss同步同一criterion，MSE另存；物理指标保持原口径。 详见[BUDDING_REL_L2_BATCH_2026-11-15.md](BUDDING_REL_L2_BATCH_2026-11-15.md)。

## 2026-11-15：相近多输出任务的 Relative L2 源码核对

本节只核查定义与参考实现，不修改训练损失、评测公式或已完成实验。结论是：**逐样本、逐变量的 Relative L2 是本课题合理的主训练损失方向；当前实现的归一化、分母保护、比值截断和历史汇总方式必须单独说明，不能把它们统称为完全相同的 Relative L2。** 不能仅因一个历史汇总指标变差，就否定整个相对损失家族，也不能据此宣称目前实现已经最优。

### 1. 筛选参考任务，而不是只查找同名 loss

以下均核对本地代码的损失调用、张量归约和预处理，不从类名推断训练行为。外部参考目录未修改、未运行其训练。

| 参考项目／任务 | 多输出与量级问题 | 实际损失路径 | 与本课题的关系 |
|---|---|---|---|
| HyperNOs 的 Hodgkin–Huxley、FitzHugh–Nagumo | HH 输出 V/m/h/n，FHN 输出 v/w；不同状态分别归一化 | 选择 `L2` 时为逐样本、逐输出通道的相对 L2，再平均通道；标准 HH 路径在归一化目标上训练 | 最接近的多变量生物 ODE 参考；不是原酵母模型 |
| GNOT 的多输出 PDE 路径 | 图节点上的多个输出场，按输出维度标准化 | 默认 `rel2`、`component=all`；每图每通道先计算相对范数，再平均图与通道。训练用标准化值，报告 metrics 时逆变换 | 支持通道分别计算相对误差，也说明训练／物理评测可以有意使用不同坐标 |
| Transolver 的 AirfRANS | 同时预测 Ux/Uy/p/nut；不同物理量分别标准化 | `MSE_weighted`：标准化空间 MSE，对体域和表面分别求损失并加权 | 多量纲任务也有标准化 MSE 的官方方案；不存在所有作者统一使用相对 L2 的规则 |
| Tau-BNO 的 FNO 路径 | 多脑区时序；各位置可具有不同幅值，但与不同分子物种并不完全等价 | `neuralop.LpLoss(d=2,p=2,reduction='sum')`；可再加非负辅助项。普通 `cat` 路径目标为时间×脑区，d=2 将这两个轴合并计算每样本相对 L2 | 是多位置轨迹参考，不能当作逐蛋白通道相对 L2 的同义实现 |

HyperNOs 注意事项：库支持 L1/L2/H1/MSE 等可选损失；**其 HyperNOs 论文数值实验节声明使用 mean relative L1，不能把“代码支持 L2”写成“该论文全部实验采用 L2”**。HH 的 GaussianNormalizer 在训练样本轴计算统计，保留时间位置维度；也不等同于本项目跨样本／时间逐变量拟合的 log1p Z-score。

HGGS 虽包含细胞周期 ODE，当前仓库回归目标主要是振荡频率特征，不是完整 39 变量轨迹；其 MSE 设置不作为选择本项目轨迹损失的直接证据。标量 Darcy 或同一个场的许多网格点也不能单独证明异量级分子通道已经得到平衡。

本地证据位置（相对项目父目录）：

- HyperNOs：`baselines/neural_model/HyperNOs/hypernos/loss_fun.py:18,176`；`datasets.py:2402,2448`；`utilities.py:132`；`train.py:393,428`；`examples/train_fno.py:64,86`；`wrappers/wrap_model.py` 的 HH 默认返回原模型。
- GNOT：`baselines/neural_model/GNOT/args.py:68`；`data_utils.py:103,363,491,521`；`utils.py:408`；`train.py:166`。图节点范数为无网格面积权重的离散求和，不能声称等价于任意非均匀网格的严格积分范数。
- Transolver：`baselines/neural_model/transolver/Airfoil-Design-AirfRANS/main.py:36,93`；`dataset/dataset.py:148,196`；`train.py:27`。
- Tau-BNO：`baselines/neural_model/Tau_BNO-main/train_scripts/train_fno.py:48`；`dataset/NpySimsDataset.py:38,67`。LpLoss 细节核对当前 neuralop 环境的 `neuralop/losses/data_losses.py:171`；不能据此断言论文原始依赖版本完全一致。

参考仓库当前 HEAD：HyperNOs `39a6cb3dd3c935b0365fe32cde131cdcc5091bae`；GNOT `5ee2e6925a43f9a340a6016bad4da2c82a452cbe`；Transolver `75e0f67643806a81cd1d3f6adc88dd8c02416fe7`。以上行为以核查的本地文件内容为准。

官方来源：[HyperNOs 仓库](https://github.com/MaxGhi8/HyperNOs)、[HyperNOs 论文 §2.2/§3.3/§5](https://link.springer.com/article/10.1007/s40574-025-00516-0)、[GNOT 仓库及 rel2 示例命令](https://github.com/thu-ml/GNOT)、[GNOT 论文](https://arxiv.org/abs/2302.14376)。

### 2. 本项目三个主要口径

设 i 为样本、v 为变量、t 为时间；N、V、T 为对应数量，时间范数使用全部采样点。

**当前训练损失**（`utils/training/losses.py:65`，无非均匀样本权重时）：

\[
z_{ivt}=\frac{\log(1+\max(y_{ivt},0))-\mu_v}{\sigma_v},\qquad
L_{train}=\frac{1}{NV}\sum_{i,v}\min\left(
\frac{\|\hat z_{iv}-z_{iv}\|_2}{\sqrt{\|z_{iv}\|_2^2+0.1^2}},10\right).
\]

每条轨迹每个变量先算一个数，再等权平均；不是逐时间点的百分比误差。训练归一化 stats 只来自本轮累计合成 train。

**历史物理空间汇总 `all_variables.Relative L2`**（`utils/evaluation/metrics.py:5`）：

\[
R_{pooled}=\frac1V\sum_v\sqrt{\frac{
\sum_{i,t}(\hat y_{ivt}-y_{ivt})^2}{
\max(\sum_{i,t}y_{ivt}^2,\ 0.1^2NT)}}.
\]

先把所有样本的同一变量串起来计算一个相对范数，再平均变量。变量之间等权，但变量内部高能量样本影响大；它不是逐株误差的算术平均。无 cap10，保护项具有物理幅值含义，且随 NT 缩放。

**`macro_trajectory_rel_l2`／验证选 checkpoint 的口径**（同文件 `:91`）：

\[
R_{macro}=\frac1N\sum_i\frac{\|\hat Y_i-Y_i\|_F}{\|Y_i\|_F+10^{-8}}.
\]

先将每条样本的变量×时间合并为完整轨迹，再平均样本。这保证样本外层等权，但一条样本内部大幅值变量仍然影响更大；**它不等于逐样本、逐变量平均**。逐株 CSV 的 `rel_l2` 则是对单株调用历史公式，其 CSV 均值也不等于全体样本的 pooled 指标。

因此这里有三种不同问题：训练如何优化、完整物理轨迹相对误差如何汇总、每株每变量如何公平计分。名字都含 relative L2，并不使它们数学相同。

### 3. 两个小型数值核验

结果保存在忽略目录 `scratch/probes/relative_l2_reference_audit_20261115/toy_checks.json`。仅 CPU 小张量调用当前损失和评测函数；未加载正式轨迹、未训练或使用 GPU。

1. 同一个变量，两条常数轨迹真值分别为 1、100，误差分别为 1、10：逐样本相对 L2 平均为 **0.55**，历史 pooled 指标为 **0.10049**。两者没有计算错误，后者主要反映高能量样本。该例固定在物理空间，只隔离归约顺序的影响。
2. 真值恒为 1，取合法示例统计 `mu=log(2), sigma=0.4, T=500`：预测恒为 1.1 时，物理相对误差为 **0.1**，当前归一化相对比值为 **27.2745**，经 cap 得到 **10**，对该预测张量的梯度为 **0**。预测 1.01 时物理相对误差 **0.01**、归一化相对比值 **2.7881**，尚未触发 cap。

第二例解释的是性质，不是本批实验发生频率：减均值后，分母衡量“偏离训练均值多少”，而非“物理信号有多大”。中心化相对损失有官方使用先例，但其含义不能解释为原始浓度百分比误差。PyTorch 的 `clamp(loss_ratio,max=10)` 会使超过上限的该样本通道项局部梯度为零；不是 `clip_grad_norm_`，也不是轨迹大值过滤。尚未统计本批真实预测中 cap 触发比例，不能据此归因现有 badcase。

### 4. 后续建议（尚未实施）

- 继续以逐样本、逐变量 Relative L2 作为优先训练方向，不因单项历史 pooled 指标变差就自动回退 Hybrid。相对 L2 已按目标轨迹能量降低不同通道／样本幅值造成的支配，但零附近分母必须显式保护。
- 先明确训练在物理值、非中心化 log1p 值，还是当前中心化 z 值计算。输入／输出编码可以保持 log1p+Z-score，而损失通过可微逆变换在另一指定空间计算；两件事不是强制绑定。若目标是浓度相对精度，物理空间逐通道损失最直接；log 空间相对误差是不同的、小幅信号友好的目标，不能预先宣称其中一个必胜。
- 评估单株 badcase 和多变量平衡时，建议补充物理空间逐样本／逐变量平均；保留原 pooled 指标和完整轨迹 macro 指标作历史可比对照。指标变更必须单独命名，不能覆盖旧结果制造改善。
- 可讨论的保护形式为 `mean_{i,v}(||error_iv|| / sqrt(||y_iv||² + T*delta_v²))`；delta_v 用训练数据的变量尺度与预先冻结的规则确定，不用 real/test 误差调参。它是相对误差加绝对容差的混合尺度，不是精确的无保护相对误差。零真值通道另报绝对误差，避免分母趋零支配排名。
- 优先统计实际 cap10 触发频率、近零归一化目标通道和它们的梯度贡献；如需稳健化，比较分母保护、平滑稳健函数或优化器梯度范数裁剪，不把硬截断比值当作无副作用的稳定措施。HyperNOs 近零目标直接报错、GNOT 无显式零分母保护，同样不适合不加分析地照搬到有 KO 的酵母数据。
- 选择损失变体和超参数应在合成 validation 上完成。real mutant 和合成 test 仅用于最终报告／诊断，不能反复据其挑选损失而仍称完全独立评测。

### 5. 当前训练保持不变；eval 对齐不等于只报告训练 loss

用户随后明确继续使用当前训练 RelativeL2Loss；本次不更改 eps、cap、编码、checkpoint 选择或优化器。GNOT 与本项目的共同点是逐样本／逐变量范数后平均；区别在于 GNOT 无本项目 eps=.1 与 cap10，其 unit 标准化也不同于 log1p Z-score。GNOT 本身的训练 loss 与逆变换后的报告 metrics 也不是同一数值。

现有逐 epoch `test_loss_real`、`test_loss_synth` 已调用当前 criterion，属于与训练一致的目标；最终 `global_metrics_denorm.json` 的历史 `Relative L2` 则是物理空间 pooled 指标。因此应补充并明确标注最终评测中的训练目标，而不是把原物理指标改名或全替换：

| 报告项（建议字段，生产 JSON 尚未改） | 空间与用途 |
|---|---|
| `relative_l2_norm` | 保存 checkpoint 的训练统计，归一化空间、eps=.1、cap10，与训练完全一致 |
| `relative_l2_norm_uncapped` | 同一归一化空间与分母，不截断，仅用于诊断超限严重程度 |
| `clipped_pair_pct`、`sample_clip_pct` | 超限的样本×变量比例、至少一个变量超限的样本比例，必须区分 |
| `macro_sample_channel_rel_l2_physical` | 对现有逐株 CSV 的 `rel_l2` 算术平均；物理空间逐样本／变量平均，沿用其物理分母保护 |
| 原 MAE、MSE、Correlation、pooled Relative L2、完整轨迹 macro | 物理尺度解释、历史对比与不同统计目标，保留原名称与定义 |

“训练与 eval 一致”需要同时固定变换、统计量、求范数维度、分母和 cap；只复用 RelativeL2Loss 类并不意味着所有模型使用同一个尺度。不同轮次／方法重拟合训练统计，归一化指标不适合作为跨模型物理精度的唯一排名依据。若未来正式改主要指标，还须讨论 validation 选 checkpoint 的一致性，不能仅修改显示标签。

### 6. cap10 的实际 checkpoint 回放证据

工具：`scripts/analysis/audit_relative_l2_caps.py`；结果目录：`/home/users/zsy/mydata/budding_yeast/analysis/relative_l2_cap_audit_20261115/`。该工具分批重新推理，读取每池所需训练／holdout 行的并集一次，不保存预测轨迹、重复训练集或新模型；报告只保存 ID 和标量诊断。用户授权使用0/2/3号卡后，本次仅用2号卡单进程执行。

已确认的显著案例为 mutation＋BAIT 的 round_01（第二轮，4000训练样本，保存 epoch=97）：

- 训练集超限项 **3544/156000=2.271795%**，涉及 **3544/4000=88.6%** 的样本，全部是 CDC14T。
- 相同预测的平均 loss 经 cap 为 **0.433437**，取消 cap 的诊断值为 **6.515810**；不表示本次改变了训练损失。
- 该状态 real 中 **119/126** 样本的 CDC14T 超限。真实浓度通常恒为2，而训练 `mean_log=1.099745` 接近 `log(3)`，导致归一化目标范数很小；不是生物浓度接近零。
- 独立 CPU 物理检查：#98 的 CDC14T 真值恒为2，预测物理 MAE约 **1.66414**，归一化相对比值约 **447.813**。既有物理预测误差，也有中心化分母放大；不能把比值直接当作浓度百分比。

本次已完成 **16项×5轮=80个主模型、240组cohort回放**。80个保存状态中，训练集27个有超限、real仅2个有超限；最后一轮16个模型的real均无超限，mutation最后轮训练也均无超限。LHS最后轮5个训练模型出现超限，8个最终模型合计23个样本×变量检查项（每模型390000项）；不是23个跨模型唯一样本。不能将中间BAIT异常解释为所有最终模型都被cap掩盖。

160组real／合成test的回放均与对应保存epoch历史监控核对，最大loss绝对差 **1.44e-8**。审计用时约900秒，进程RSS峰值 **4.741 GiB**、CUDA allocated峰值 **0.365 GiB**；审计进程已退出。完整结果及资源记录见结果目录的 `audit.json`、`report.md`、`summary.csv`；物理复核为 `cdc14t_physical_check.json`。训练仍为原配置，无重训。

历史边界：保存的是各轮 best checkpoint，不是每次 optimizer step；历史 epoch CSV 没有 cap 计数。回放为 eval/no-autocast FP32，原训练为 bf16 mixed，因此不能把回放频率冒充历史训练全过程频率。未超限的 checkpoint 不能证明训练中间从未超限；超限项的局部零梯度也不能单独证明某个最终 badcase 的因果来源。ensemble 辅助成员不在本次主模型回放范围。

### 7. 用户最终确认：正式 eval 新增物理空间、不截断的主 Relative L2（已实施）

本节取代第5节“在最终 eval 增加归一化指标”的待定提议。用户明确最终评测应在物理空间进行，不截断误差；训练默认仍为当前归一化 RelativeL2Loss，除独立诊断对照外未更改。

公共实现 `utils/evaluation/metrics.py:sample_channel_relative_errors`：

\[
R_{SC}=\frac1{NV}\sum_{i,v}\frac{\|\hat y_{iv}-y_{iv}\|_{t,2}}
{\sqrt{\|y_{iv}\|_{t,2}^2+0.1^2}}.
\]

输入是反归一化模型预测和保存的原始ODE轨迹，不取log、不减均值、不截断误差比值。只保留正分母保护，保证KO的零轨迹也能评分。eps=.1与训练分母结构相同，是**轨迹范数的保护量**；不等同于旧物理指标的每时间点RMS=.1下限，也不等同于无保护的纯百分比误差。

- JSON：新增 `primary_metric=sample_channel_rel_l2`、`metric_space=physical`、`sample_channel_rel_l2`，在旧汇总块之前展示；`all_39_vars/all_16_vars/core_9_vars` 中的 `Sample-channel Relative L2` 也排在原 `Relative L2` 前。旧指标的计算保留不改名，便于历史比较。
- CSV：公共逐样本表新增 `sample_channel_rel_l2`；独立eval表新增 `Sample_Channel_Relative_L2`。原 `rel_l2/Relative_L2` 仍是旧保护定义。
- 图与汇总：预测图主Rel-L2改为新指标、明确标注sample/channel和physical；旧值另标Legacy。performance.csv/图优先新指标，W&B新增明确的物理指标key，离线存储约定不变。
- 真值路径：完整池benchmark本来就直接使用原始轨迹；独立 `eval.py` 与 `predict_real_checkpoints` 已改为按源行索引读取原始真值，不再把归一化目标反变换当真值，保留有限微小负值、避免编码／解码舍入。独立eval的参数说明也读取原参数，而非重建参数。
- 边界：训练／逐epoch优化诊断仍可记录归一化loss，它不是正式物理精度指标。validation选择checkpoint仍按此前完整物理轨迹macro口径，未因新增报告指标悄悄改变历史训练协议。没有批量覆盖既有正式结果或重画旧图。

验证：新增物理指标与 `RelativeL2Loss(max_clamp=None)` 在相同输入上的数值一致；零真值／预测1的短轨迹测试产生大于10的评分并原样输出JSON/CSV/图，证明无cap；源行乱序和微小负值真值读取通过。公共采样流程、输出、逐epoch监控联合回归 **62项通过**（既有Lightning/W&B告警）。

### 8. 训练cap10／不截断的实际小对照（已完成）

脚本 `scripts/analysis/probe_relative_l2_training_cap.py`；输出 `/home/users/zsy/mydata/budding_yeast/analysis/relative_l2_training_cap_probe_20261115/`，含report.md、comparison.png、results.json、comparison.csv及逐变量实测。

固定mutation＋BAIT round_01的相同保存权重（原epoch97）、相同4000训练行与原统计，不新增采样。两分支seed42、batch128、AdamW重置、LR1e-4、warmup2、cosine到1e-5；各20epoch/640步，均用梯度范数裁剪1，唯一处理差异是训练比值cap10或None。复用公共YeastLitModule、make_trainer、优化器／验证逻辑，无第二套FNO训练实现。512条固定同池validation选best；另512条同池test和126 real仅事后诊断。两组before四类指标完全相同，步数一致。

固定20epoch后的last结果（不是挑选有利epoch）：

|评测集合／指标|训练cap10|训练不截断|
|---|---:|---:|
|合成validation，新物理sample/channel RelL2|0.736101|1.022140|
|合成validation，MAE|0.415667|0.451412|
|合成validation，MSE|39.499632|55.817971|
|合成test，新物理sample/channel RelL2|0.792796|1.065693|
|real，新物理sample/channel RelL2|1.093951|1.205166|
|real，MAE|0.351466|0.359928|
|real，MSE|17.697548|22.663455|
|real，CDC14T MAE|1.140966|0.157075|

实测cap分支连续20个epoch都存在3544/156000个超限训练项；不截断分支末epoch仅剩2项。梯度范数裁剪前的最大总范数分别约1.076与37.031，无非有限loss，均正常完成。real逐变量检查显示取消cap后31/39变量的新物理RelL2变差，CDC14T修复并不等于整个模型提升。验证集自身也退化，建议不依赖real结果就能得出“当前不应直接全局去cap”的判断。

这是针对异常状态的单seed恢复实验，改变学习率／优化器起点但对两臂相同；不是从头训练的完整benchmark，不能推广为“cap永远更好”。新原规则best（完整物理轨迹macro）在uncapped分支选择第1epoch，等同几乎不更新的起点；best/last两套结果完整保留。中心化分母让某些通道得到很大优化权重是机制解释，尚未单独隔离所有共享网络效应。

官方对照：本地GNOT、HyperNOs的L2、Tau-BNO所调用LpLoss均没有该ratio cap10。GNOT的`clip_grad_norm_`是梯度范数裁剪，与损失截断不同。本项目的采样baseline都共用当前训练器，因此当前项目Random/Uncertainty等都具有cap10；这是统一任务训练协议里的自定义稳健化，不能称为上述作者的原版loss。正式训练默认暂保留，后续若要消除超限零梯度盲区，应预注册分母／平滑稳健化或冷启动对照，不能仅凭这个小实验直接替换全部正式训练。

### 9. 仅采用新物理Relative L2；旧定义移出生产路径

用户最新倾向只采用新指标，因此新评测不再计算／展示旧pooled分数：公共`relative_l2_loss`、JSON的`Relative L2`、逐株CSV的`rel_l2`、独立eval的`Relative_L2`和逐epoch物理监控现在统一为新sample/channel定义。旧的第二列／Legacy图标题已删除，旧实现仅在`utils/archive/evaluation_metrics_legacy.py`显式复现历史数据，生产代码不导入它。完整轨迹macro仍是独立的验证／动力学诊断，不冒充新的逐通道指标。

新JSON/CSV与新history标记`physical_sample_channel_rel_l2_v1`。历史文件不覆盖、不改名，不将旧值假装成新值。多seed汇总拒绝混合不同定义，历史loss对比脚本也拒绝把新定义当旧pooled分数比较。

两个经代码核验的物理例子（均500个时间点）：

|例子|新定义|旧定义|
|---|---:|---:|
|同一变量两株：真值1/100，预测2/110|0.549995≈55%|0.100494≈10.05%|
|零真值，预测恒0.01|2.236068|0.1|

第一例体现先算每株、再平均的区别：旧结果是样本能量加权的范数比，容易被真值100的样本主导。两种定义原本都在最外层等权平均变量，不能宣称旧定义没有变量间平衡。

第二例体现不同零附近保护：新分母为sqrt(轨迹能量+.1²)，旧分母的物理RMS至少.1。零真值时不能将任一保护评分解释为严格百分比误差；新定义对低信号的伪表达更敏感。单株图N=1时，第一种差异消失，因此信号不小时图上新旧数值本来就可能很接近。

随后启动60epoch冷启动对照：同一4k训练集合和统计、seed42相同随机权重、batch128、LR7.5e-4、warmup4、minLR1e-5、梯度范数裁剪1；每臂1920步，只改变训练cap10/None。起始模型权重SHA完全相同，且确认不是原训练checkpoint的权重；源码checkpoint仅提供结构／冻结统计。结果目录`analysis/relative_l2_cap_60epoch_coldstart_20261115/`。大而有限的loss允许继续优化，NaN/Inf loss或梯度会明确报错，不以截断掩盖数值失败。完整结果见下文，训练默认尚未改。

60epoch现已完整完成，两臂初始权重SHA、before指标、数据顺序seed均一致，各1920步；记录的loss和梯度全部有限，初期整体loss没有爆炸。

|60epoch冷启动，固定预算last|cap10|不截断|
|---|---:|---:|
|初始／最大训练loss|1.061545|1.061545|
|末轮训练loss|0.258658|0.258306|
|末10轮训练loss平均每epoch下降率|0.3041%|0.2968%|
|全部训练项中ratio>10的次数|24/9360000|22/9360000|
|合成validation，新物理Relative L2|1.776419|1.771358|
|合成test，新物理Relative L2|1.904881|1.901505|
|real，新物理Relative L2|2.097527|2.097539|
|real，MAE|0.365647|0.364959|
|real，MSE|18.216617|17.863616|

这次cap极少实际触发，因此两组整体表现接近；结果支持“不截断也能稳定结束”，不支持“所有任务中cap都无用/有害”的推广。此前20epoch是异常已训练状态的恢复实验，不能与这次冷启动直接当作同一组训练长度比较。按末段变化率也不能断言已经严格收敛，且学习率已接近下限。单seed结论不用于自动切换正式训练默认。

本次单进程CPU RSS峰值3.044GiB、CUDA allocated峰值0.902GiB，训练和后台tmux会话均已正常退出。`stability_summary.json`单列loss/梯度有限性与变化率；`metric_examples.json`包含上文新旧分数例子。相关公共流程62项集成回归，以及6项物理指标专项（4项与集成重叠）均通过，共64个不同测试。

### 10. 用户确认：训练默认不截断，旧指标实现完全删除

最新指令已实施：`RelativeL2Loss`默认`max_clamp=None`；Hybrid/GradNorm及内部复用它的相对项也不再默认启用cap10。保留显式可选cap参数仅供既有历史诊断脚本复现，正常模型／训练入口没有任何隐式cap10。没有修改其它不同目标的物理残差或时序导数限幅。

旧pooled物理评测的归档实现`evaluation_metrics_legacy.py`也已删除，全部Python引用和专项测试的旧实现依赖清除；新结果只用新Relative L2。历史JSON/CSV和历史分析报告作为已完成实验记录保持原样，不将旧数字冒充新指标。

去掉损失上限后，仍保留：

1. **损失内部的正分母保护**：`sqrt(sum(target_norm²)+eps²)`，默认eps=.1，故分母至少.1；构造时拒绝eps≤0、NaN或Inf。零目标不会导致除零，大误差仍产生梯度。
2. **训练器梯度范数裁剪1**：优化更新前缩放过大的梯度，既不改变报告的loss值，也不直接删除超10项的梯度方向。`train.py`和公共`make_trainer`的既有配置保留。
3. **非有限loss检查**：`YeastLitModule.training_step`检测NaN/Inf并明确报错，不靠损失截断掩盖数值失败。这是失败检测，不是保证永不失败。
4. **既有预处理**：状态log1p后逐变量Z-score、参数逐维Z-score；标准差加1e-6，缓解尺度与零标准差问题。这属于编码，不是loss比值截断，正式eval仍用原始物理真值。

验证：零目标、500点预测全1时，默认loss约223.6068（大于10），所有预测分量梯度有限且为正；完美预测的loss为0且梯度有限。正式模型与Hybrid默认均无cap，新checkpoint的`training_objective_definition`记录实际eps、max_clamp、空间和归约方式。相关29项检查通过，包含8种采样baseline的五轮小型集成、物理评测和逐epoch监控；另对新增checkpoint记录作专项复核。没有启动新的正式批次或覆盖历史权重。


### 11. 2026-11-16：max分母及epsilon受控对照

用户确认分母改为 `max(||target||_time, eps)`。RelativeL2Loss的forward/per_sample、GradNorm复用相对项、公共物理评测及所有调用它的监控同步采用floor。训练仍在归一化空间，正式eval仍在原始物理空间，均按样本/变量等权平均，不截断loss；梯度范数裁剪1不变。范数≥eps时分母精确保持原值。max只依赖固定目标，不会在模型参数的分母路径产生阈值折点；但较小eps会增加小范数通道权重，不能凭数学可微性保证精度。

物理指标版本升为 `physical_sample_channel_rel_l2_floor_v2`，新结果不可和smooth v1/旧pooled值混算。正式输出schema升为v4，source校验阻止旧结果目录续写。checkpoint新增 `denominator_rule=max_target_time_l2_eps`，恢复当前floor checkpoint时保留实际eps；历史权重/结果不改写。

新增诊断入口 `scripts/analysis/probe_relative_l2_epsilon.py`，复用cap探针的数据适配器、数据读取及评测，训练仍走YeastLitModule、公共make_trainer和原优化器。只做单seed探索，不启动新的正式采样批次。

对照：mutation/BAIT round01的同一4000条已查询训练输入；使用这4000条原有train-only归一化，重新初始化模型而非接续已训练权重。FNO192/4blocks/modes24，batch128，seed42相同初始权重及每批顺序；各60epoch/1920步，LR7.5e-4，warmup4，cosine到1e-5；训练eps分别0.03/0.1/0.3，无loss cap，梯度范数裁剪1。CPU加载一次、单GPU进程顺序训练三臂，不保存轨迹副本。

三臂物理评测统一采用floor eps=0.1。预先定义用第60epoch的512条固定合成validation物理逐样本/变量RelL2选择eps；best另按既有validation macro保存。只有validation_decision.json写入后才推理512条合成test及126株real；不允许依据其结果改选。训练loss因eps不同不能直接跨臂排名。epsilon是统计权重而非机器精度，不保证0.1最优；训练与物理评测使用相同数字也不代表相同物理尺度。

输出：`/home/users/zsy/mydata/budding_yeast/analysis/relative_l2_epsilon_floor_20261116/`，包含训练ID引用、范数触发比例、每臂history/loss/梯度诊断/best及last、统一comparison.csv/png和验证决策；正式数据与旧结果保留。

对照已完成：三臂均60epoch/1920步，相同初始权重SHA、相同批顺序seed；全部记录loss/梯度有限。按预设的最后epoch合成validation物理RelL2选中训练eps=0.3，validation_decision.json在测试/real推理之前写入。best按原validation macro选取，结果方向同样支持0.3。

|训练eps|validation物理RelL2|合成test物理RelL2|real物理RelL2|real MAE|real MSE|裁剪前最大梯度|
|---:|---:|---:|---:|---:|---:|---:|
|0.03|1.906388|2.078948|2.272277|0.375326|19.174719|8.939169|
|0.1|1.779945|1.957462|2.224718|0.373068|18.294945|7.007946|
|0.3|1.536348|1.710899|2.006778|0.342805|14.722330|3.180620|

0.3相对0.1：validation物理RelL2降低13.69%，合成test降低12.60%，real降低9.80%；real MAE降低8.11%，MSE降低19.53%。所有对照的物理评测eps始终固定0.1，差异来自不同预测，不能解释为更换评测分母降低了分数。real只作事后报告，不用于eps选择。
156000个训练样本/变量项中，eps0.03没有触发下限，0.1与0.3均保护3544项（2.2718%），全部来自CDC14T；其变量内比例88.6%，范数0.076561。这里物理真值约2，并非KO零浓度，而是中心化后接近均值。0.3降低这部分项的相对训练权重；与其它变量改善相一致，未单独隔离所有因果通路。CDC14T本身的real MAE约0.1513/0.1513/0.1518，基本不变。
资源：单进程RSS峰值3.032GiB、CUDA allocated峰值0.902GiB；后台诊断已正常退出、GPU释放。诊断首轮发现复数梯度记录丢弃虚部，已修正并从头重跑；中止记录单独保留，不计入上述三臂。
代码检查35项通过：18项损失/评测/保存恢复/复数梯度专项＋17项采样五轮/逐epoch监控集成；包含8种baseline，git diff --check通过。comparison.png及三张loss.png已校验。
建议下一步训练优先采用eps=0.3，评测eps仍固定0.1；但当前生产构造函数默认eps暂保留0.1，未自动把单seed探索结论推广到全部模型/五轮实验。未启动正式重训或改写历史指标。
复现脚本、源文件哈希、行ID、统一指标版本、验证选择、best/last完整指标和范数统计保存在上述目录；verification_summary.json记录配对不变量及有限性核验。


### 12. 2026-11-16最终选择：平滑分母、epsilon=0.01

用户最终选择恢复 `sqrt(||target||_time² + eps²)`，训练与物理评测均默认eps=0.01，替代第11节的max规则及默认0.1。逐样本、逐变量沿时间求相对误差再平均，无loss cap；梯度范数裁剪1、非有限loss检查保留。状态编码/统计拟合和验证选checkpoint协议均未改变：训练仍用归一化目标，正式指标仍用原始物理真值。

RelativeL2Loss将未截断的逐通道误差集中在channel_errors，forward/per_sample及诊断共用；Hybrid、GradNorm、时序/DTW损失复用的相对幅度项默认eps同步为0.01。其它物理残差/导数目标的独立eps不作混改。普通model.test、公共eval/reporting、逐epoch测试监控统一调用新的物理helper，不再有显式0.1的相对轨迹评测分支。物理helper用float64范数及hypot计算平滑分母（数学等价sqrt(norm²+eps²)，避免平方溢出）。

较正常分母的改动可量化：原始非零目标范数d时，保护后与原始Relative L2的比例为d/sqrt(d²+0.01²)。d=0.1时约0.995037，d=1时约0.999950；不是严格不改正常分母。epsilon是统计保护尺度而非已证明的最优值；同样的0.01在归一化/物理空间没有相同的尺度意义，平滑也不能自动解决不同变量单位问题。这是用户确定的共同工程默认，并非依据real测试调参得出。

新物理指标版本 `physical_sample_channel_rel_l2_smooth_eps001_v3`、输出schema v5；新旧分数禁止混算，原有结果/数据/权重不覆写。checkpoint记录实际denominator_rule和eps；当前smooth checkpoint恢复其实际eps，显式历史floor checkpoint仅按记录的floor规则准确读取，不会被静默解释成新规则。fresh随机初始化的诊断只取旧checkpoint的预处理信息，不继承其历史损失设置。旧max专用epsilon脚本对当前版本明确拒绝执行，防止旧报告规则与新代码混用；已有0.03/0.1/0.3对照没有验证这次smooth eps0.01的长期性能。

没有生成新数据、重新评测历史正式结果或启动正式训练。相关检查结果补充如下。

验证完成：37项检查全部通过，覆盖精确平滑公式（含eps0.01）、forward/per_sample一致性、零目标及完美预测有限梯度、无cap默认、实际eps/规则保存恢复、历史floor读取、原始物理真值/指标版本/绘图CSV、8种baseline五轮小型流程以及逐epoch监控不影响训练。git diff --check通过。没有跑新的60epoch或正式重训，不把历史max对照当作本配置已验证的最终精度。


### 13. 2026-11-16：验证选checkpoint与主物理指标对齐

用户确认新的sampling benchmark用物理逐样本、逐变量Relative L2选best：`val_sample_channel_rel_l2`，与公共评测helper相同sqrt(norm²+0.01²)、无cap。原始物理validation真值（包括微小负值）保留，不以归一化目标逆变换替代；每batch先求每个样本/变量的时间范数比，按样本数累计，避免末尾短batch改变权重。checkpoint只看合成validation，test/real仍log-only。验证forward关闭autocast，按float32推理，与独立评测一致；训练仍bf16-mixed。旧macro模式仅供显式历史/诊断，不再是新默认。

配置增加`validation_metric=sample_channel_rel_l2`与`relative_l2_eps=0.01`；实际criterion接收eps，checkpoint记录实际规则，训练完成/恢复身份包含monitor，防止换选择规则后续用旧best。epoch日志增加`val_rel_l2_phys`，history保存完整名字，loss五曲线保留real物理曲线，并正确注释新的validation checkpoint选择依据。

新协议`seed42_02eebd57592f`由旧`seed42_bb035a147b38`派生；所有源池、seed42、初始样本ID和切分不变，四个输入/切分缓存inode保持相同，无轨迹副本。正式默认FNO192/modes24/blocks4、batch128、LR5e-4、minLR1e-5、weight decay1e-4；五轮2k/4k/6k/8k/10k、100/100/100/100/200epoch、warmup5/5/5/5/10。每轮冷启动，仅当前累计已查询合成训练样本拟合归一化。

相关检查分批通过：损失/评测/八baseline五轮及memory guard等55项、物理validation批大小/float32推理3项；另资源/共享训练/日志/曲线检查中发现历史图注文字变化，已恢复旧图注并验证新旧两种选择规则的五曲线图（5项定向复核通过）。新增普通失败继续/停止两种策略均验证。没有启动本配置的完整正式训练，不能据此保证新损失的精度和最终收敛。


### 14. 2026-11-16：完整结果不支持把新协议当作整体升级

新16项全部完成。按相同物理sample/channel smooth eps0.01重算新旧32个最终模型：mutation八方法平均MAE/MSE/统一RelL2分别增加15.90%/34.99%/20.25%；LHS相应变化+0.45%/-2.57%/-0.34%。只有一个seed，且同时改变eps、cap、best选择和自适应数据，不能单独判定某项损失改动必然有害。

新Random/Power的最终best来自56/36epoch，虽然都训练了200epoch；新Random物理Relative L2有96.68%来自4.97%的近零目标项。KO伪表达确实重要，但主指标与其它幅值/波形目标可能存在权重冲突。之前建议直接对齐best与主指标仍须单独验证；公式一致不代表归一化训练与物理评测的权重一致。下一步可用固定Random数据、同一训练同时保存macro-best/sample-channel-best/last作对照，不能从real结果中挑最有利checkpoint。本轮未改生产配置，也未启动该对照。完整结果见5号文档及analysis/sampling_protocol_comparison_20261116/report.md。


### 验证曲线与checkpoint复核（2026-11-16）

核对了新mutation/Random、Power最终轮history.csv、training.json与checkpoint epoch：保存逻辑确实选择val_sample_channel_rel_l2最低的56/36epoch。图中Val Synth Loss是归一化validation loss（val_loss_synth），最低均在200epoch；它不是当前checkpoint监控量。新Random：56epoch归一化val=0.233026、物理val=8.061796；200epoch为0.192378、8.150735。新Power：36epoch为0.260035、8.343722；200epoch为0.194164、8.365493。配置/保存逻辑一致，无证据是选错最小值的实现bug。

现有五曲线图只画了归一化Val Synth Loss，物理validation仅在checkpoint文字中标注，缺少监控曲线，展示有歧义。诊断补图validation_monitor_check.png将两者分别展示（20--200epoch放大，未改原实验图）。训练Relative采用log1p+Z-score后的逐样本/变量比；物理Relative采用原始浓度，近零和通道权重不同。旧批训练eps0.1且cap10、新批eps0.01无cap；旧best按物理整轨迹macro，新best按物理逐样本/变量。更早的pooled评测属于另一历史定义，不能混称为同一旧loss。本轮只核查与解释，未改选择规则、未重训。

### 15. 2026-11-16：不同空间的验证指标与较小epsilon对照

归一化Relative与物理Relative不是同一目标的简单换单位。即使只有线性Z-score，忽略epsilon时，单通道比值的分母也从`||y||`变成`||y-mean||`；再加log1p后，分子也对不同浓度赋予不同权重。因此归一化validation下降不保证物理validation同步下降，更不能据此认定checkpoint保存有bug。

使用最新mutation/Random最终轮CDH1T的训练统计（log均值0.63941938、标准差0.2686342），构造500点恒值例子，均预测高0.05：真值0时归一化Relative约0.0763、物理Relative约111.8034；真值1时相应为0.4596、0.0500。两种空间都用smooth eps0.01。此例是按真实统计构造的权重说明，不是声称模型实际输出了这两条曲线。它说明归一化目标接近0表示接近训练均值，物理目标接近0才表示近零浓度。

核对本地作者代码（未修改或导入执行）：

|参考实现|验证量及best处理|
|---|---|
|GNOT `data_utils.py:528`、`train.py:104`|loss在编码空间计算，但metrics通过normalizer.inverse计算，按原始空间rel2记录best；本地入口未恢复/保存best对应权重，不能直接把best记录当作完整checkpoint方案|
|Tau-BNO `train_scripts/train_fno.py:78`、`dataset/NpySimsDataset.py:31`|常用cat/zero分支先在dataset标准化输出，Trainer按val_l2保存best，验证入口没有inverse，因此是归一化空间|
|AL4PDE `models/model.py:201`、`scripts/run_active_learning.py:87`|按给定epoch训练后保存当前模型；该主流程没有逐epoch恢复validation-best，不能称为统一采用某种best规则|

不存在全部baseline都强制物理或归一化best的共同规则。应事先规定与科研目标一致的合成验证指标，所有采样方法统一执行，real/test不参与选择。生产配置本轮保持物理sample/channel validation选best，归一化loss仅作优化监控；Lightning本身允许monitor任意已记录量，见[官方checkpoint文档](https://lightning.ai/docs/pytorch/stable/optional-api/api/lightning.pytorch.callbacks.ModelCheckpoint)。

新诊断脚本`scripts/analysis/probe_smooth_relative_epsilon.py`复用既有FNO/Lightning训练器和cohort工具，未新增模型训练实现。使用历史mutation/BAIT round01已查询的同一4000条训练数据及固定512条validation；与此前epsilon测试同源，归一化只来自该训练集。三个独立冷启动臂均seed42、初始权重SHA相同、批顺序相同、FNO192/4/24、batch128、60epoch（各1920步）、LR7.5e-4、warmup4、最低LR1e-5、bf16训练、梯度裁剪1。仅训练epsilon变化，所有物理验证/评测固定smooth eps0.01；同时保存normalized-best、physical-best和last。validation结果落盘后才做test/real推理，不据real挑epsilon。

|训练epsilon|末轮物理validation RelL2|末轮物理validation MAE|单项Relative相对未保护形式最大减小|裁剪前最大梯度|
|---:|---:|---:|---:|---:|
|0.01|8.377018|0.494015|0.842249%|8.6782|
|0.003|8.628620|0.503551|0.076683%|8.6078|
|0.001|8.526945|0.501721|0.008529%|9.0804|

三组180个epoch全部完成，loss与梯度均有限；physical-best分别59/58/60epoch，normalized-best均60epoch，checkpoint与history最小值核验一致。梯度裁剪触发1913/1898/1910步（每组1920步），故只能声称在既有裁剪保护下完成，不能保证去掉裁剪也稳定。20项已有损失/物理评测回归通过。RSS峰值3.093GiB、CUDA allocated峰值0.902GiB，单GPU2进程已结束。

这批156000个归一化样本/通道项的最小目标范数0.076561，没有精确0或范数低于0.01的项。epsilon0.01平均只将原始Relative减小0.019237%，没有任何一项减小超过1%。公式为`R_eps/R_0=d/sqrt(d²+eps²)`（仅d>0）；当d≥7.018eps时改动≤1%。因此0.01已经接近原比值，这次小对照没有支持继续减小的验证收益。但该数据缺少极近零归一化范数，不能据此证明所有后续训练集合都稳定或0.01全局最优。严格零目标没有有限的原始Relative，必须明确分母保护；减小epsilon会提高这类项的权重。

结果位于`/home/users/zsy/mydata/budding_yeast/analysis/relative_l2_smooth_epsilon_20261116/`，包括report.md、comparison.png（归一化验证/物理验证/梯度并排）、comparison.csv、training_denominators.csv、coordinate_weight_example.csv及源行ID。历史正式结果不覆盖；本轮未改生产训练epsilon、物理评测epsilon或best规则，也未重新启动16项正式实验。训练epsilon的这次稳定性证据不能当作物理零浓度阈值的生物学标定。

#### 同批mutation/Power：epoch36与末轮的历史监控对照

新协议`seed42_02eebd57592f`的Power最终round04：物理validation-best为epoch36，归一化validation-best恰好为epoch200（last）。同一次训练的只读监控显示，物理real Relative从12.866506降至12.552222（-2.443%），归一化real Relative从0.289884降至0.214094（-26.145%）；物理validation却从8.343722升至8.365493（+0.261%），因此按既定validation规则保存36没有实现错误。不能把归一化loss降低26%解释成物理Relative降低26%。

此比较没有重训或独立加载末轮权重重评；epoch36监控物理Relative与最终eval JSON相差不到1e-7。正式run只留下best模型，末轮权重已随恢复文件清理，监控也没有物理MAE/MSE/Correlation或逐株预测，故这些末轮指标和预测图目前无法恢复。源history、完整精度、哈希及变化表保存在mydata的`analysis/sampling_protocol_comparison_20261116/power_checkpoint_choice.md/.csv`。不据real改选正式checkpoint，训练与物理评测epsilon继续保持0.01。AL4PDE主流程则在每轮固定epoch完成后保存当时状态，即该轮最后epoch的模型，而非validation-best。

#### 随后补齐：同一Power最终轮的物理MAE/MSE/Corr

原末轮权重已删除，因此新增诊断脚本`scripts/analysis/replay_sampling_final_round.py`通过共享`fit_round`和通用Trainer重跑固定round04。复用原10000条训练数据、全部5000条同池validation、原始初值、归一化、派生模型seed536378203及200epoch配置。新增只读observer记录物理real指标，固定保存36/50/100/150/200epoch；未用real选checkpoint或改变生产配置。通用Trainer的`extra_callbacks=()`仅为可选诊断扩展，不另写模型专用训练器。

复现核查：四项归一化张量逐位相同；第36epoch模型参数SHA与原best完全一致；200个epoch全部既有数值history字段与原训练逐项完全一致。故下表有强复现依据，但来源仍标记为受控重跑，不声称直接找回已删除的原末轮文件。

|Epoch|real物理MAE↓|real物理MSE↓|real物理Relative L2↓|real Correlation↑|
|---:|---:|---:|---:|---:|
|36：原物理validation-best|0.276126|10.217533|12.866506|0.799023|
|50|0.233186|5.656558|12.682310|0.829627|
|100|0.187816|3.007052|12.756761|0.877653|
|150|0.172359|2.422906|12.591127|0.886759|
|200：last及归一化validation-best|0.171135|2.441152|12.552222|0.887489|

36→200：MAE下降38.02%、MSE下降76.11%、相关系数增加0.08847，而Relative仅下降2.44%。只看Relative会漏掉明显的幅值和波形改善；不能据此说所有后期学习都没有价值。反过来，MAE/MSE受较大幅值通道影响，不能单靠它们证明每个KO/近零通道也学好了。checkpoint应由预先规定的合成validation规则选择，不能依据上表real结果改选。

输出位于`/home/users/zsy/mydata/budding_yeast/analysis/power_final_round_physical_replay_20261116/`。逐epoch物理指标为`real_epoch_metrics.csv/.png`；固定checkpoint的共享评测、逐株CSV及36/200epoch图在对应`epoch_XXX/eval/`；复现证明为`reproduction_audit.json`。可选observer的无干扰回归及原有监控/共享训练回归共18项通过。仅新增权重和标量报告，没有重复保存训练轨迹或覆盖正式结果。

独立复核五个固定checkpoint的全部5000条同池合成validation，物理MAE从0.418817降至0.275642（-34.19%），MSE从128.481277降至56.005394（-56.41%），Relative从8.343722升至8.365493（+0.261%）。这说明绝对误差改善并非只出现在real上；仍不能用real临时改选checkpoint。原流程物理Relative选36符合配置，但不足以概括其它指标的进步。

126株real中，36→200的MAE/MSE/Relative/Corr分别有111/105/94/105株改善。ORI的MSE从369.2827降至85.3155，贡献整体MSE下降的约93.6%；CLB2的MSE从4.3934降至0.4045。幅值指标的改善也存在通道集中性。相反，CDH1T的平均Relative从427.7506升至443.3215，占末轮主Relative约90.6%，抵消了其他通道的部分收益。#22的MAE从0.7529降至0.3739、MSE从39.9595降至11.5114，但Relative反升1.0527→1.1658，直接说明指标可以给同一mutant不同方向的判断。

补查脚本`scripts/analysis/compare_replayed_checkpoints.py`只读固定权重和源数据，可独立于慢绘图执行；生成`validation_and_real_checkpoints.csv`、`validation_report.md`、`real_per_variable_checkpoints.csv`和`per_mutant_checkpoint_changes.csv`。`checkpoint_metrics.png`放大36--200epoch物理四指标，避免前几epoch大loss掩盖后期变化。没有借本次事后诊断改变生产best选择、训练损失或epsilon。

### 16. 2026-11-16：用户确认改用归一化validation-best

最新默认`BenchmarkConfig.validation_metric='normalized_loss'`，监控`val_loss_synth`并恢复该值最低epoch。当前主损失为Relative L2，故验证与训练使用相同criterion（smooth eps0.01、无截断），验证仅限同池固定合成validation。训练仍bf16、验证仍float32；物理sample/channel validation继续计算和记录，但不参与新best选择。主模型与辅助ensemble共用该规则，轮末评测和下一轮采样使用同一个归一化validation-best模型。

每轮创建新的DataModule，按当前累计已查询训练集拟合P逐维Z-score与Y的log1p+逐变量Z-score。固定validation的原始轨迹和ID不变，但用本轮训练统计重新编码；在该轮所有epoch内统计不再变化。checkpoint保存对应本轮完整预处理契约，不能给旧模型直接换上下一轮统计。validation/test/real及尚未查询的轨迹不参与拟合。跨轮或跨采样方法的归一化loss不能直接排名，正式精度比较继续用共同物理空间。

新协议`seed42_b5bff21cb8de`从上午`seed42_02eebd57592f`派生，唯一配置变化是validation_metric；四个输入/切分NPZ为同inode硬链接，不重新生成候选池、初始选择或holdout，也不复制轨迹。checkpoint完成/恢复身份包含实际monitor，旧物理best不会被当成归一化best接着使用；新结果根目录独立。

上午两池各8方法、共16项中，无一项五轮best全部相同。共同首轮mutation物理best/归一化best为99/100epoch，LHS为83/54epoch。模型相关方法的后续评分、选样及归一化都可能改变，完整新协议应从第一轮重跑。Random的候选顺序与采样seed不依赖模型，可复用原有每轮行ID；若只要最终分数，只需重训固定10000条末轮，若要完整五轮曲线则补训各轮。Power的固定旧10000条末轮诊断不是新选择协议的完整五轮结果。

准备与checkpoint选择审计保存在`/home/users/zsy/mydata/budding_yeast/analysis/normalized_validation_readiness_20261116/`。本次没有启动整批正式训练；旧结果保留为物理best规则的对照。

相关CPU回归41项通过，覆盖两种指标故意相反时正确选归一化best、新一轮train-only统计更新与固定validation重新编码、checkpoint恢复、只读real/test监控、八baseline各五轮、原物理指标兼容及W&B离线/完整五曲线图。没有启动GPU正式训练，也没有改变并发与内存保护设置。

### 17. 2026-11-16：全流程复查后的工程保护

新增验证预测/目标/归一化loss有限值检查，避免无效checkpoint评分拖到轮末才报错。Predictor在逆变换前检查raw预测，防止正无穷被expm1的安全clamp变成有限浓度；逆变换/float32转换后也检查。已有主流程逐epoch监控本已检查real/同池test的raw预测，本次修复不构成历史主结果遭Inf污染的证据。

EpochSummaryCallback按epoch开始时记录实际LR；history的lr对应本epoch，lr_next为调度后值。之前结束时读到的下一epoch LR容易误读；没有改变warmup/cosine公式、梯度裁剪、损失或best规则。相关两批共45项CPU回归通过。

全20万轨迹审计发现11条数值疑点（7条TEM1明显越总量、4条明显负尾），严格复算全部恢复正常。最严重mutation row38053整轨迹差异0.6810，但未被上午任何方法选入训练；11条均在candidate侧，不在固定validation/test。源池未修改，建议定点修正后再冻结正式重训。完整证据和方法学边界见[全流程审计](EXPERIMENT_FLOW_AUDIT_2026-11-16.md)。

用户随后确认额外保存每轮末epoch纯权重，best选择规则不变；末文件在state_dir/last_weights.ckpt，含本轮归一化契约而不含optimizer/scheduler恢复状态。meta记录best_epoch/last路径与哈希，完成复用会验证文件；绘图中断后无需重复训练即可完成保存/提交。26项相关CPU回归通过。归一化跨轮比较例子：同一500点恒值真值1、预测2，log空间std同为1，均值从0.2变为0.6时normalized Relative从0.82220变为4.35290，物理MAE始终1、物理Relative约1。这是按当前log1p/Z-score与eps0.01构造的解释例子，不是实际实验精度。
