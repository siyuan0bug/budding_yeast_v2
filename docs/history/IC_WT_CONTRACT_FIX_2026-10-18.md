# IC 布局与 WT 参数表示修复（2026-10-18）

## 范围

按用户要求，只修复 IC 读取和 WT 差分/对数特征的表示错配。
没有修改图加载、图连接、ODE、数据生成/采样、损失或训练配置；没有启动训练。
没有使用任何 real-mutant 参数、轨迹或评估指标来设定变换。

这不是新的预测模型，注册名及已有可训练参数形状均不变。
新增 `models/ours/input_contracts.py` 是输入契约工具，可训练参数量为 **0**。

## 1. IC 从原始输入提取

DataModule 的实际布局为 `(B,V,2,T)`：`x[:,:,0,:]` 是重复的初值，
`x[:,:,1,:]` 是时间坐标，**不是初值的一阶差分**。
直接 reshape 后通道交错为 `[IC_0,time,IC_1,time,...]`，不能取前 V 个通道当初值。

`extract_initial_state(x,V)` 返回 `(B,V,1)`，支持：

| 输入 | 初值读取 |
| --- | --- |
| 标准 `(B,V,2,T)` | `x[:,:,0,:1]` |
| 旧 IC-only `(B,V,T)` | `x[:,:,:1]` |
| 明确交错展平的 `(B,2V,T)` | `x[:,0::2,:1]` |

最后一种不代表按块拼接的 `[全部IC,全部time]`。模型内部都传入**原始 x**，
避免把 3D 旧接口的块布局与 4D 展平的交错布局混淆。
原有卷积特征 `x_in` 的布局不改，以限制本次修复范围。

已修复：

- `EquilibriumRelaxationStream`：正确取初值；松弛式采用等价形式
  `x0 + (y_eq-x0)*(1-decay)`，保证 t=0 精确等于 x0，避免浮点减加抵消。
- `BioRefinerAttractor` / `BioRefinerAttractorPlus`：平原流传入原始 x。
- `BioRefinerAttractorPlus` / `BioRefinerCausalReach`：容量门控的基线 x0 从原始 x 提取。

## 2. WT 特征统一在物理参数空间计算

`ParameterInputTransform` 保存与 DataModule 对应的 P 变换，
WT 来自公共 ODE 默认参数表，顺序仍为原有 141 参数顺序。

| `norm_type` | P 输入表示 | 比较前的处理 |
| --- | --- | --- |
| `log1p` / `linear` | P 的线性 Z-score；`log1p` 在此指 Y 的变换 | 按 p_mean/p_std 还原；WT 同步编码以保证零差分 |
| `asinh_z` | `asinh(Zscore(P))` | sinh 逆变换，保留 std≤1e-5 的常量坐标处理 |
| `asinh_scale` | `Zscore(log10(max(P,1e-8)))` | 10 的幂逆变换，保留常量坐标及 DataModule 的 [-8,6] 截断 |
| 显式 `physical` | 未归一化的 P | 不变换；仅供明确传入物理参数的调用 |

WT 参考与输入共用相同编码/解码规则，包括不可逆的地板与常量处理。
因此编码后的 WT 输入，其物理差分与对数特征为精确零。
静态因果模块保留原来的 signed-log 形式；原本使用普通 log-ratio 的 Dynamic/Latch
模块仍用 log-ratio，只修正计算所在的参数空间。

只有这些**机理特征**使用物理参数。路由器、全局融合、粗预测和细化器等学习分支
继续收到原来的模型参数表示；没有把整个网络输入改成物理 P。

已接入的五类模块：

- `CausalMutantAdapter`：覆盖 CausalBase 及其 Curvature/Pyramid/Graph/Plus/主别名。
- `CausalReachabilityGate`：覆盖 CausalReach 和 AttractorPlus。
- `bio_refiner_dynamic_causal.TimeVaryingCausalAdapter`。
- `bio_refiner_latch_causal.TimeVaryingCausalAdapter`。
- `bio_refiner_dynamic_plus.TimeVaryingDynamicAdapter`。

本次没有顺带修改 Dynamic/Latch 中其他物理事件、状态门控的建模公式。
当前 DataModule 的统计量拟合拆分也没有改变；这是另一个需要单独处理的评估协议问题。

## 3. 训练、评估与手动调用

`YeastLitModule.setup/on_train_start` 的现有统计量注入流程新增参数变换配置；
`eval.evaluate_model_on_datamodule` 在第一次预测前执行同样的配置。

绕过上述入口、直接使用模型时，应显式传入**与输入 P 对应**的统计量：

```python
model.set_parameter_stats(dm.p_mean, dm.p_std, norm_type=dm.norm_type)
prediction = model(x, p_normalized)
```

只有输入本身确实为物理 P 时才能显式使用：

```python
model.set_parameter_stats(norm_type="physical")
```

缺少统计量时会明确报错，不再将标准化 P 静默当作物理 P。
不能通过设置 `physical` 来绕过真实归一化输入所需的统计量。

## 4. Checkpoint 兼容性

新 checkpoint 在每个适配器的 `parameter_transform` 下保存
`p_mean`、`p_std`、`norm_code`，不增加可训练权重。

旧 checkpoint 缺少这三个 buffer 时，仍允许严格加载所有旧权重：
已由 DataModule 注入的统计量会保留；否则保持未配置状态，预测前必须注入。
仅缺少部分字段的新格式 checkpoint 仍会按损坏状态报错，不静默补全。

**能加载不等于复现旧输出。** IC/WT 特征语义已经纠正，旧权重在错误语义下训练；
修复后复评属于不同前向行为，不能与旧输出混称同一基准，更不能冒充重训收益。

## 5. 验证

仓库根目录执行：

```bash
/home/users/zsy/.conda/envs/neuralop/bin/python -m pytest -q tests/test_ic_wt_contracts.py
```

测试覆盖人工标识 IC、所有门控/平原分支、五个适配器、四种归一化模式与显式物理输入、
物理参数变化与 DataModule 逆变换一致性、零 WT 机理特征、原归一化学习输入保留、
缺统计量失败、checkpoint buffer 持久化/旧格式严格加载、float64/CPU autocast、
六个注册模型的训练入口前向反向、独立评估入口，以及现有审计探针。

本工作区验证结果：**61 passed**（CPU，包含下述三份历史权重检查）。

另用现存 Attractor、AttractorPlus、CausalReach 三份旧 checkpoint 做严格加载与
人工输入前向冒烟检查；不读取 real-mutant 样本，不计算新排行榜指标。
这些兼容测试在缺少历史权重的其他机器上会跳过。

`tests/` 是仓库现有 `.gitignore` 忽略目录；本次遵守其存放规范，不修改忽略规则。
历史诊断结果保留，当前 WT 审计探针新增 `legacy_*` 对照列，避免继续把旧公式当成修复后的实现。

**本次验证的是实现正确性与接口兼容性，不是 MAE/MSE/relL2/corr 已经提升。**
