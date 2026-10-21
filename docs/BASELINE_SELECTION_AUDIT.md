# 当前标准 baseline 清单与适配边界（2026-10-21）

本文件取代上一轮“保留旧实现兼容”的方案。按用户最新要求，baseline 目录仅保留选定标准骨干、
必要的参数/完整初值/查询时间输入输出适配，以及这些模型需要的数值核心和来源许可证。
已删除调制改造版、重复入口、普通历史窗口预测模型和未采纳模型，不再保留 pure / ref / 1d 别名。
原始论文仓库 `../baselines/` 不修改。

## 1. 保留的 18 个具体实现

MLP 和 ResNet 分别计数，其余对应先前采纳名单。所有代码入口见 `models/baselines/__init__.py`。

| 训练入口 | 实现文件 | 保留的标准机制 | 默认参数元素数 |
| --- | --- | --- | --- |
| `mlp` | `mlp.py` | 普通 MLP | 26,663 |
| `resnet` | `resnet.py` | 普通残差 MLP | 47,463 |
| `unet` | `unet.py` | 一维双卷积、池化、转置卷积及 skip 拼接 | 10,849,383 |
| `kan` | `kan.py` | efficient-kan B-spline 边函数及 base activation | 263,680 |
| `transformer` | `transformer.py` | Tau_BNO 多头 self-attention、FFN、RMSNorm、位置编码 | 171,687 |
| `mamba` | `mamba.py` | 官方 Mamba-1 输入依赖 dt/B/C、因果卷积与选择性 scan | 242,343 |
| `deeponet` | `deeponet.py` | branch / trunk 内积 | 203,111 |
| `fno` | `fno.py` | 时间域 Fourier kernel + pointwise kernel | 434,855 |
| `wno` | `wno.py` | 最粗近似/细节的位置相关系数核 | 2,119,335 |
| `mwt` | `mwt.py` | Legendre 多小波、递归 A/B/C 和逆变换 | 1,204,727 |
| `gnot` | `gnot.py` | 官方异构归一化注意力、self-attention、几何专家 | 1,039,407 |
| `transolver` | `transolver.py` | 官方 slice / slice self-attention / deslice | 123,767 |
| `lno` | `lno.py` | 官方 PhCA 编码/解码及潜空间 vanilla attention | 200,199 |
| `neural_ode` | `neural_ode.py` | 学习条件向量场并积分 | 26,663 |
| `sc_fno` | `sc_fno.py` | 完全相同的 FNO 骨干，区别在灵敏度损失 | 434,855 |
| `bregman_fno` | `bregman_fno.py` | Bregman 双状态后激活更新 | 434,855 |
| `lamo` | `lamo.py` | 官方 latent slice / Hydra / deslice 机制 | 362,855 |
| `konode` | `konode.py` | Koopman 模式 -> 演化向量场权重 -> 状态演化 | 1,163,451 |

参数统计：V=39、P=141、width=64、n_blocks=4、modes=24、T=500；小波 haar、level=3；
MWT k=4；其余未单独指定的参数采用类默认值。复数参数按 `numel()` 的复数元素计 1。
这些数字不是匹配容量后的论文配置，也不表示公平调参或性能排名。

原方法固有的注意力、几何专家、门控状态空间等予以保留；不添加参数调制、生物图先验或自创跨注意力。
MLP / ResNet 是普通架构对照，不声称对应某篇神经算子论文。

## 2. 删除与名称调整

- 删除 `pure_fno.py`、`pure_wno.py`、`fno1d.py`、`resnet_film.py` 以及旧的 `mwt1d`、`latent_lno`、`transolver1d` 简化实现。
- 删除 iTransformer、DLinear/NLinear/Linear、Autoformer、Informer 系列、Flowformer 系列、Reformer 系列、Flashformer 系列、FEDformer、Pyraformer。
- 删除未采纳的 GCGNet、NOBLE、ICON、NCFlow、WDNO、BAMPNO、G-FNO、DPOT、CNO、Galerkin、Transolver-3、DMD/Koopman 解码器、PNO/MPNO、IGANet 等旧实现。
- 删除旧 `unet1d`；补回普通一维 U-Net，统一入口 `unet`。这是标准架构的时间轴适配，不宣称逐行复现 RealPDEBench 的扩散/注意力 U-Net。
- 补回标准条件回归 `kan` / `transformer` / `mamba`；不恢复旧 `vanilla_transformer` 入口。来源与配置见 [新增模型说明](OPERATORFREE_BASELINES_2026-10-21.md)。
- Bregman 本次只保留已采纳的 Bregman-FNO，删除额外 Bregman-WNO。
- 删除旧 `reference_operators.py` / `configuration.py` 和失效的 components 导入转发文件；标准文件直接命名为 `fno.py`、`wno.py`、`mwt.py`。
- `MODEL_REGISTRY` 中 `wno`、`lno` 现在指标准基线。独立 `models/ours` 中原图/跨注意力方法使用 `cross_wno` / `cross_lno`，不冒充基线；ours 源码未删除。
- 默认训练模型为 `fno`。运行脚本中的活动旧 FNO 配置已改为标准名称，已删除模型的旧示例不再作为运行建议。

这是有意的结构和接口清理：旧模型 checkpoint 不自动转换为新标准架构；旧实验指标不得改名后归属于新模型。

## 3. 统一任务接口

输入 `X:(B,V,2,T)`：`X[:,:,0,:]` 是重复初值，`X[:,:,1,:]` 是查询时间；参数 `P:(B,D)`。
输出 `(B,V,T)`。除 DeepONet / NODE / KoNODE 自身接口组织方式外，普通输入投影使用
`[全部 y0, 全部 P, t]` 的拼接，不给任何基线额外的未来轨迹。

- 变量是输出通道，不折叠成相互独立的 batch 样本；每个输出能接收全部初值。
- 时间特征必须由调用方显式提供，有限、递增且所有变量一致；FNO/WNO/MWT 另要求均匀查询。
- KAN / Transformer / Mamba：全部初值与 P 广播到时间点，拼接查询 t；不读取未来真值。KAN 为逐点映射；Transformer 和 Mamba 沿时间轴处理。保留师兄原 backbone 的规范化/FFN；未添加条件调制。
- U-Net：双卷积、MaxPool、转置卷积和 skip 拼接；右侧补齐后裁回原时间长度。不引入调制、注意力或图模块。
- FNO：RealPDEBench FNO3d 的谱/点卷积框架减为一维时间，尾部零填充 8；不引入物理图。
- WNO：对应师兄仓库一维 WNO 的 coarsest low/detail 核结构；以 ptwt 替换 pytorch_wavelets，使用 symmetric 边界。训练/调参入口传入数据实际 T 和 wavelet_level。不同小波后端边界系数没有宣称已逐项完全相同。
- MWT：保留来源滤波器和层级递归；时间长度延拓到下一 2 的幂并裁回，width 必须为 k 的倍数，输出投影扩展为 V。
- GNOT：官方 DGL 图批处理替换为密集时间点张量；参数和全初值是两个输入分支，时间是几何坐标。归一化 attention 和几何专家核心保留。
- Transolver：直接保留官方不规则网格 attention 和 residual block；时间点用作一维节点，输入/输出投影适配 V 和 P。
- LNO：直接保留官方 PhCA 和 vanilla latent attention。分支输入包含全部条件和时间，避免重复常量值经 PhCA 后退化为时间常量。
- LaMO：采用官方 latent-slice/Hydra 分支。Hydra 未融合路径的投影、双向 dt/B/C、卷积、shift、D skip、gated RMSNorm 和输出保留。compiled SSD 改为等价 PyTorch 离散递推，未用简化固定衰减 SSM 冒名替代。小潜序列能在 CPU 测试，不声称拥有原 CUDA 内核吞吐。
- KoNODE：保留 `(y,w)` 增强动力学、由 Koopman w 产生各层随时间变化的权重 theta；已知物理 P 作为向量场的静态输入。为使代理映射确定，随机 w0 改为由 (y0,P) 的输入映射初始化；不把物理 P 与网络 theta(t) 混为一谈，不人为限制谱为负。
- Neural ODE：普通状态/P/时间拼接的向量场，积分使用实际查询时间；每个样本不同时间网格时逐样本积分。输出层小随机初始化避免初始化时完全忽略 P。
- Bregman-FNO：选择官方允许的无 optional channel MLP、无 norm 的配置；保留 zero spectral/local 更新和激活前 skip 累加，无额外参数模块。

## 4. SC-FNO：标准 FNO 加参考灵敏度损失

SC-FNO 不是给 FNO 加一个名为 sensitivity 的调制层。`SCFNO` 继承标准 `FNO`，同权重输出完全相同。
训练损失在 `utils/losses.py` 注册为 `sc_sensitivity`：

`MSE(Y_pred,Y) + sensitivity_weight * mean_k MSE(dY_pred/dP_k,S_k)`。

使用可微 JVP 计算每个选定参数的完整轨迹导数，训练参数能够收到灵敏度项的梯度。
参考导数必须为固定全部初值时的 `d(normalized Y)/d(normalized P)`，不能拿另一个样本或
当前网络自己的预测导数当真值。参数子集 K 如少于 D，须在论文中明确说明。

普通轨迹 NPZ 不含这些真值，不能凭空补零或悄悄训练为普通 FNO。提供独立求解器敏感度方程/
可靠数值差分等得到的参考文件后，`data/sensitivity.py` 在 DataModule 中校验并附加训练监督。
状态重置、分岔和不可微事件附近的参考生成须单独处理；当前接口拒绝非有限训练目标。

NPZ sidecar 的必要键：

| 键 | 内容 |
| --- | --- |
| `dy_dp` | `(N,V,T,K)`，同源数据原行序，规范化坐标偏导 |
| `parameter_indices` | `(K,)`，不重复且有效的整数参数索引 |
| `data_sha256`、`params_sha256` | `dm._dataset_contract` 中对应哈希 |
| `preprocessing_sha256` | `data.sensitivity.preprocessing_fingerprint(dm)` |
| `initial_conditions_held_fixed` | 布尔 True |

指纹包含当前 Y/P 统计量、norm_type 和 s_c，拒绝用不匹配的归一化导数继续训练。
验证/测试仍只预测轨迹、计算轨迹误差，不读取测试 Jacobian。
当前 sidecar 合并只支持 `--al_strategy none --rollout_type one_time`；在线新仿真样本没有相应参考时不能直接追加。

## 5. 来源与可追溯性

- 新增 KAN / Transformer / Mamba：师兄 `OperatorFree`；KAN 与 Mamba 官方来源固定 commit、路径和 SHA256 在 `_upstream/operatorfree_sources.json`，适配及数值验证见新增模型说明。
- GNOT：`../baselines/GNOT/models/mmgpt.py`，保留 normalized heterogeneous attention 和 geometric experts。
- Transolver：`../baselines/transolver/Physics_Attention.py` 与标准 benchmark 的 irregular-mesh residual block。
- LNO：`../baselines/LatentNeuralOperator/LNO-PyTorch/ForwardProblem/module/model.py`。
- LaMO：`../baselines/LaMO/model/Latent_SSM_unshared.py` 和 `mamba_2D_v2.py`。
- KoNODE：`../baselines/KoNODE/main.py` 中 `ODEFunc`、`update_w`、`assign_weights`。
- Bregman：`../baselines/bregmano/neuraloperator/neuralop/layers/fno_block.py` 的后激活 dual-state 分支。
- SC-FNO：官方 `fno_ode /ODEs/ODE1/ode_FNO_IG_EQ_auto_loss.ipynb` 的 Jacobian/reference loss。
- FNO / MWT：RealPDEBench 对应模型及师兄 Tau_BNO 的 MWNO；WNO 来自师兄 WNO 并追溯原方法。

原数值核心摘取放在 `_upstream/`；仅移除不需要的图打包/可选 backend 依赖，einops 排列写成等价
Torch reshape/transpose；LaMO SSD 与归一化采用未融合数学实现。来源声明及已有授权文本在 `licenses/`。
`BASELINE_SELECTION_AUDIT.json` 记录包装层、保留数值核心和官方来源文件的 SHA256。
“标准骨干任务适配”不表示未经修改的官方酵母实现，也不表示所有原实验超参数相同。

## 6. 运行

```bash
export PYTHONPATH=/home/users/zsy/projects/budding_yeast_cell_cycle:$PYTHONPATH
cd /home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2
python train.py --model fno --dataset_name /absolute/path/to/dataset.npz --loss_type hybrid_mse_rel --width 64 --modes 24 --n_blocks 4 --al_strategy none
python train.py --model unet --dataset_name /absolute/path/to/dataset.npz --loss_type hybrid_mse_rel --width 32 --n_blocks 3 --al_strategy none
python train.py --model wno --dataset_name /absolute/path/to/dataset.npz --loss_type hybrid_mse_rel --width 64 --n_blocks 4 --wavelet haar --wavelet_level 3 --al_strategy none
python train.py --model sc_fno --dataset_name /absolute/path/to/dataset.npz --loss_type sc_sensitivity --sensitivity_path /absolute/path/to/sensitivity.npz --sensitivity_weight 1.0 --al_strategy none
python scripts/analysis/audit_baseline_alignment.py --output docs/BASELINE_SELECTION_AUDIT.json --require-ready
```

输入信息、数据划分、归一化和调参预算仍须公平控制；有已知方程/额外灵敏度监督时应单列信息成本。
当前平滑 Neural ODE / KoNODE 不自动包含酵母事件重置；轨迹仿真真值与实验测量应区分。
没有重训整套论文 benchmark，没有把历史模型指标移植到当前实现。


## 7. 本次验证结果

此前 289 项 CPU 回归通过；本轮加入 U-Net、KAN、Transformer、Mamba 与实际 CLI 链路验证，最新完整回归 **371 passed**，结果见 [链路核查](BASELINE_CLI_CHAIN_AUDIT_2026-10-21.md)。
覆盖完整输入依赖、18 个模型的 39 变量/141 参数/500 点输出、梯度、标准注册、严格权重重载、
图与归一化/checkpoint 回归、Fission 共享接口，以及 MWT/GNOT/Transolver/LNO/KoNODE 的实际来源计算对照。
LaMO SSD 递推与独立稠密公式的前向/梯度一致，Hydra 未融合路径与官方类接稠密 SSD 后端输出一致。
SC-FNO 对照解析函数验证 Jacobian、网络参数梯度、同权重 FNO 输出及 sidecar 数据/归一化绑定。
本轮新增模型测试含 CPU/CUDA BF16 前反向（小规模），以及官方 Mamba 前向和梯度对照；完整回归结果见新增模型说明。
频域/小波测试含 BF16；未将 CPU 结果推断为完整 GPU 吞吐验证。没有重跑正式论文 benchmark。

测试命令：

```bash
python -m pytest tests/test_operatorfree_additions.py tests/test_baseline_cli_chain.py tests/test_standard_baselines.py tests/test_reference_operators.py tests/test_graph_model_migration.py tests/test_runtime_contract_repairs.py tests/test_fission_extension.py -q
```
