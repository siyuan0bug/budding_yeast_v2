# Fission 的图、训练和物理扩展

Fission 与 Budding 使用独立 SystemSpec。Fission 为16变量/58参数/300分钟/500点/20 real；
`train.py` 从 DataModule 读取维度，`--system auto` 默认自动选择，亦可指定 `--system fission`。
checkpoint 保存 system_id；跨系统数据或参数顺序不匹配会报错。

## 标签保持原始 Fission 规则（2026-10-21）

Fission 标签严格继承 `data/fission_random_sampler.py`，不接入 Budding 的六分类或临界标记。
`systems/fission/phenotypes.py:assign_labels` 与原函数的语法树一致，阈值不变：
无分裂时按 `max_MPF > 0.5` 区分 M-phase/G2 Arrest；最后出生质量原始尺度 `<0.625` 为 Too Small，
继续沿用1050分钟续跑得到的出生质量；分裂时 `RDNA<0.99` 为 cut。
`Pattern_A`/`Pattern_B` 仅在通过生物判定后按事件分裂次数 >=2/==1 赋予。
原代码中无分裂或不存活时 pattern 为 `None`，NPZ 中为字符串 `'None'`，仍原样保留，
不会补成 Budding 的 Pattern_C。原函数的 Pattern_C 分支保留，但当前前置返回使其不会被正常触发。
数值失败、无效参数、质量溢出仍按原状态处理，标签类别与求解有效性分开。
新旧文件读取保持 `labels`、`pattern_labels` 原值，不从轨迹重判 Fission 标签。
标签函数/阈值、导出与新旧读取、实际WT/突变体仿真及31个归档源码哈希的专项回归通过。

## W&B 和实验流程核查（2026-10-21）

W&B 的标签首先用于监控图分组与标题；可选分类辅助头/标签损失才用于训练监督。
Fission 原 pattern=None 仍在 NPZ 中保存为 `'None'`，DataModule 内部训练目标改为 `-1`，
不再错误地默认成B。分类交叉熵跳过这些无标签行，全批无标签时返回可微零值；
基础轨迹损失仍作用于全部样本。原有效 A/B/C 的0/1/2编码保持不变，批大小1及列向量目标受支持。

Fission 监控按每个 train/test/real 拆分实际存在的 `(BIO, pattern)` 组合各选一个代表，
不使用 Budding 的112/87/100或15/22/20/42/76编号，不将缺失C类回退成A/B样本。
图标题显示 system、BIO、原pattern；Fission没有Budding六分类。
变量名由Fission自身16个名称提供，时间轴为300分钟。
Budding保留既有监控样本选择，仅在标题补充BIO与中性六分类。
可视化仅由主进程在启用W&B时执行，跳过sanity check，结束后恢复原模型模式。

`config.json` 和W&B配置保存已解析的 `system_id`、16变量名、维度、时长和各拆分原标签计数。
默认W&B项目为 `fission_yeast_v2`，显式 `--project` 优先；Budding默认项目不变。
`--no_wandb` 沿用现有“离线W&B”语义，仍记录本地图片与指标，本次验证没有上传日志。
HPO trial也保存系统和标签摘要，study绑定系统与数据指纹，拒绝混合不同数据集/系统；
有已完成trial但没有新系统契约的旧study需使用新的study名称。
`run_tune.sh` 允许通过环境变量设置MODEL/LOSS_TYPE/AL_STRATEGY/DATA_DIR/DATASET_TAG，
例如Fission必须选择已支持的模型；默认Budding批量实验矩阵不自动转换为Fission实验。

验证：共285项相关回归通过：102项数据/标签/Fission、181项监控/损失/运行契约，以及2项Budding损失/归档源码检查。
另实际完成CPU一轮训练、W&B离线图、checkpoint恢复、20 real/8 synth评测、一张16变量图，
以及一次实际HPO objective/最佳权重重载/real物理指标计算。冒烟数据来自20条既有real和复制扩充的80条合成轨迹，
只验证接口，不构成独立训练样本或精度实验。W&B离线保存15张图片，20 real的原BIO/pattern逐行对齐，
均未出现Budding六分类。产物位于 `scratch/probes/fission_monitoring_2026-10-21/`，核验摘要为该目录的 `verification.json`。
未验证线上W&B上传、多GPU或长时训练。
Fission仍要求已支持模型与 `--al_strategy none`；不支持的生物结构/AL配置会在训练前报错。

## 图

`build_yeast_graph(system_id='fission')` 读取 Fission 自己的 constants/ODE/event simulation，
传播标量中间量和分支依赖，不执行源码、不读取 real 规则。
连续图67条非自环边；包含 START、SFINISH、DIV 及 kS 隐含参数切换的 hybrid 图75条。
实际浓度重置与 kS 对 Mik1/Cdc25 导数的影响分别记录。
`build_yeast_relations(system_id='fission')` 使用 Fission 的反应参数注释，
六类关系与未分类通道的并集严格等于选定图。方向统一 A[target,source]。
Budding 的218/226条边与来源指纹不变；这些图表示潜在依赖，不表示全局固定正负调控。

DataModule 和通用模型图接口在16维时选 Fission 图，不再要求人为关闭图。

## 训练、评测与模型范围

已验证 FNO1d、UNet、DeepONet、NeuralODE、ResNetFiLM、CrossFNO、GraphResNetFiLM、UltimateFNO
在16/58维上的前向/反向。完整允许集合见 `models/yeast_lit_module.py:FISSION_MODELS`。
Budding 专用 HANO、BioRefiner 生化池/事件结构没有被强行改成16维，选择这些结构会明确报错。
Fission 主动学习 oracle/候选池策略尚未迁移，训练和调参当前使用 `--al_strategy none`。

评测输出 `all_variables` / `all_16_vars`，不套用 Budding 的 core_9。
变量名和绘图网格按系统选取。空合成测试拆分输出 count=0、指标 null，不再 torch.cat 空列表。
Budding 的 `all_39_vars/core_9_vars` 历史指标键保持兼容。

```bash
python train.py --model cross_fno --dataset_name /path/to/fission.npz \
  --system fission --loss_type pinn_residual --modes 24 --width 32 --n_blocks 2 \
  --al_strategy none --max_epochs 100
python eval.py --ckpt /path/to/fission.ckpt --dataset_name /path/to/fission.npz
```

## 物理约束

`utils/fission_physics.py` 把 canonical NumPy RHS 的 AST 转换成可微 Torch 运算，
没有再复制一份16条方程。数值稳定的 sigmoid 与 NumPy 的 Y_switch 数学等价。
在 RHS 处将相对 mass 乘1.4048，返回导数时除回；训练目标和图的变量顺序不变。
RHS计算将5个分数活性通道投影到[0,1]，数据误差仍作用于完整预测。

kS 不在58维导出参数中，不能根据单点 RDNA<1 就认定 kSmax：cut 株可从 RDNA<1 开始，
canonical 仿真仍从 kSmin 开始。损失以完整目标轨迹重建事件历史：
RDNA向下重置对应START，达到0.99999对应SFINISH，按最近事件确定 kS；初始为 kSmin。
单点 RHS 接口要求显式 kS，避免猜测隐状态。该重建针对当前500点采样的生成协议；
若另造极稀疏时间网格、在相邻点间漏掉完整S期，则需另存事件历史，不能假设可还原。

PINN、PINO、Hybrid PINO、HybridAll 共用系统分派与四种归一化解码。
事件掩码使用目标的重置、阶段改变及阈值邻域；decoupled排除mass/RDNA及有数值微重置的MPF/preMPF。
START/SFINISH/DIV 的可微重置返回新状态和 kS 参数，与 canonical 求解器分支数值对照。

## 验证边界

52项Fission专项通过，包括20组real参数×两种kS的RHS数值对照、事件重置、梯度、
四类物理损失×四种归一化、8种架构、实际文件加载、训练和checkpoint恢复。
465项相关测试通过（包含专项的早期49项），并修正了两个旧事件测试遗漏参数统计量的测试设置。
实际CLI完成一次Fission训练、20 real评测与一张16变量图；这是接口冒烟，不是预测精度实验。
