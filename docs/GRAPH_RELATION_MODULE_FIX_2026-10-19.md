# 关系图与模块方向修复（2026-10-19）

本阶段按用户要求修复：六关系图的结构/读取方向、DynamicPlus 的 27 模块图方向。
保留 soft attention，不改成硬掩码；未训练、调参或评测真实 mutant。
这是在 [10 月 18 日单图迁移](GRAPH_MODEL_MIGRATION_2026-10-18.md)之后的修复，
该旧文档中的“六关系/模块方向尚未处理”属于当时的阶段记录。

## 六关系图不只用于一个模型

| 注册模型/家族 | 当前关系图使用位置 |
|:---|:---|
| `bio_hetero_causal_graph` | 多关系图卷积 |
| `hetero_graph_resnet_film` | 关系消息传递残差层 |
| `hetero_graph_dualpath` | 双路径网络的图支路 |
| `dyrfo` / `dynamic_reaction_flux_operator` | 参数调制的动态图消息传递 |
| `pc_bio_operator` / `physics_constrained_biochemical_operator` | 内嵌 DyRFO 分支 |
| `biom_refiner_graph_flux`、`biom_refiner_mechanistic` | Stage 2 的 flux 图注意力 |
| HANO、NoEvent、NoFlux、Refine、Dual | 多关系 soft attention；NoGraph 没有此分支 |
| `hetero_graph_channel_hyper_fno` | **只加载/保存，目前 forward 不使用这些图**，来源报告明确标记 |

上述生产调用链都已接入新关系来源。`BioRefinerPlus` 普通单图分支不使用六关系图。
未修改归档或 `models/components` 的历史副本。

## 结构唯一依据仍是 ODE/事件源码

`data/yeast_graph.py` 在原有符号解释器中增加不可变 ODE 反应项记录：
对源码层面的加减、乘法分配拆项，展开相同辅助量，记录潜在来源变量、语法、参数键和行号。
不导入执行模拟器，不读取默认/真实 mutant 参数值，不使用轨迹，不做按测试 mutant 定向构图。

`data/yeast_relations.py` 给这些反应项附上显式、可审查的机制注释，而不是按物种名
手写连边。六类为：复合体交换、磷酸化、去磷酸化、生产、降解、活性切换。
“生产”不都等于转录，“降解”不都等于泛素化，“活性切换”不都等于 GK。
这是对源码反应项的注释，**不是从 ODE 唯一识别生物机制，也不是永久导数正负号**。

明确的速率/参数锚点见 `RATE_ANCHORS` / `PARAMETER_ANCHORS`。不识别的项，或同一
乘积匹配多个机制的项，不强行分类。关系候选还须与完整连续 ODE 支持求交，避免
拆项抵消造成的伪边。未被已分类反应项覆盖的依赖，由独立的 **untyped 补充路由**保留。

所有矩阵都是同一来源的内存派生视图，不生成第七个独立维护的二进制文件：

| 非自环结构 | continuous | hybrid |
|:---|---:|---:|
| 六类覆盖层并集 | 217 | 217 |
| untyped 补充 | 1 | 9 |
| 总支持 | **218** | **226** |
| 相对选定源码视图的缺边/额外边 | **0 / 0** | **0 / 0** |

连续图未分类的一条边是 `Vi20 → CDC20`；hybrid 另有 8 条事件新增边。
因此“六张覆盖层自身是完整 ODE 图”仍不成立；完整结构指 **六类并集 OR untyped**。
这也不是全部有限时间影响或跨事件 saltation 支持。

六类非自环计数分别为 `[34,114,29,26,44,10]`，允许一条依赖参与多个不同反应类型，
故不能把六个数相加当作独立边数。39 个网络自环只加入 untyped 一次。

已确认原先的 `CLB5 → SWI5T`、`CDH1 → CDC20T` 不再作为直接结构边；
`IEP → CDC20`、`CDC20T → CDC20` 纳入活性切换项。
统一使用 **`A[target,source]`**，不再让 `[source,target]` 数据直接进入 `A @ V`。

## 模型适配及 soft attention 保持

`models/ours/relation_graph_contracts.py` 将六类覆盖层及 untyped 注册给所有消费者：

- GCN：补充 `mean(flux_weights) * A_untyped @ H @ mean(relation_weights)`。
- DyRFO：有效邻接增加 `mean(effective_weights) * A_untyped`。
- GraphFlux/Mechanistic：有效邻接增加 `mean(flux_weights) * A_untyped`，再用原 softmax 偏置方式。
- HANO：图偏置增加 `mean(softmax(relation_weights)) * A_untyped`。

这些是无类别路线的确定性共享权重选择，不是物理通量等式。**没有新增可学习参数**：
flux head 仍为 6 维，关系变换仍为 6 组，旧权重的名字和形状保留。
关系分类含义有所修订，保留旧权重不等于已学会新结构；应重新开展合成训练/验证。

无边位置仍可通过原有 soft attention 通信；未新增 `masked_fill` 或“无边必须为零”的规则。
GCN/DyRFO 原本的图乘法结构仍保留。SoftTopologyGAT 与普通 StoichiometricGraphAttention
的注意力公式未修改。GraphFlux 当前模式只保留一个单位自环；历史模式保留旧 `+I` 行为。

旧 NPZ 的 `rel_graphs` 不再供生产模型读取；NPZ 仅保留提供 `circuit_motifs`、
`gene_bipartite` 元数据。若 NPZ 缺失，由原有命名构造函数生成相同类型的元数据，
不静默替换为六个单位阵。图本身始终由源码派生。
内部 `load_multirelational_graphs()` 现在返回 `RelationBundle`，调用 `.register(module)`
才能完整注册 typed/untyped 及来源契约；仓库内生产调用者已同步迁移。

## DynamicPlus 的 27 模块图

将 `adj[src,dst]` / `sign[src,dst]` 改为 `adj[dst,src]` / `sign[dst,src]`，与注意力行接收
列发送一致；二者同时修正。没有重画模块边、改模块列表/参数分组，或改变 soft GAT。
这是既有手写模块先验的方向修复，**不是把 39 状态图投影成已证明正确的 27 模块图**。
其他方向已经一致的模块模型未在本阶段重新生成图。

## checkpoint、来源追踪及实验隔离

固定图加载契约扩展到 relation/module：

- 默认 `graph_load_policy=current`：按配置的源码视图刷新关系 buffers、untyped 和
  DynamicPlus 模块邻接/符号 buffers，保留全部已保存的学习权重，输出替换警告。
- `graph_load_policy=checkpoint`：保留保存的原矩阵，不擅自转置。老六关系文件缺少
  untyped buffer 时只将补充路线置零；缺少来源元数据则标 `legacy_unknown`。
  重新保存这种历史模式后，下一次仍能识别其历史图语义。
- 已有新元数据但缺少图 buffer、维度/数值错误、来源字段损坏，都会报错，不能借
  `strict=False` 吞掉错误。只有明确新增的 untyped/来源字段允许兼容旧文件。

“复现”仅指保存图及对应图层行为，不会撤销之前的 IC/WT 修复、承诺全模型 bug-for-bug 复现。
`mutant_aware.py`、旧 `.npy/.npz`、旧 checkpoint 及旧指标均未覆盖。
`graph_report()` 分别报告 variable/relation/module，包含实际来源、方向、支持边数，
并区分 ChannelHyper 的“加载但未使用”，而非只靠配置推断成功。
关系来源指纹覆盖 ODE 内核、规则版本/锚点及实际项注释；模块图指纹覆盖模块/边列表与方向。

CLI `--graph_view continuous|hybrid` 同时作用于单图和关系图，默认 hybrid。
训练自动分层目录新增 `rterms1`（关系图）及 `mdir1`（DynamicPlus 模块方向）标签，
不与修复前自动目录混用。显式输出目录仍需用户自行选择新的路径。
未改变既有 `tune.py` 使用 real 指标选优的问题；在改为 synthetic-only 目标之前不要用于本任务。

## 验证方式

```bash
# 只读审计，旧图对比针对冻结资产，不是当前模型的关系图。
python scripts/analysis/dataset_diagnosis/audit_yeast_graph.py --relations --check-legacy

# 回归只使用符号源码、人工输入/权重及公开 WT，不读取真实测试数据。
python -m pytest -q tests/test_relation_module_repair.py tests/test_graph_model_migration.py \
  tests/test_yeast_graph.py tests/test_ic_wt_contracts.py
```

覆盖全部关系模型注册别名的双视图来源、前反向；错误旧边移除和新边方向；未知/歧义项
补充；六维学习权重严格加载不变；旧注意力公式逐值重放、新元数据回环；27 模块保存
buffer 的迁移/复现；坏状态拒绝；无边 soft attention 仍可通信；旧资产未改；CPU 混合精度和 double。
本次结果：**219 passed**（新增关系/模块回归 61 项，单图迁移 47 项，原图来源 50 项，
IC/WT 61 项），约 41 秒。14 条警告均为测试刻意触发的旧图刷新/旧无图分支兼容提示。
只读审计再次确认 hybrid 总支持 226，旧冻结 NPZ 仍是历史 124 条并集，不再用于生产连边。
编译及 `git diff --check` 通过。没有新的 real-mutant MAE/MSE/relL2/corr，不声称 badcase 已改善。
