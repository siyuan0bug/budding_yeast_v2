# 酵母图来源统一：第一阶段（2026-10-18）

> 后续模型迁移见 [单图迁移](GRAPH_MODEL_MIGRATION_2026-10-18.md)及
> [2026-10-19 关系图/模块方向修复](GRAPH_RELATION_MODULE_FIX_2026-10-19.md)。
> 下文保留第一阶段的范围与审计证据，不代表所有生产图分支的最新迁移状态。

## 范围与当前状态

权威变量拓扑入口为 `data/yeast_graph.py:build_yeast_graph()`。它从
`data/mutant_aware.py` 的 ODE、可达标量辅助函数、事件注册、触发条件与重置分支
提取依赖记录。矩阵均由同一批记录按需派生，不再人工维护三套新拓扑。

本文件记录第一阶段，当时**没有将现有模型切换到新图**，没有修复旧加载路径/注意力方向，没有替换
checkpoint 中的图 buffer，也没有训练、修改 ODE、数据集、采样或损失。
旧文件暂留供历史复现，不能把它们描述成已经迁移到新权威来源。

后续经用户批准，第二阶段已迁移**单一 39×39 变量图消费者**，默认 hybrid，并
增加 checkpoint 的迁移/复现策略。六关系/27 模块图仍独立，未重训。
当前生产状态请看 [模型图迁移](GRAPH_MODEL_MIGRATION_2026-10-18.md)；下文保留第一阶段审计记录。

这是潜在结构图，不是固定正负号图、统计相关图、有限时间因果响应图，也不是
完整的跨事件灵敏度（saltation）支持图。结构统一不构成 real-mutant 指标改善证据。

## 构图契约

- 39 个变量的名称、顺序和索引绑定直接来自源码，不另复制一份变量顺序。
- 始终使用 `A[target, source]`：行接收、列发送；保留有向关系。
- 解析 Python AST，**不导入或执行 `mutant_aware.py`**，不调用仿真，不使用真实
  mutant 参数字典、测试轨迹或其统计量。默认参数表也不参与拓扑提取。
- 参数保持符号，不在 WT 某个状态点取导数来决定有没有边。
- 展开源代码中的辅助函数和代数表达式；SymPy 做表达式规范化，消除可确定的
  简单抵消。它不是任意 Python/所有代数恒等式/物理可达域的完备证明器；结果是
  保守的潜在支持，某些参数置零或特殊状态可以使某条边的实际局部作用为零。
- 支持当前源码的标量运算、参数条件分支、显式状态写入及事件重置顺序。
  遇到未支持的调用、循环、状态条件 Python 分支、不一致索引/事件注册等，
  明确抛出 `GraphExtractionError`，不静默给单位阵或跳过未知公式。
- 原始方程记录与事件记录保留变量依赖、源码行和表达式。事件记录另保留方向、
  terminal 标志、触发变量和重置变量；不是把事件硬当作普通连续激活/抑制边。
- 来源 SHA-256 由参与提取的 AST 生成，覆盖 ODE、实际调用的辅助函数、状态索引、
  事件注册顺序、事件编号选择、guard、重置及事件属性；不包含真实 mutant 表。
  仅改测试表不改变来源指纹。源码内容变化会使进程内构图缓存失效。

## 同一来源的不同视图

```python
from budding_yeast_v2.data.yeast_graph import build_yeast_graph

graph = build_yeast_graph()  # 默认路径相对模块，不依赖启动目录
a_ode = graph.adjacency("continuous", self_loops="none")
a_event = graph.adjacency("event", self_loops="none")
a_hybrid = graph.adjacency("hybrid", self_loops="none")
assert (a_hybrid == (a_ode | a_event)).all()
```

| 视图 | 来源/含义 | 当前非自环边数 |
|:---|:---|---:|
| `continuous` | 展开辅助量后的连续 ODE 状态依赖 | 218 |
| `event_trigger` | 事件 guard 变量 → 该事件显式重置的变量 | 12 |
| `reset_value` | 显式重置值读取的原事件前状态变量，遵守顺序赋值 | 0 |
| `event` | `event_trigger OR reset_value` | 12 |
| `hybrid` | `continuous OR event` | 226 |

事件投影与连续图有 4 条非自环重叠，故新增 8 条，而不是“事件图总共 8 条”。
`reset_value` 的非自环数为零，不代表没有自身读取：MASS 缩放和 epsilon 调整
包含自依赖。原始 guard/reset 记录不会因为投影后某些边重合而被丢弃。

自环策略显式选择：

- `physical`（默认）：只保留源码中的自身依赖；连续图当前共 254 条边，含 36 个自身依赖。
- `none`：去除对角线，便于比较非自环边。
- `all`：显式添加神经网络记忆自环；这不是额外发现的物理反馈。

输出为新分配的 NumPy bool 矩阵，修改它不会污染缓存或其他调用。记录对象不可变。
API 要求显式给出视图，不提供含糊的“默认 signed 图”。

## 正负号、间接影响与模块图边界

不编码永久 `+1/-1`。例如 CLB2 对 SWI5 同时有 MCM1 介导的合成作用与失活作用：
`d(dot_SWI5)/d(CLB2) = ksswipp*d(MCM1)/d(CLB2) - kiswi*SWI5`。
用公开默认参数/初值、只改变 CLB2 浓度做检查，局部导数从约 `+0.761049` 变成
`-0.039650`；同一条潜在结构边应继续存在。这不等于预测最终 SWI5 总量的正负响应。

CLB2 对 CDC20 没有直接连续边，但可以通过 IEP/CDC20T 间接影响它。
ORI 也可通过事件时间、Vi20 再影响 CDC20。不能因此把有限时间响应或 saltation
中的非零元素直接补成普通 ODE 边；事件接近擦边或切换序列改变时还需额外分析。

27 功能模块图、`27×39` 模块—变量映射及 `4×39` circuit mask 是不同语义，不能
当作重复的 `39×39` 邻接矩阵删掉。当前模块定义不是状态的唯一自然划分，本阶段
不重新生成/宣称验证它们。将来动态边权只能用合法输入和预测状态，不能读取测试
真轨迹或隐藏的真实参数；多关系分类与模型迁移另行完成，不把总图复制六次凑接口。

## 审计与可选导出

在 `neuralop` 环境、仓库目录下：

```bash
python scripts/analysis/dataset_diagnosis/audit_yeast_graph.py --check-legacy
```

默认只打印结果，**不写文件**，也不读取 checkpoint 或数据集。
`--check-legacy` 只比较本地旧图的支持，不验证/修改正负号。当前结果为：

- signed 旧图的非零支持等于 `continuous` 加人工自环。
- boolean 旧图的非零支持等于 `hybrid` 加人工自环，**不是无向图**。
- 六关系旧图按其构图器的 `[source,target]` 方向转置后，124 条唯一非自环边中
  114 条匹配连续支持，缺 104 条，另有 10 条不是连续直接依赖。比较不转置模型 buffer。

需要持久记录时，可在已存在的目标目录中显式导出一个新的 JSON：

```bash
python scripts/analysis/dataset_diagnosis/audit_yeast_graph.py --output scratch/yeast_graph.json
```

JSON 保存统一记录、来源和摘要，不另存三份矩阵。它是可重建的审计导出，不是
模型自动加载的新权威文件。输出必须为 `.json` 且不存在，拒绝覆盖旧证据/图文件。
构图仅有进程内缓存，无需生成第四张独立维护的二进制图。

## 旧资产的冻结与迁移

三个旧二进制文件未删除、未覆盖；生产模型仍沿用其历史加载行为。
`data/build_bio_multirelational_graphs.py` 的三个构图函数保持数值不变，兼容 HANO
等现有导入；其 CLI 不再默认重写历史 NPZ。仅允许显式历史导出到新文件：

```bash
python data/build_bio_multirelational_graphs.py --legacy-export --output scratch/legacy_reproduction.npz
```

该导出明确不是权威 ODE 图，且不能覆盖任何已有文件。后续迁移需单独处理加载
链路、视图选择、方向、关系分类和旧 checkpoint 图 buffer；不能只改文件路径就
宣称旧 checkpoint 已修复。

回归测试为 `tests/test_yeast_graph.py`，覆盖源码未执行、符号参数/抵消/辅助量、
事件方向与重置顺序、来源追踪/指纹、未知语法失败、多启动目录、独立数值 Jacobian、
同一边变号、导出保护和旧资产不变。测试目录按当前仓库规范被 gitignore 忽略。
依赖 NumPy 与 SymPy（当前 neuralop 环境已安装），未新增模型参数或训练超参数。

本次验证命令：

```bash
python -m pytest -q tests/test_yeast_graph.py tests/test_ic_wt_contracts.py --disable-warnings
```

结果：**111 passed**（图契约 50 项 + 既有 IC/WT 回归 61 项）。编译检查与
`git diff --check` 通过；没有生成新的模型评测指标。
