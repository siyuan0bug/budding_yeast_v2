# 单变量图模型迁移（2026-10-18）

> 本文件保留 10 月 18 日的阶段记录。10 月 19 日已继续修复关系图结构/方向及
> DynamicPlus 模块方向，soft attention 保留；当前状态以
> [后续修复说明](GRAPH_RELATION_MODULE_FIX_2026-10-19.md)为准。

## 已落地的契约

本阶段在[图来源统一](YEAST_GRAPH_SOURCE_CONTRACT_2026-10-18.md)之后执行。
`models/ours/graph_contracts.py` 是模型适配入口，默认直接调用
`data/yeast_graph.py:build_yeast_graph()`，不再加载历史 signed/boolean 二进制图。
没有更改 ODE、采样、损失或数据集；没有启动训练、HPO 或 real-mutant 评测。

- 默认 `graph_view=hybrid`：226 条非自环边；`continuous`：218 条。
- 始终 `A[target,source]`，行接收列发送，不对称化、不反向。
- 每个变量添加一个网络记忆自环；总图有 265/257 个非零。
  行归一化层不再执行已有对角线上的 `A + I`，避免自环被加两遍。
- 默认是无符号潜在支持。旧 `adj_act/adj_inh/adj_signed` 名称仅为兼容 ABI；
  默认正数表示 attention affinity，不表示物理激活。
- 原来的 soft attention 仍为 soft attention；不将全部有限时间影响硬裁成 ODE 直连。
  UltimateFNO 原有 mixer 的硬 mask 保留，未新增到其他模型。
- 未提供图而变量数不是 39、图维度错误、出现 NaN/Inf、显式路径不存在时直接报错，
  不静默回退单位阵。启动目录不再影响图来源。
- 显式 `adj_matrix`、UltimateFNO 的 `causal_matrix` 或 `--adj_matrix_path` 是
  **覆盖/消融输入**，要求调用者保证 target/source 方向，不被认证为源码拓扑。
  文件内容被保存进 checkpoint hyperparameters，恢复不依赖原文件路径。

## 迁移范围及例外

| 家族 | 本次状态 |
|:---|:---|
| BioRefinerGraph/Plus/Bifurcate、MassAction、Attractor/AttractorPlus、CausalReach | 单变量图已迁移 |
| CausalGraph/CausalPlus、DynamicCausal、LatchCausal、DynamicPlus、EventHybrid 及注册别名 | 其中的 39 变量图已迁移；27 模块图未改 |
| CrossFNO、WNO、LNO、ChannelHyperFNO | 单变量图已迁移；原先没有图分支的旧权重需按下述策略加载 |
| GraphResNet、GatedGraphResNet、MultiScaleDiffGraph、DualPath、BioStateSpace、UltimateFNO | 已迁移；多跳矩阵由同一来源确定性派生 |
| GraphFlux/Mechanistic | 仅无 flux 的单图后备分支迁移；常用 flux 分支仍用旧六关系图，**不能算全部迁移** |
| Hetero 系列、DyRFO、PC-Bio、HANO 有图变体 | 六关系分支仍为 legacy，构造时明确警告，未把单图复制六遍冒充关系分类 |
| 非图基线、27 模块图、27×39 映射、circuit masks、历史 `models/components`/可视化副本 | 未替换 |

新增单图加载契约本身没有可学习参数。曾经未启用图的 Cross/WNO/LNO/Channel 层，
现在会启用其原有设计的可学习 affinity 参数（每层 `39×39=1521` 个）；
已有图分支的这些模型及其余单图家族不因契约新增可学习参数。

迁移回归还发现 EventHybrid 缺少向内嵌 adapter 转发参数归一化统计量的入口，
已补 `set_parameter_stats`，不改变归一化公式或新增参数。

## 旧 checkpoint：迁移和复现必须显式区分

`graph_load_policy=current`（默认）：使用本次构造所选择的新图，刷新明确声明的
固定图 buffers，并保留已保存的可学习权重。若旧文件里的单位阵被替换，会警告。
这不是旧模型的严格复现，也不代表重新训练完成；特别是 UltimateFNO 旧权重的
非对角项可能为零，换 mask 不会凭空学会新通路。

`graph_load_policy=checkpoint`：严格保留 checkpoint 保存的固定图。旧文件没有
来源元数据时标记为 `legacy_unknown`，绝不冒充已迁移。若历史 Cross/WNO/LNO/
Channel 同时没有图矩阵和对应 affinity 参数，复现策略将该分支置零，保留无图
行为；迁移策略仅为这类明确缺失的图参数使用既定的 0.1 初始化，不吞掉其他
缺失学习权重。图元数据损坏/非有限矩阵会报错，包括 `strict=False` 路径。

模型每个适配层保存 schema、来源种类、视图、SHA-256 指纹；
`YeastLitModule.graph_report()` 报告实际已加载的来源，而非只读配置宣称成功。
训练/评测输出 `graph_provenance.json`；原始 checkpoint 文件不会被修改。
历史审计/冻结重放脚本显式选择 `checkpoint`，继续核查旧实验而不偷偷替换图。

## 入口及使用

`train.py`、`tune.py` 的 `--adj_matrix_path` 现在真正传给模型；生产 DataModule
不再拥有/注入模型图。DataModule 的旧 `use_adj_matrix` 接口仅保留兼容暴露，
未指定路径时也派生同源图，不再静默忽略坏路径。
训练的自动分层目录增加 `gsrc1-hybrid` / `gsrc1-continuous` 标签，和历史目录区分。
自定义图仍应使用独立实验目录/标签，避免不同覆盖矩阵之间碰撞。

```bash
# 从仓库目录执行，使用 neuralop 环境；以下只是使用示例，本阶段未运行训练。
python train.py --model biom_refiner_plus --loss_type hybrid_mse_rel \
  --dataset_name mutant_aware_pure_random_100000_210m_500s.npz \
  --graph_view hybrid --graph_load_policy current --no_wandb

# 加载旧权重后，按当前源码图推断；必须与历史指标分开记账。
python eval.py --ckpt /path/to/best.ckpt --dataset_name /path/to/dataset.npz \
  --graph_load_policy current --graph_view hybrid --save_dir /new/output/migrated

# 复现保存的历史图；不要同时传 graph_view 或矩阵覆盖。
python eval.py --ckpt /path/to/best.ckpt --dataset_name /path/to/dataset.npz \
  --graph_load_policy checkpoint --save_dir /new/output/historical
```

评测 CLI 默认输出目录为 `eval_graph_current` / `eval_graph_checkpoint`，不再直接
使用历史 `eval/`。显式给出的输出路径仍由使用者管理，不能指向要冻结的旧证据。

**盲态提醒**：本次只给 `tune.py` 接通图配置，没有认可其调参目标。
当前该脚本仍用 real-mutant MAE 选优，且有不完整的反归一化路径；
在完成 synthetic-only 目标整改前不要运行它用于本任务。训练/AL 的既有 real
监控也不能用于选择配置或早停。正式比较应预先确定合成训练/验证协议，最后再评测真实集。

## 下一阶段图问题（已定位，尚未在本次混改）

1. **六关系方向与分类**：旧构图为 `[source,target]`，若直接用于 `A @ V` 或
   `einsum('nm,btmc->btnc',A,x)` 则消息方向反了；旧并集 124 条非自环边不能当
   完整 ODE 图。应先审定关系记录、引入单独的方向/来源契约，再迁移 typed 分支
   和其 checkpoint，而不是简单转置之后就宣称结构完整。
2. **27 模块方向不一致**：DynamicPlus 构造 `adj[u,v]/sign[u,v]`，而 SoftTopologyGAT
   的 `softmax(QKᵀ) @ V` 需要行是接收端；Causal/Reach/DynamicCausal/Latch 则存
   `[target,source]`。需统一模块契约并对旧保存 buffers 采取显式迁移/复现策略。
3. **正负号与有限时间传播**：潜在结构不等于恒定正负号；后续若引入状态/参数
   条件边权，只能使用合法输入和预测状态。不能访问真实测试轨迹，不能拿某个
   real-mutant 参数定向采样。soft bias 不等于严格因果模型。

验证只使用源码、人工输入及只读历史权重，不读取真实测试数据做调参。
没有新的 MAE/MSE/relL2/corr 结果，不能声称 badcase 已改善。

## 验证结果

```bash
python -m pytest -q tests/test_graph_model_migration.py \
  tests/test_yeast_graph.py tests/test_ic_wt_contracts.py
```

结果：**158 passed**（迁移 47 项、图来源 50 项、IC/WT 61 项）。覆盖单图注册别名的
构造/来源路由、适用模型的人工前反向、无重复自环、路径/形状校验、旧单位阵两种
策略、新 checkpoint 回环、缺失旧图分支、来源损坏、文件不被修改及训练目录隔离。
GraphFlux/Mechanistic 只验证其后备单图路由，未将常用 typed 分支算作迁移完成。
旧 Attractor/AttractorPlus/Reach checkpoint 严格加载并以人工输入前向通过。
从 `/tmp` 启动 train/tune/eval 的 `--help` 均识别图选项；编译及 `git diff --check` 通过。
警告是测试预期的旧图替换/legacy 六关系提示，不是忽略加载错误。
