# 模块边界与复用规则

2026-11-17补跑入口：`bash run_sampling_budding.sh extensions --directory <队列目录>`。它仅负责调度独立的固定池、dynamic、generation流程，不把训练或算法堆进入口；seed42两池共30项，0～3号卡先4后最多8，W&B offline。队列状态、实验路径、适配边界与资源验证见[补跑说明](../docs/history/SEED42_BASELINE_EXTENSIONS_2026-11-17.md)。这里的新流程不能直接套用下方旧固定池16并发的资源结论。

更新：2026-11-14。目标是**职责独立、逻辑只实现一次**。文件按职责拆分；不以文件数量最少为目标，也不让实验入口承担模型、数据读取、训练和评测的所有实现。

当前增量默认五轮：2000→4000→6000→8000→10000，每次追加2000；epochs为100/100/100/100/200，warmup为5/5/5/5/10。每轮冷启动并按累计已查询训练集重拟合归一化；冻结初始统计仍是待讨论的对照，没有自动切换。自定义/历史协议可使用其他轮数，一次性选择方法仍只进行最终训练。

2026-11-16最新checkpoint规则：`BenchmarkConfig.validation_metric='normalized_loss'`，按同池固定合成validation的`val_loss_synth`最低保存/恢复模型。当前主损失rel_l2，因此该值是在本轮train-only log1p+Z-score空间的Relative L2；辅助ensemble也遵循同一规则。每轮新增数据后P/Y统计重新拟合，轮内固定，validation/test/real仅用该轮统计编码，不参与拟合。物理验证Relative保留在history作诊断，正式评测仍为物理空间。旧物理best配置仍可显式复现；新协议`seed42_b5bff21cb8de`复用原四个缓存和划分，不覆盖上午结果。本次仅准备，未自动启动正式重训。

用户已确认保留末epoch纯权重：共享Trainer每轮仅在最后epoch额外保存一次`<state_dir>/last_weights.ckpt`，无optimizer/scheduler/callback状态，附该轮归一化契约；完成后仍清理resume.ckpt。best继续用于评测/下一轮选择，last用于固定数据诊断。training.json记录best_epoch与last路径/哈希。新训练的保留策略进入恢复身份，历史已完成轮次不会被冒充有last；缺失/改动末权重会拒绝直接复用该完成缓存。

结果路径统一为`experiments/budding_sampling/<protocol_id>/<pool>/<method>/seed_<seed>/`，没有A/B目录层级；方法的信息访问条件记录在配置与账本中。批量启动器的`train.log`与单实验输出使用同一路径。

共享服务器负载保护（2026-11-17）：离线batch默认总16项、GPU0–3每卡最多4项，加载并发8、候选评分并发4，共用动态队列。12/16项完整规模资源诊断均退出0；16项PSS峰178.76GiB、系统最低可用522.32GiB。正式保留100GiB可用内存，不设自有内存上限；memory PSI仅记录预警。全局IO PSI>5%并同时存在严重设备拥塞持续60秒才终止本批隔离进程组，设备信息缺失回退strict20秒；CPU负载保护和GPU4–7每卡独占仍保留。每次轮询采样可用内存、PSI、设备队列及自有读写量。单项/库默认strict，batch明确选择disk_correlated/available策略。资源测试为warm-cache的2epoch与候选评分，不是完整五轮吞吐保证。详见[资源复测](../docs/history/BUDDING_REL_L2_BATCH_2026-11-15.md)。

默认方法为`random/topk/power/max_dist_pred/coreset_ll/lcmd_ll/bait_ll/vessal_penultimate`，两池共16项；可显式`--methods`选用子集。当前主模型仅使用来源池的验证集选checkpoint，仍报告两个来源池的测试指标；新协议为`seed42_18bf9a3289fa`。旧显式协议缺少`validation_scope`字段时保留原双池验证语义，不会静默改变历史运行。

```text
数据生成 data/systems/ ──→ 完整数据池
                               │
当前流程 sampling/benchmark/ ──┼──→ 算法 baseline/ 或 ours/
  protocol.py  实验规则         │       select(context, budget)
  runner.py    阶段编排         │              ↓ 只返回选择结果
  rounds.py    当前流程数据适配 ├──→ query.py 查询权限与访问账本
                               ├──→ training/runner.py 通用训练
                               └──→ evaluation/ 公共推理、指标和报告
```

## 每层只负责什么

| 模块 | 职责 | 不在这里实现 |
|---|---|---|
| `data/systems/` | mutation/LHS生成、ODE、实际初值、标签 | 模型训练、选样打分 |
| `data/dataset_archive.py` | NPZ/清单读取、指定行读取、哈希和数据引用 | 选择什么样本 |
| `data/normalization.py` | 一套统计拟合、编码、逆变换 | 如何划分训练集 |
| `sampling/contracts.py` | 输入视图与选择结果的数据结构 | I/O、训练、仿真 |
| `sampling/registry.py` | 方法名称、能力、实现入口 | 算法公式 |
| `sampling/baseline/`、`sampling/ours/` | 从允许的信息计算选择结果 | 写训练集、执行训练 |
| `sampling/query.py` | 检查可查询ID、访问真值、记账 | 采集分数 |
| `sampling/benchmark/` | 当前完整池实验协议与编排 | 通用梯度循环、复制模型 |
| `training/runner.py` | Lightning执行、训练日志、checkpoint与恢复 | pool、mutation、采样策略 |
| `models/`、`models/yeast_lit_module.py` | 模型结构、训练step、优化器/调度器 | 候选池生成 |
| `evaluation/checkpoints.py` | 统一checkpoint加载和分批推理 | CLI参数、实验日程 |
| `evaluation/metrics.py` | 指标定义 | 文件路径或选样 |
| `evaluation/reporting.py` | CSV/JSON报告、轨迹图 | 模型训练 |
| `experiments/curves.py` | 训练与实验性能曲线 | 采样公式 |

`train.py`和`eval.py`保留命令行入口及相应编排，通过公共服务完成工作。新完整池流程不导入这两个CLI模块。`offline_fno.py`已删除；所有新轮次使用原有`YeastLitModule`，保存标准Lightning `best.ckpt/last.ckpt`，可由同一个checkpoint loader读取。短暂存在过的`yeast_sampling_fno_v1`文件仅保留读取兼容。

## 下一版通常改哪里

| 变化 | 修改范围 |
|---|---|
| 新增采样算法 | 一个算法文件＋`registry.METHODS`一条注册 |
| 改轮次、每轮预算、epochs、warmup | 当前流程配置；不改算法和训练服务 |
| 改初始选点、split或标签权限协议 | 当前流程的`protocol.py`/编排；保持算法接口 |
| 新增一套流程，如不同的数据追加方式 | 独立流程模块/数据适配，复用训练与评测服务 |
| 换模型 | 选择已注册模型；LL方法还需该模型支持隐藏特征接口 |
| 换候选池来源 | 数据层生成/读取适配；无需复制选择器或训练器 |

新增算法示例：

```python
# sampling/ours/my_rule.py
from ..contracts import SamplingResult

def select(context, budget):
    # 只使用context允许的信息；这里实现评分并返回budget个源池行ID。
    ids = ...
    return SamplingResult(ids)
```

在`registry.METHODS`登记`Method('my_rule', family='ours', implementation='my_rule:select')`即可；不必修改当前`offline.py`分派、流程runner、训练或评测代码。原有方法保留已有适配入口，不改其选择规则。

通用训练接口为`fit_model_round(model_factory, prepared_datamodule, ..., monitor=...)`，不接收候选池或采样算法。当前`benchmark/rounds.py`只负责把已授权的训练ID清单与固定验证域适配成DataModule。未来流程可以提供自己的DataModule。若未来改变任务本身的输入输出定义，例如产生池外新输入，则应显式扩展契约，不能把非法输入伪装成现有池ID。

诊断工具可通过可选的`extra_callbacks=()`接入只读观察器或保留事先指定epoch的权重，仍复用相同训练服务与合成validation checkpoint规则。示例为`scripts/analysis/replay_sampling_final_round.py`：冻结已有轮次的数据及模型seed，逐epoch记录real物理MAE/MSE/Relative L2/Correlation；real只作诊断，不参与选择best。默认正式实验没有额外观察器，既有输出/权重清理行为保持原配置。

## 数据权限、历史兼容与运行

A轨道选择器只能访问候选输入、模型预测和已查询训练轨迹；B轨道另提供有访问账本的真标签reader。验证和real测试不交给算法。参数逐维Z-score、变量log1p后Z-score均按当前训练集拟合。实际初值从`y_init`读取，响应探针使用最终`overrides_json`，不从救援前提案重建。

旧在线流程隔离在`archive/online/`，仍支持Fission及显式历史重放；原归档保留。`_compat.py`集中维护旧模块别名与pickle路径。新代码使用当前服务；不再新增逐文件转发shim。

```bash
cd budding_yeast_v2
bash run_sampling_budding.sh list
bash run_sampling_budding.sh batch --dry-run --methods random topk power lcmd_ll bait_ll
SKIP_PREPARE=1 bash run_sampling_budding.sh batch --methods random topk power lcmd_ll bait_ll --gpus 0 1 2 3
```

当前共享协议：`/home/users/zsy/mydata/budding_yeast/benchmarks/budding_sampling/seed42_c0d0d9a0169c/`。两源池未重生成，新旧共享协议的split与初始ID完全一致。轮次JSON只引用源池，勿删除源数据。W&B强制本地offline。历史在线重放显式设置`SAMPLING_MODE=online`。

正式训练尚未启动；接口测试不代表已证明收敛。方法来源和待适配项目见[Benchmark方案](../docs/history/BUDDING_SAMPLING_BENCHMARK_PLAN_2026-11-13.md)。

当前完整池实验采用紧凑输出：每轮model.ckpt/history.csv/loss.png/dataset.json/selection.json/eval，恢复与辅助模型集中到_state；源池轨迹不复制。详见[输出规范](../docs/history/SAMPLING_OUTPUTS_2026-11-14.md)。

2026-11-15：逐epoch只读测试监控位于`training/monitoring.py`，由当前流程传入同池test/real引用；训练执行、监控、采样评分独立。曲线共用原TrainingCurveLogger，恢复五条主线；checkpoint仍仅依赖validation。


2026-11-16当前离线默认：smooth Relative L2 eps0.01，按同池归一化validation loss选best，保留末epoch纯权重；数值修复视图对应协议seed42_a866aef5d988，输入和划分不变。`run_sampling_budding.sh`默认总8项、每卡2项、重加载8项；依据有效剩余内存及压力控制，不设置自有内存额度。资源诊断已完成，尚未启动该新协议的正式五轮重训。


2026-11-17并发诊断新增独立`resource_slot('scoring', ...)`与`SAMPLING_SCORING_CONCURRENCY`（仅资源探针的候选评分当前启用，正式runner尚未包裹）；数据槽与评分槽互不占用。资源监控记录自有`read_bytes/write_bytes`及同步设备`diskstats`，不能将全局PSI直接归因本批。`profile_sampling_concurrency.py --io-policy disk_correlated --scoring-concurrency 4`为显式诊断：全局IO压力需联合持续严重设备拥塞才停止，缺失设备数据回退strict；只在该进程/子进程设置IO准入warn，保留100GiB余量、memory PSI、CPU和有界槽位。生产默认strict未改，不得仅凭诊断启动就提升正式并发。实时状态resource_profiles/scoring_limited_12_16_20261117/status.json。


## 新输入生成流程（2026-11-17）

`bash run_sampling_budding.sh generate --help`进入独立generation工作流，支持`random_generate/oed_temporal/ogas_round_projected`。算法核心在sampling/baseline/oed.py、ogas.py；生物只读适配、存储、cohorts和流程分别在sampling/generation/。训练/归一化/best/日志/评测继续共用公共服务，初始轨迹和累计快照均引用已有源，新增轨迹只写一次。W&B显式offline，默认4个solver worker，资源尚需正式规模验证。OGAS是批次冷启动/合法提案投影适配，不能称原版异步在线复现。详见[实现说明](../docs/history/RECENT_BASELINE_IMPLEMENTATION_2026-11-17.md)。新模块增加源身份，历史结果目录不改写；旧seed42结果要按已保存源码身份审计复用，不能删除校验后在原目录覆盖。


## OGAS在线任务适配（2026-11-17）

`generate --method ogas_async_projected`已接入持续训练/CPU后台条件DDPM/异步ODE/replay；对照为`random_async`。原分轮`ogas_round_projected`保留且明确v1，新在线默认依据OGAS v2。生成与源存储在sampling/generation/，算法仍集中sampling/baseline/ogas.py，训练/损失/评测共用公共服务。在线初始train统计冻结；原固定池五轮逐轮refit不变。新结果独立`budding_generation/<protocol>_online<hash>/...`，新增轨迹只保存一次，恢复状态另存。33项CPU回归通过，未用GPU、未正式并发。论文公式、必要适配及未复现的速度限制器等见[核对说明](../docs/history/BASELINE_FIDELITY_AND_OGAS_ONLINE_2026-11-17.md)。


## 四类回归采样扩展（2026-11-17）

新算法核心在`sampling/baseline/{regression,peaks,eracs,low_rank,hggs}.py`，工作流独立在`sampling/dynamic/`和既有`generation/`。入口统一`run_sampling_budding.sh`：`dynamic --method peaks_regression/eracs_regression`、`run --method low_rank`、`generate --method hggs_projected`。动态流程使用公共fit_round/optimizer/loss/eval，初始统计冻结；原固定池五轮逐轮refit不变。新数据仅保存source/ID/权重，HGGS新增轨迹只存一次。评分可用训练侧真值，按实际读取/历史曝光计费。回归reference-kernel、Batch-OMP、gradient-energy与合法投影的差异见[实现与公式](../docs/history/REGRESSION_SAMPLING_BASELINES_2026-11-17.md)。没有加入默认正式批次，未完成生产资源验收前不套用16并发。


## 2026-11-20：统一结果目录与迁移

所有当前采样流程共用`utils/sampling/layout.py`生成`<模型>/<mutation或lhs>/<算法>/<可读超参>/seed_<主seed>`，结果根为`/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling`；固定池、持续训练、生成式不再分不同主目录。Transport/Closure的pool与generate变体分名，旧HGGS归入hggs_legacy、旧OGAS归入ogas_legacy。哈希保留在科学元数据，目录不再生成哈希名。

共同输入/划分已移到`shared/pool100k/seed_42`；生成seed仍各42，实验主seed及派生流在`seed_manifest.json`。不同训练配置可使用`configurations/<可读超参>`协议视图，输入和划分以硬链接共用，无第二份轨迹。已有协议的科学指纹和内容不改；共享池或划分设置发生变化须使用新的共享数据名称，不能覆盖。

最新冻结协议全部65项完成实验已同盘rename迁移；原配置/权重/指标字节不改，旧路径留兼容软链接，避免历史绝对引用失效。35,907个常规文件inode/大小/mtime逐项一致，341个链接及231份逐轮数据引用可访问。更早其他协议保留历史状态，没有混入本次最新65项。队列归入batches，汇总归入summary；直接打开[结果导航](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/index.html)或[路径索引](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/experiments.csv)。迁移审计`batches/layout_migration_20261120/verification.json`。

统一选定方法批量入口（预览，不启动训练）：

```bash
cd /home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2
bash run_sampling_budding.sh extensions --suite all \
  --directory /home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/next_seed42 \
  --gpus 0 1 2 3 --dry-run
```

共23项=八个baseline×两池16项＋三个ours的实际适用池/流程7项。去掉dry-run才会启动；已有同名实验身份不匹配会拒绝覆盖。此轮没有启动正式训练。


### 2026-11-21：采样脚本更名及历史目录归档

生产入口已更名为`run_sampling_budding.sh`和`run_sampling_fission.sh`，调度器、README、文档和测试引用同步更新，不更改各系统现有算法/训练设置。Budding仍以`extensions --suite all`预览/启动统一任务清单；Fission仍为原有流程，更名不代表已接入Budding的新完整池协议。

当前结果根的三个旧seed哈希目录不是重复空目录：02e约9.921GiB，bb0约9.795GiB，a866残留约0.576GiB。已同盘移至`experiments/archive/sampling_legacy_paths/`；原位置为兼容链接，不能在仍有历史绝对路径引用时当垃圾直接删除。归档22,663个常规文件inode/大小/mtime保持，65项新结果及旧路径均可访问；未删除实验数据，也没有释放约20GiB空间。清单`batches/layout_migration_20261120/archive_cleanup.json`。


### 2026-11-21：按授权精简历史结果并移除旧哈希入口

已删除三个归档批次中的旧权重和预测图（21,126个文件，逻辑大小21,434,413,555字节，约19.96GiB），保留配置、指标JSON/CSV、训练历史、日志与数据文件。当前结果371份JSON中的路径已重定位，272个符号链接已规范化，25份完成记录的受影响校验值同步更新；修改前原文保存在`batches/history_prune_20261121/before`。模型权重、轨迹与评测值不更改。三个旧seed哈希根入口已移除；历史归档保留精简记录，旧目录不再用于访问当前结果。

根目录现在仅batches/fno/shared/summary，后续其他模型与fno平行。Budding统一入口的run/dynamic/generate/batch以及extensions默认均写入该根；显式覆盖output/results-root或选择历史online模式仍可指定其他位置。清理审计`batches/history_prune_20261121/verification.json`，当前汇总`summary/all_current/report.md`，导航`summary/index.html`。汇总工具已改为按迁移清单发现真实新路径，不再依赖已移除的旧路径。
