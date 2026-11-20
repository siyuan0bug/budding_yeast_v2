# 主动学习模块整理与兼容说明

`utils/al_callback.py` 从迁移前的 2931 行缩减为 16 行兼容入口。实际实现位于 `utils/active_learning/`；`coreset_maxdist` 未实现；Fission 已开放五种通用策略，见 [Fission AL](FISSION_ACTIVE_LEARNING.md)。

## 职责和调用

| 模块 | 负责什么 |
|---|---|
| `registry.py` | 15 种策略、4 个别名、ensemble 要求、补采方式和支持系统的唯一清单 |
| `engine.py` | 初始化、调用策略、执行仿真、追加数据的统一入口 |
| `contracts.py` | 配置、只含候选 ID 的选样计划、查询批次的数据约定 |
| `pool.py` | 参数、初值、名称、完整覆盖字典和突变元数据的行对齐管理 |
| `prediction.py` / `features.py` | Ensemble、输入敏感性、轨迹特征和共享投影 |
| `oracle.py` / `fission.py` | 各系统自己的候选/oracle、初值协议、多进程 ODE、超时和过滤 |
| `dataset.py` / `io.py` | 冻结归一化统计量、张量/标签追加、NPZ/CSV 保存 |
| `backends.py` | 实际 bmdal 选择器及局部 NumPy 2 兼容 |
| `strategies/` | 选样计算，返回计划，不执行 ODE 或修改训练集 |

```text
train.py 训练模型
  → engine.select → 策略返回 SelectionPlan
  → engine.query → oracle 执行真实 ODE
  → dataset 追加数据与标签 → io 保存
  → 下一轮冷启动训练
```

公共用法：

```python
from budding_yeast_v2.utils.active_learning import ActiveLearningEngine

engine = ActiveLearningEngine(strategy="al4pde_random", al_pool_dataset="pool.npz")
engine.setup(dm, device="cuda", exp_dir=exp_dir)
# train.py 在每轮训练之后提供当前模型；ensemble 策略还需提供独立训练成员。
n_added = engine.select_and_append(model, dm, device="cuda", exp_dir=exp_dir)
```

CLI 仍是原 `train.py` 及原 `--al_strategy` 名称；CLI 可选项与 ensemble 训练需求直接读取 registry。原 `al4pde_setup`、`al4pde_select_and_append`、`set_ensemble_members` 和 `train_acquisition_ensemble` 接口继续保留。

## 候选和数据约定

- 候选参数、初值、名称、覆盖字典、元数据由一个 CandidatePool 持有，删除候选时统一处理全部字段。
- 候选 ID 在池缩减后不变；SelectionPlan 保存有序 ID、池身份和版本。旧版本计划、重复候选及不同策略的计划在仿真前拒绝。
- `single_batch` 与 `refill_valid` 沿用各策略原规则，指数预算、随机抽取顺序和失败时轮次计数不变。
- QueryBatch 区分已消费候选和成功入库候选；配额提前结束后尚未消费的任务仍按原规则处理。它不等同于完整的仿真成本账本或 AL 断点恢复状态。
- 参数向量不能替代完整 overrides；额外初值和特殊开关继续由原生物系统解释。
- 原 BIO、ABC、中性六分组、参数顺序、时间网格、NPZ 文件名和 CSV 格式保持原契约；使用原冻结归一化统计量。
- 原 real-envelope、名义通量先验及不同策略的补采差异保留。该重构不使实验自动成为测试盲态，也没有将全部项目启发式变成官方算法。

## 兼容与归档

- 旧 `from budding_yeast_v2.utils.al_callback import ActiveLearningCallback` 指向新引擎。
- 冗余 `utils/bmdal_compat.py` 已删除；审计与测试直接使用 `active_learning.backends`。归档目录内的同名文件继续支持快照的相对导入。旧 worker 导入路径可解析，新 worker 是 `oracle.py` 中的顶层函数，支持 spawn。
- 私有属性保留必要的无复制映射；原私有选样辅助方法可继续执行旧的“选样并查询”行为，但新引擎正式路径使用返回计划的注册策略。
- 新测试/插件要在真实模块上 patch，例如 `active_learning.oracle.ProcessPoolExecutor`。给旧 facade 的复制导入符号赋值，不承诺传到真实执行模块。
- `utils/archive/al_callback_legacy.py` 保留原旧轨迹池实现；`al_callback_before_package.py` 是本次拆包前 2931 行的逐字节快照，生产代码不导入归档。归档中的相对 backend 导入由同目录兼容文件解析。

`tune.py` 原来只挂载没有 epoch 钩子的 callback，会静默变成普通训练。现在明确只接受静态 HPO（`--al_strategy none`）；非 none 在创建 study/训练之前报错，不能把它视作 AL 调参已实现。

## 单独验证的修复

1. 初始化剔除训练集已有候选时，原实现没有同步裁剪 `_pool_mass_meta`；现在统一裁剪，修复后续生物学优先级的行错位。
2. 原 332 维全池预测特征可能跨轮沿用旧模型结果；现在每次正式 acquisition 重新计算。共享随机投影仍保留，避免改变 BAIT/LCMD 的特征坐标。

这两项是明确的正确性修复，可能改变受影响的后续选样；它们与“算法主体搬迁保持一致”分别验收。

## 验证入口

```bash
python scripts/analysis/audit_al_package_refactor.py \
  --output scratch/probes/al_package_2026-10-21/selection_audit.json
python scripts/analysis/audit_al4pde_alignment.py \
  --output scratch/probes/al_package_2026-10-21/official_alignment.json
```

迁移对照使用拆包前快照：15 种策略 × 3 个 seed 共 45 项，比较数值计算主体、有序候选、补采目标以及 NumPy/Torch/Python 随机状态。386 项工程回归测试通过，覆盖真实 spawn ODE、入库后 DataModule 读取、下一轮训练、标签、官方算法数值对照及 Fission 静态训练。

额外使用实际 `train.py`、PureFNO 和真实 ODE 完成 Random、Power 各两轮冒烟；Power 包含独立辅助成员训练及 checkpoint。初始目标为人工测试数据，关闭图像输出和远程 W&B，不用于解释科学性能或更新榜单。全部输出在 scratch；未改写既有生产数据或历史实验。

完整结果记录在 `scratch/probes/al_package_2026-10-21/verification.json`。

## utils 后续精简

- 删除顶层 bmdal 转发文件；实际依赖适配保留在 `active_learning/backends.py`。
- `biological_guard.py` 的未使用投影函数完整归档；生产文件仅保留原变量分组与索引，未改动损失分组、ODE 或训练后处理。历史函数测试改为显式导入归档。
- `al_callback.py` 兼容入口保留；物理计算模块、损失、指标及日志模块本轮不搬迁。

本次精简后重新运行 390 项测试，全部通过（含历史保护函数测试、AL 实际 spawn/训练/入库以及 Fission 训练评测）。另逐项核对 43 项变量和索引定义不变，归档快照的 bmdal 导入正常。测试日志：`/tmp/utils_cleanup_tests.log`。

Fission 扩展后，完整回归 420 项通过。对照审计的快照加载路径已改到 archive 命名空间，避免顶层 bmdal 转发文件删除后的相对导入失效；快照本身未改动。
