# 运行链路修复与保留决策（2026-10-19）

> **历史阶段说明**：本文件记录先前授权的运行修复，保留当时的 all-synthetic 拟合范围和 350 项结果。
> 用户随后已授权新训练切换 train-only，当前策略与后续回归见
> [TRAIN_ONLY_NORMALIZATION_2026-10-19.md](TRAIN_ONLY_NORMALIZATION_2026-10-19.md)，不得将下文旧拟合边界当作当前默认。

本轮只实施用户批注明确授权的工程修复。未启动正式训练、HPO 或 real-mutant 重评，
没有修改/覆盖 NPZ、旧权重或存档指标。坏例收益尚未通过重训验证。

## 1. 范围与明确保留项

- 保留 Optuna 最终优化 **real-mutant 物理 MAE** 的目标；这是用户明确选择的测试集指导开发，不称严格盲测。
- 用户补充确认：**al4pde 筛选规则保留，只修工程链路**。real-envelope、nominal-flux 阈值、
  选样算法、real 去重/采样范围均未调整，不能声称本轮移除了测试信息来源。
- 不调整合成数据的切分、重复样本分组、统计量拟合范围；仍为 `all_synthetic_legacy`，不是 train-only。
- 不改生成器随机 seed 管理、corr/relL2 定义、104 号干预输入、通用第二系统入口、图软注意力。
- 不重写 iInformer/iFlowformer/iReformer/iFlashformer 的重复架构，仅修复超参数传递。

## 2. 统一的归一化与 HPO 物理指标

权威函数为 `data/normalization.py`，DataModule、PINN/PINO、HANO、al4pde、eval/tune 共用。

| 模式 | Y 编码 | P 编码 |
|---|---|---|
| log1p（默认） | Z(log1p(max(Y,0))) | Z(P) |
| linear | Z(Y) | Z(P) |
| asinh_z | Z(log1p(max(Y,0))) | asinh(Z(P)) |
| asinh_scale | Z(asinh(max(Y,0)/(2*s_c))) | Z(log10(max(P,1e-8))) |

默认均值/标准差仍在全部合成样本拟合（不含 real），沿用标准差 epsilon 和恒定参数处理；
本轮未迁移为 train-only。asinh_scale 的 s_c 沿用原 90% 分位数/下限规则，缺失则报错，不能默认为 1。

默认变量逆变换为 `max(expm1(z*std_y+mean_y),0)`，参数逆变换为 `p_norm*p_std+p_mean`。
旧 tune 仅做 `z*std+mean`，实际上得到 log(1+Y)，不是真实浓度。人工目标 10、预测 20，
旧方法会得到约 0.6466 的差，而物理 MAE 应为 10。
现在 `tune.evaluate_real_metrics` 调用 DataModule 完整逆变换，仍返回 real_mae 作为 objective。
旧 HPO 分数与修复后的物理 MAE 不可混合比较，需使用新的 study。
study 记录 `objective_contract=real_physical_mae_v1`；恢复含旧分数但缺少此标记的 study 会报错，
不删除旧 study，不自动把旧数值当成新口径。

三种物理损失子类显式接收/转发 `norm_type/s_c`，解决 on_train_start 的 TypeError；
恢复 checkpoint 时物理损失、HANO 与 WT 适配器绑定同一套统计量。
HANO 不再写死 log1p 及线性 P 解码。al4pde 预存池、即时池和标注入库都使用同一编码。
未更改损失公式、权重、事件策略或最终评估指标。

### 历史对照的正确解释

下列数据来自原实验 `BioRefinerPlus/hybrid_mse_rel` 的 JSON，不是本轮重新评估。

| 模式 | real MAE | real MSE | pooled relL2 | pooled corr |
|---|---:|---:|---:|---:|
| log1p | 0.086912 | 5.184606 | 0.151820 | 0.972276 |
| asinh_z | 0.101784 | 5.511863 | 0.171317 | 0.965456 |
| asinh_scale | 0.092190 | 2.424230 | 0.205412 | 0.957554 |

log1p 在此组 MAE/relL2/corr 最好，但 **MSE 最低的是 asinh_scale**。
asinh_z 和 log1p 的 Y 变换相同、P 变换不同，不能误称只改变 Y 的单因素试验。
历史文档的 Min-Max、standard_zscore 标签和错误 MSE 已在归一化专题纠正；不推广为全模型最优结论。

## 3. checkpoint 的预处理契约

新 checkpoint 顶层增加 `preprocessing_contract`，schema_version=1，包含：

- norm_type、mean_y/std_y、p_mean/p_std、asinh_scale 所需 s_c、物理 t_max；
- 原始数据 shape、参数维度、可用的 param_names/variable_names；
- 已加载原始轨迹、参数、名称、Pattern 标签的 SHA-256（视图计算，不再解压一遍 NPZ）；
- 原 NPZ 的 train/val/test/real 索引、split seed、明确的拟合范围标记。

新评估恢复这些统计量而不是重新拟合，拒绝不同数据、不同顺序、不同切分、不同变换及损坏统计量，
也检查持久化 HANO/物理损失统计量是否与预处理载荷冲突。
独立 eval 和 Lightning load_from_checkpoint 都恢复物理解码；缺统计量的物理反归一化会报错。
eval 从 checkpoint 恢复全部受支持构造超参数，而不是把 ODE 精度、损失系数等恢复为默认值。

兼容旧 checkpoint：归一化优先级为新载荷 > 模型超参数 > 旧 DataModule 超参数 > 显式选择/默认。
显式选择与已记录变换冲突时拒绝；旧权重缺完整契约时发出警告，仍须用原数据集重建统计量，
无法追溯认证其完整输入契约。不会把旧权重冒充已具备新契约。

AL 会改变有效训练集，但这里冻结的是**原数据文件拆分**及预处理拟合来源。
这不是完整 AL pool/优化器/采样 RNG 的断点恢复功能，不声称只靠该载荷可以重放 AL 选样全过程。
第二系统 SystemSpec、时间解析和输入维度通用化仍按用户要求延后。

## 4. 自回归和基线构造

Refiner/HANO 训练与推断使用同一个滚动状态序列。Refiner 的 coarse 监督沿同一状态序列，
分类监督使用初始输入的 pattern logits；HANO 的事件和物理辅助头逐步收集。
使用张量列表堆叠，避免可变视图污染反向图。one_time 分支保持原行为。
沿用原自回归“相对局部时间网格 + 取输出索引 1”的协议，不增加时间干预控制输入。

通用基线显式检查构造签名，将 width 映射到 d_model/width 等，将 n_blocks 映射到
e_layers/n_layers/num_layers 等；无对应结构参数的模型不会凭空增加宽度/层数。
支持的实际 kwargs 存入 `resolved_constructor`；新版本保存 `constructor_contract_version=1`。
旧 checkpoint 缺版本标记时按历史版本 0 构造，以保持实际 128/3 等旧尺寸，绝不硬塞进新 192/6 网络。
层级新运行添加 ctor1；HANO/物理损失/自回归运行添加 runtime1，避免与旧目录混淆。

## 5. 求解失败、截断与超时

`mutant_aware.simulate_mutant` 与 al4pde 使用的 MASS 模拟器新增：

- 验证 solve_ivp.success/status、数组形状、有限性、单调时间和正常终止覆盖时长；
- 达到事件循环上限但未到 tf 的轨迹拒绝保存；最终插值禁止向未积分时间区间外推；
- POSIX worker 主线程 5 秒计时保护，并恢复调用方已有 alarm；
- 生成器父进程 35 秒无完成结果时终止自身 executor 的 worker 并中止生成，不导出静默残缺数据；
- al4pde 在线标注按原候选顺序消费结果，有序消费和配额提前停止均带 watchdog 清理。

父进程是**无进展 watchdog**，不是对排队中的每个 future 计算 35 秒。
worker 的 5 秒是逐次模拟限额；Python 信号无法可靠抢占所有 native 阻塞时由父进程兜底。
当前保护/验证针对用户提供的 POSIX/Linux 环境，不能冒称已验证 Windows 或非主线程直接调用。
本轮没有重采样或重写已有数据。已有 126 条 real 的先前重积分检查未发现失败，不据故障注入断言其损坏。

公开 WT、人工 C0=0.5、人工 mdt=150 三个控制，与修复前 HEAD 模拟函数比较：
210 min/500 点最大绝对差均为 **0**，标签相同。
ODE/事件图 kernel SHA 保持 `ed66b9d3b6e0dfa2edd43a3bf4f732e3416a0f3d5cbfdad2e34d8f394f34284b`，
连续/混合非自环边仍为 218/226。

## 6. 未修改项的解释

重复指同一个 `mutant_aware_pure_random_100000_210m_500s.npz` 中不同名称/行号对应等价输入和完整标签。
既有完整哈希审计记录：合成 val 1,633/10,001、合成 test 1,604/10,001 与 train 重合，real 有 3/126。
不是把其他数据集误并入；不代表所有行重复。对应计数见原 badcase 审计及 duplicate_inputs_summary.json。
标准流程应先切分、仅在 train 拟合统计量，之后冻结用于 val/test。本轮按要求只解释并保留原实现。

seed 问题：局部 RandomState(seed) 控制基因/模式抽取，但辅助函数仍调用全局 np.random。
只设局部 seed，不会重置另一条随机流，因此同样的 --seed 也可能得到不同连续参数。未修生产 RNG。

常数目标通道的 Pearson corr 数学上未定义；当前实现返回 0，即使预测与目标完全相同。
pooled corr 将各 mutant 的时间序列拼接后计算，macro corr 则先逐 mutant 计算再平均，二者不能互换。
人工两个样本各自恒为 1/10、预测完全正确时，pooled corr=1、逐样本 corr 都是 0。
因此低 corr 可能包含常数通道口径影响，不等价于真实波形很差；指标定义仍未修改。

## 7. 验证

回归包含：四种编码的独立数学公式/往返、未改变拟合范围、物理损失启动及 warmup 后梯度、
HPO 物理 MAE=10 的人工控制、Lightning 严格恢复及旧 checkpoint、数据指纹冲突、基线新旧构造、
自回归一致性、失败/截断/NaN/事件卡死注入、worker 与父进程超时、有序候选消费、al4pde 两种入库路径。

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q -p no:cacheprovider \
  tests/test_runtime_contract_repairs.py tests/test_relation_module_repair.py \
  tests/test_graph_model_migration.py tests/test_yeast_graph.py \
  tests/test_ic_wt_contracts.py tests/test_mechanistic_bifurcate_pino.py
```

最终回归：**350 passed，15 个预期的旧图迁移/旧 checkpoint 契约警告，31.72 秒**。
这些测试不等于完成了所有模型×损失×AL 组合审计，
不构成 real-mutant 性能改善或第二生物系统已可运行的证据。
