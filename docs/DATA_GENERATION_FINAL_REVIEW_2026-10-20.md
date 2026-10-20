# 数据生成最终复查与 real_mutants 更名

## 更名

两个新模块已分别改为：

- `data/systems/budding/real_mutants.py`：唯一的 Budding MA real 规则、名称和碰撞比较。
- `data/systems/fission/real_mutants.py`：20 条 Fission real 规则（含 WT）。

生产代码、测试、审计脚本和整理文档已更新引用。新目录中不再保留 `benchmarks.py`。这次只改变新模块名字；迁移前 44 个 Python 文件及旧生成入口全部保持原样，real 规则内容也没有改。

## 本次找到并修复的问题

先增加边界回归，复现了 9 个失败场景，修复后再运行回归。之后还补查了派生配置和来源标记。

| 问题 | 触发条件与原行为 | 修复后的行为 |
| --- | --- | --- |
| Fission 吞掉超时 | 预热或稳态标签扩展捕获普通 `Exception`，把 `TimeoutError` 当成可恢复失败，可能在期限已用完后继续求解 | 预热、主体积分和标签扩展明确继续抛出超时，由外层 worker 记录失败 |
| 救援后模式信息过期 | Motif 等候选已有 `metadata.pairs`，救援改变 KO 模式后只更新文字描述 | 同步更新机器可读的基因/模式对 |
| Twins 换碳源破坏配对剂量 | 只有 A 碰撞时，把 A 的救援结果作为 B；即使只换碳源，也可能把 B 的连续剂量替换为 A 的固定剂量 | 碳源救援只同步两侧 mdt，保留原有两侧剂量；模式改变才按两套规则重建 |
| 配对结果静默截短或错配 | 对两侧结果直接 `zip`，少一侧结果时可能静默丢行，编号不对应也未拦截 | 校验候选数、返回数和 sample_id/pair_id；失败样本仍使用共同成功掩码 |
| 源数据身份信息丢失 | `_source_rows` 未恢复 pair_id、候选 metadata 和仿真诊断 | 完整保留这些字段，旧文件缺失时采用明确的默认值 |
| 定向消融编号冲突 | 新目标从 sample_id=1、pair_id=0 开始，与共享基底已有编号重合 | 从源数据已有编号之后分配，写入时拒绝重复 sample_id |
| real 身份校验不充分 | 源文件自称 real，参数、初值和覆盖字典相互一致，但整体已偏离该 real 的标准规则 | 按名称中的真实编号检查 canonical MA 规则，拒绝伪装成原 real 的修改行 |
| 参数与元数据矛盾 | 合成源数据的 params 与 overrides_json 不一致，或初值、行数、状态不匹配 | 读取派生源数据时逐项校验参数、初值、数组行数、有限值和成功状态 |
| 失败状态可写成有限轨迹 | 上游传入有限数组，但 `status=numerical_failure`，原 writer 只校验数组 | 轨迹数据只接收 success 状态；参数池继续使用独立的无轨迹契约 |
| KO mirror 忽略 no-real | 源文件含 real，即使指定 `--no-real`，镜像仍带上它们 | 明确按 include_real 过滤，合成行及配对身份保持原顺序 |
| 分布来源标记失真 | 从 MASS 源数据做镜像/消融时，输出一律标成 mutant_aware | 镜像和 control 保留来源 profile；混合 MASS 基底与 MA 目标的消融输出标为 mixed，并记录来源；缺失来源的旧文件标为 legacy_unspecified |
| 输出路径检查过晚 | 后缀错误或带 `~` 的已有文件可能到仿真结束才发现 | 采样前展开、解析输出路径并校验 `.npz` 后缀及覆盖条件 |

`mixed`、`legacy_unspecified` 只是派生文件的来源描述，不是新增突变或采样实现。生产采样仍只有 MASS / mutant_aware 两套 Budding 规则，以及 Fission 自己的规则。

## 新旧一致性复查

本轮重新执行：

- 两套分布、参数修改及模块审计：**381/381**，报告 `scratch/probes/mutation_profiles_final_review.json`。
- 固定 seed 的采样内核及无碰撞入口调度对照：**105/105**，报告 `scratch/probes/sampler_final_review.json`。
- 全部 126 条 Budding real 和 20 条 Fission real：新旧有效轨迹逐元素相同，最大绝对差均为 **0**。日志分别为 `scratch/probes/real_parity_final_review.log`、`scratch/probes/fission_parity_final_review.log`。
- 原有 RNG、运行契约、IC/WT、图及关系图回归：**399 项通过**，15 条预期的旧 checkpoint 迁移提示。日志 `scratch/probes/original_regression_final_review.log`。
- 新生成流程及本轮边界回归：**82 项通过**；加上原有 399 项，本轮共 **481 项测试通过**。见 `tests/test_modular_datasets.py`、`tests/test_modular_final_review.py` 和 `scratch/probes/final_review_tests.log`。
- 原有 44 个 Python 文件的 SHA256 全部与迁移前清单相同，所有新模块通过语法解析，旧 `benchmarks` 导入已清理。

## 验证边界与旧版保留问题

此次验证没有发现上述修正之外的阻塞问题，但不能保证所有参数、依赖版本和运行规模均无错误。没有实跑正式十万轨迹或百万候选池，峰值内存、长时间运行和底层原生求解器卡死的全部情形仍未覆盖。

旧 MASS 的 real 初值问题、旧脚本不同的碰撞丢弃/救援政策，以及 BAMS 历史进程随机状态等仍留在原入口中，便于历史复现；不会为了让旧版输出一致而把这些问题带回新流程。新版本继续遵循此前确认的 MA 基础模型、统一救援和失败过滤政策。

新旧完整 NPZ 的字段、压缩字节，以及碰撞后的整批序列仍不保证相同。采样一致性测试如何隔离政策差异，见 [采样一致性核验](SAMPLER_EQUIVALENCE_AUDIT_2026-10-20.md)。
# 后续接口迁移说明

本文记录模块重命名后的复查。之后已按要求迁移训练、AL、图和加载接口；
当前原样归档范围为 31 个旧生成/LHS 文件，辅助模块已更新。
现状和文件名见 [当前速查](DATA_GENERATION_QUICKSTART.md)。

