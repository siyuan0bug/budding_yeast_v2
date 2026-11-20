# 两个10万完整轨迹池：生成完成与全量验收

## 当前结论（2026-11-13）

两个池已正常完成，各100000条、无real行；共48个ODE工作进程并发生成约26分钟，启动脚本退出码0。旧在线AL训练入口尚未迁移到这两个离线完整池，本次没有启动训练。

| 检查 | Mutation | 匹配边际LHS |
|---|---:|---:|
| 最终轨迹数／唯一有效输入数 | 100000／100000 | 100000／100000 |
| 保存输入与real碰撞 | 0 | 0 |
| 仿真失败记录 | 0 | 0 |
| 数据块数 | 391 | 391 |
| 目录占用 | 5.54 GiB | 6.18 GiB |
| 含任意负值的轨迹 | 18061 | 6551 |

`scripts/analysis/audit_budding_complete_pool.py`已逐块检查完整参数、canonical实际初值、轨迹首点、字段/源文件哈希、有效输入去重及real排除、mutation规则与环境、ABC及六分类一致性。LHS的14100000个参数坐标通过原始Latin分层及完整CDF映射核对，最大概率约束误差8.70e-12。本次LHS没有拒绝或补采，保存集合保留了原始设计的一维分层。

Mutation的1159种合法模块集合均有覆盖，最终L1/L2/L3为1017/12972/86011。共112094次提案，11772次重复、322次real碰撞救援耗尽，最终保留238条救援样本。计数器中的real_collisions只累计进入仿真阶段的碰撞救援候选，不能将238误称为生成期间全部碰撞次数。

**匹配边际针对mutation抽样提案，而不是承诺最终两个池的经验边际完全一致。** Mutation的有限容量组合在去重/real排除后占比降低，LHS当前没有发生拒绝；最终L比例也不同于提案理论比例1.6393%/14.7541%/83.6066%。参数经验WT/零值比例已导出，论文应同时披露接受机制及LHS额外90维背景扰动，不能将此比较解释为只改变了Latin分层。

六分类为中性轨迹分组，数量如下；BIO和ABC另存，不能把P1/P2直接当作长期极限环或衰竭的机理证明。

| 分组 | Mutation | LHS |
|---|---:|---:|
| P1 | 5343 | 3832 |
| P2 | 25417 | 18750 |
| P3 | 10909 | 19305 |
| P4 | 9902 | 12096 |
| P5a | 36936 | 34072 |
| P5b | 11493 | 11945 |

### 数值质量复核：结构检查通过不等于严格解精度证明

按用户既定政策，有限负值没有被删除或截零。Mutation全池最低值约-1.64e-6；LHS最低CLB2约-0.03125，低于-1e-6/-1e-4/-1e-3/-1e-2的轨迹分别11/4/2/2条。

对LHS负值最明显的12条，使用相同参数、实际初值规则和事件流程分别以rtol/atol=1e-8/1e-10、1e-10/1e-12复算，共24次全部成功：12条均恢复非负，六分类均不变，两档严格解之间最大完整轨迹RelL2约2.89e-5。证据支持负值来自宽松容差下的数值误差。原解对严格解的最大完整轨迹RelL2约2.30%，且并非出现在负值幅度最大的一条；不能凭负值大小推断所有解的精度。这是针对12条的诊断，不能外推全池已经严格精度验收。

**原始池、轨迹和标签未被复算覆盖。** 复算是独立诊断，后续若统一提高求解精度，应对两种生成范式采用同一协议，并保存新的求解来源，不能静默替换冻结数据。

验收输出位于`/home/users/zsy/mydata/budding_yeast/analysis/complete_pool_audit_20261113/`：

- `mutation/`、`lhs/`：`audit.json`、逐参数统计、逐变量统计；mutation另有模块集合与模式计数，LHS另有设计核对。
- `lhs_negative_diagnosis/`：`diagnosis.json`、`recomputations.csv`、`trajectories.npz`，含原始与两档复算轨迹，未修改源池。

生成日志记录Mutation在02:30:54、LHS在02:31:12发布完成，脚本02:31:13退出；这些时间按生成日志原样记录。`completion.json`的completed_at先于最终清单哈希计算，应以日志COMPLETE行识别发布结束时刻。

## 最终采用的规则

用户确认按匹配边际方案生成，但导师明确要求LHS的90个背景参数全部扰动。因此**保留90参数U(0.5WT,1.5WT)**，不执行上份论文建议中的“两池均固定WT”。Mutation池的背景参数仍是WT；两者是两种完整生成范式，论文必须公开背景参数域差异，不能将性能差异全部归因于LHS分层或mutation联合关系。

- Mutation：从1159个合法L1–L3模块集合等概率抽一个，再随机选择该集合的合法完整模式组合，强度使用现行MA随机分布，无Sobol或集合固定配额。MDT不占遗传模块，含GAL固定150，其他80%/15%/5%选90/150/160。
- LHS：45个模型参数按已审阅MA提案边际，90个背景参数±50%；2个事件控制、3个初值覆盖、mdt保留配置边际与必要初值依赖。使用完整混合CDF的广义反函数，处理重叠连续支持、WT/零原子、离散MC及DAmP裁剪端点原子，不能将模式区间简单串接。
- 每池100000条有效、唯一的合成轨迹，均有完整参数、实际初值、轨迹、BIO/ABC/中性六分组。real不进入这两个源池；mutation池保留模式、参数修改和L，LHS明确mutation不可用、pairs为空、L=-1。
- 共用canonical初值、39维ODE、四种事件和重置，210分钟500点、BDF rtol=1e-4/atol=1e-6、5秒单次截止。失败/非有限/不完整/不可编码为float32的轨迹拒绝补采；有限大值和有限负值不按幅值过滤。负值最小量、个数和变量在逐次诊断记录中保留。
- real碰撞按既有有效输入容差排除；mutation碰撞走既有70/30救援，不改环境或模块集合；LHS不伪造mutation救援，碰撞行直接拒绝补采。两池分别对有效输入签名去重。

## 生产入口与存储

新入口：`python -m budding_yeast_v2.data.systems.budding.offline_pool`。

```bash
cd /home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2
bash scripts/launch/prepare_budding_offline_pools.sh
```

脚本只准备数据，不启动训练或上传W&B。默认两个池各24个ODE进程，合计48；各100000行、生成seed42。PYTHON_BIN为neuralop环境。可用POOL_WORKERS、GENERATION_SEED、POOL_SAMPLES、POOL_ROOT、LHS_CONFIG覆盖。LHS配置默认机械盘已审阅的offline_pool_plan_20261112_exact_zero/mixed_lhs_domain.json。

实际输出根目录：`/home/users/zsy/mydata/budding_yeast/offline_pools/`（当前真实路径为/data/zsy/budding_yeast/offline_pools）。

```text
offline_pools/
├── mutation_100000_seed42.log
├── lhs_matched_100000_seed42.log
├── prepare_20261113.log
├── mutation_100000_seed42/
│   ├── run_config.json
│   ├── progress.sqlite / progress.json
│   ├── chunks/chunk_000000.npz ...
│   ├── chunks/attempts_000000.json ...
│   ├── pool_metadata.npz        # 完成时写入
│   ├── dataset.json             # 完成时原子发布完整数据清单
│   └── completion.json
└── lhs_matched_100000_seed42/
    ├── 同上
    ├── lhs_specification.json
    └── designs/design_0000.npz ...
```

每256个唯一候选构成一批，按提交顺序接受结果。每块NPZ仅保存一次轨迹，dataset.json通过已有data.dataset_archive.open_dataset读取全部命名数组，不再生成第二份完整累计NPZ。完成前不发布看似完整的dataset.json；进度读取progress.json。原始LHS设计保存141维物理参数、设计seed和潜在Latin坐标哈希，失败补采使用后续独立设计并记录design_id/row。接受后集合不能宣称仍严格满足原始Latin分层，需审计经验边际。

SQLite保存已提议输入签名及批次进度，提交数据块后再提交状态事务；中断后--resume从已提交进度继续，不重复计数。进程锁拒绝同目录两个生成器。配置、范围指纹或生成源码变更时拒绝盲目续跑。提交窗口与父进程35秒看门狗复用既有simulation_guards，脚本对退出最多重试3次并保留日志。源码及NumPy/SciPy/Python版本记录在run_config.json。

## seed的含义

- sampling_seed=42是生成种子，控制模块、模式、强度、环境、LHS格内随机点与排列。Mutation和LHS使用不同的kind子流，同为42不意味着参数样本相同。
- Mutation每个proposal_id使用完整SeedSequence熵生成PCG64流，不因worker完成顺序改变输入；不是将全部候选压缩到同一个32位子种子空间。
- LHS设计seed由生成seed和design_id派生；LHS样本数量也影响设计，不能声称256点设计是100000点设计的前缀。
- ODE没有内部随机数，但求解版本和壁钟超时可影响跨机器接受结果，因此保留所有参数、诊断及已提交块，以已存样本作为复现来源。
- split_seed、training_seed与生成seed分别记录；换训练种子不需要重生成10万完整轨迹。同一源池内所有选择器应复用初始化及holdout协议。

## 样本字段及读取兼容

共用write_dataset及现有BIO/六分组实现，新增扩展字段不能覆盖已验证核心字段。参数/初值float64、轨迹float32、训练输入仍按现有DataModule转float32。

每行保存params、y_init、data、名称/稳定sample_id/candidate_ids、is_real=false、overrides_json、实际mutation列表与可用标记、levels/requested_levels/module_counts、labels/pattern_labels/pattern_group_*、observed_division_counts和simulation_diagnostics_json。LHS另存design_id/row，mutation保存提案强度、原override及救援后的实际模式/参数；元数据不会把救援前强度冒充最终修改。

data/dataset_utils现在转发sampling_paradigm/dataset_role/mutation_available；YeastDataModule优先使用显式生成范式，独立参数LHS不再因为旧_LHS_名称提示走real家族/Pattern划分。旧文件没有该元数据时保留既有历史分支。is_real显式为false时不会伪造10个real测试样本。

当前源池可被通用数据读取器/DataModule读取，但**没有把旧run_sampling_budding.sh在线选样实验偷偷改成新离线流程**。两个10万源池的离线选样接入、2000→4000→7000→10000的训练快照和共同评测文件属于后续工作，不能现在直接把旧参数池验证器用于这些完整源池。

## 上线前验证

- 新增8项测试全部通过：混合CDF重叠/原子、141维反CDF、Latin一维分层、初值共享控制、无Sobol的随机mutation、完整小池字段/读取/拒绝错误续跑、中断恢复与种子分流。
- 现有dataset_archive/sample_metadata共26项回归通过，合计34项。
- 两池各256条试生成，8进程与1进程分别独立生成：params、y_init、data、BIO/ABC/六分组、mutation列表、candidate_ids、overrides全部逐元素相同；试生成均无求解失败。完整初值与轨迹首点在float32编码精度内一致。
- 两池实际YeastDataModule.setup均通过，按seed拆成204/25/27，无real行，train-only归一化通过；没有启动模型训练。
- 验证记录：analysis/offline_pilot_validation_20261113.json、complete_offline_pool_tests_20261113.log、offline_storage_regressions_20261113.log。这里analysis根目录为/home/users/zsy/mydata/budding_yeast/analysis。
- 试生成的最大变量值分别约1473与4631，只作完整性检查，没有据此增设阈值或调整分布。

## 正式后台状态快照（历史进度）

2026-11-13 02:05:25（服务器本地时间）启动tmux会话`budding_offline_pools_20261113`，两个任务均已进入ODE并持续提交数据块。02:06:33快照：mutation4352/100000，LHS3840/100000；两者求解失败均0，mutation记录real碰撞22次且均救援，去重82次。快照不是完成结果，最新值以progress.json和completion.json为准。

```bash
tmux attach -t budding_offline_pools_20261113
tail -f /home/users/zsy/mydata/budding_yeast/offline_pools/mutation_100000_seed42.log
tail -f /home/users/zsy/mydata/budding_yeast/offline_pools/lhs_matched_100000_seed42.log
```

tmux在IDE关闭后独立运行；Ctrl-b再d只脱离会话。目前任务已正常结束，因保留退出窗口而显示dead，不代表生成失败。未修改data根目录原始生成器、canonical参数/ODE/real或Fission。
