> **2026-10-30 存储更新**：下表字段同样适用于`dataset.json`还原的完整数组。新增轨迹仅存于增量NPZ，初始数据共享；样本内容和dtype不变。[无损布局](AL_DATASET_STORAGE_LAYOUT.md)。

# 数据集样本内容契约

2026-10-28：统一新系统静态导出、AL新增样本及累计快照的样本信息。原始生成器、已有NPZ、ODE、突变赋值、采样分布和标签判据不变。模块集合等额的新配额仍待接入，此次只补齐保存内容，不生成正式数据或启动实验。

## 每行保存什么

同一NPZ内，以下逐行数组都以第i项表示同一条样本；名称和时间轴为共享字段。

| 内容 | 字段 | Budding含义 |
|---|---|---|
| 动力学参数 | params、param_names | N×141，物理值及顺序；保留各历史配方原有参数dtype |
| 变量初值 | y_init、variable_names/var_names | N×39，系统实际准备的初值，float64单独保存 |
| 突变说明 | mutant_names、mutation_descriptions | 实际合成描述或原始real名称 |
| 结构化突变 | mutation_pairs_json、mutation_pairs_available | 最终基因—模式列表，复合模式展开；布尔字段表示是否有该结构化表示 |
| 参数和初值修改 | overrides_json、overrides_available | 保留init_*、事件开关等，不能只靠141维参数向量重建 |
| 候选来源 | candidate_metadata_json、candidate_ids | 包含模块集合/模式和source_pool/source_candidate_id等来源（如适用） |
| 行标识/配对 | sample_id、pair_id、is_real | sample_id在当前文件内唯一；无配对或候选来源时对应ID为-1 |
| 突变阶数 | levels、requested_levels、module_counts | 合成样本L1–L3；real保留0/-1哨兵，不把real的四模块情况删掉 |
| 物理轨迹 | data、time_grid | N×39×T，默认T=500、0–210min；只在已完成仿真文件中存在 |
| 生物标签 | labels、biological_labels_available | 原生viability/arrest等；不是六分类替代品 |
| ABC | pattern_labels | 保留原有三类训练目标 |
| 六分类 | pattern_group_codes/labels/ids | P1/P2/P3/P4/P5a/P5b，有限窗口的中性描述分组 |
| 分裂与分类规则 | observed_division_counts、pattern_group_version/rules_json | 分裂次数及可核对的规则版本 |
| 求解/初值来源 | simulation_status、simulation_diagnostics_json、initial_state_source | 成功状态、已有诊断及初值来源；未采集的诊断为{}，不伪造求解器细节 |

sample_metadata_version=sample_provenance_v1标记补齐后的样本元数据。静态导出同时记录system_id、model_version、benchmark_profile、mutation_profile、distribution_profile、catalog_version、seed、RNG契约、生成配置及报告。AL快照和新增文件继承初始数据中的模型/分布/catalog等版本，来源候选写入逐样本metadata，轮次/策略另见实验config和acquisition报告。

real mutant的原始名称和真实overrides完整保留。有些真实等位基因不能等同于合成catalog的KO/ts等，因此不会从名字或参数反推一个虚假的合成模式。此时mutation_pairs_available=False、mutation_pairs_json=[]，并不表示real名称或参数修改丢失。

## 候选池与轨迹文件的区别

参数候选池也保存完整参数、初值、mutation描述/列表、overrides和L信息，但没有data，也没有生物标签、ABC、六分类或分裂次数；simulation_status为parameter_only。它可以有计划使用的time_grid，该字段不表示已经进行完整轨迹仿真。上述依赖轨迹的标签在初始集仿真或AL候选被选中并成功仿真后计算。

正常轨迹数据集只保存通过数值有效性检查的完整轨迹，不用假的Pattern_C代表求解失败。合法完成但生物上不可存活的样本仍有对应原生标签。

## AL全过程保存

- 初始文件：systems/al_initial.py使用统一write_dataset，记录source_pool和原候选行ID，再保存固定train/val/test/real索引。
- 新增样本：acquisitions/after_round_XX/added_samples.npz保存同样的输入、mutation元数据、标签和来源。
- 累计视图：rounds/round_XX/dataset.json引用real、累计train及固定val/test，完整保留全部样本元数据；不只保存参数/轨迹和L。
- oracle在删掉候选池中的已消耗行之前提取provenance，因此失败过滤、动态补采和重排不会把mutation信息或ID错配给另一条轨迹。
- y_init独立于float32轨迹保存。模型的float32输入、归一化、训练目标不变。Fission仍保留自己的原生标签及None，不附加Budding六分类；质量单位沿用原协议。

全部16种Budding静态配方（含twins、KO镜像和BAMS诊断派生）最终共用systems/dataset_io.py的导出；本次没有另写一套按采样器区分的字段规则。

## 已有文件与验证范围

已有NPZ不会自动补字段，也没有被改写。兼容读入缺少provenance的历史文件时，overrides_available=False；初值只能来自轨迹首点时标记trajectory_t0，不能声称恢复了原始精度。当前新生成的文件记录真实来源，新的AL快照会保留初始源文件中已有的完整信息。

验证覆盖16种静态配方的公共导出、无轨迹池不产生标签、复合/real突变说明、float64初值往返、候选池删除后的ID稳定性，以及Budding/Fission真实ODE追加与重载。静态配方导出测试使用可控轨迹fixture，不能当成重新审计了所有采样算法的生物学行为；此次未修改采样算法。

本次验证：综合回归69项通过；随后针对压缩列单次读取和来源往返的补充检查21项通过（与前者有重叠，合计70个不同测试）。没有运行正式模型实验或生成正式候选池。
