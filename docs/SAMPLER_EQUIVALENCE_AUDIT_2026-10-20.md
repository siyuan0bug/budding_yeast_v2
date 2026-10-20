# 新旧采样算法、分布与功能模块一致性核验

## 结论与范围

本次专门验证采样代码，补充此前 126 个 Budding / 20 个 Fission real 仿真的数值对照。验证中发现迁移遗漏并已修复；不能用此前的生物模型对照代替采样流程对照。

修复后，固定相同配置、seed 和初始随机数状态，采样内核与旧入口的候选调度对照 **105/105 组通过**；两套突变实现、分布和功能模块检查 **381/381 项通过**。这里的候选调度对照临时绕过新旧碰撞处理，以独立核对算法、层级顺序和随机流。碰撞分支另有针对性测试；用户要求的“旧丢弃改为救援”仍会改变最终整批样本。

原有 44 个 Python 文件继续逐文件校验 SHA256，未修改。此次未生成正式大数据集、未运行训练。

## 采样算法如何验证

脚本：[audit_sampler_refactor.py](../scripts/analysis/audit_sampler_refactor.py)。使用 seed `0、7、42`；每个常规对照抽取 256 个候选，曲率 k-DPP 的调度对照按旧逻辑抽取 `ceil(256 × 1.35) = 346` 个。

| 对照对象 | 检查内容 | 结果 |
| --- | --- | --- |
| Pure Random、KO interval、MASS Random、MASS Random + MA 分布、旧 MASS motif、新 MA MASS | 基因/模式描述、覆盖参数值、逐候选全局/局部 RNG 状态 | 一致 |
| Gibbs、Gibbs-Sobol、k-DPP、曲率 k-DPP、Motif | 完整候选构造及图内核 RNG 状态 | 一致 |
| BAMS | 骨架、模式、参数和二分搜索任务描述；包含触发细化所需的任务信息 | 一致 |
| Sobol 参数池、Fission Random | 候选与 RNG；Fission 按旧脚本首次运行的网格初态对照 | 一致 |
| 上述 14 个常规配置的旧入口 | 实际调用原生成函数，在得到完整候选前缀后中断；对照新 `prepare_candidates`，检查真实配额、顺序、打乱及随机流 | 一致 |
| WT-Blind | `flat/s5` 模式频率 × `flat/tier2` 强度策略，共 4 种组合的候选与 RNG | 一致 |
| WT-Blind pilot + ledger | 执行旧函数实际预采样和接纳循环，在仿真前截取结果；每个 seed 为 32 个 pilot、16 个目标候选，对照候选、D、ledger 接纳信息、尝试次数及 RNG | 一致 |
| Twins | 每个 seed 的 256 对共享骨架、两侧参数及每一步 RNG | 一致 |
| BAMS targeted ablation | 实际旧成对生成函数的 256 对结果与新实现、最终 RNG | 一致 |

两侧不是分别重写一个预期公式再比较：常规调度直接调用保留的旧入口，截获采样函数结果；WT-Blind 在旧函数仿真前截取其局部结果。仅为隔离已授权改变的碰撞策略，临时把碰撞判断设为不碰撞；不会改写源文件，也不会运行 ODE 或导出伪数据。

机器报告：`scratch/probes/sampler_parity_audit.json`，日志：`scratch/probes/sampler_parity_audit.log`。105 组包含 84 组常规内核/调度对照、12 组 WT 候选策略、6 组 Twins/定向消融和 3 组 WT pilot/ledger。

## 两套突变规则和分布如何验证

脚本：[audit_mutation_profiles.py](../scripts/analysis/audit_mutation_profiles.py)。MASS 与 mutant_aware 分别对照自己的旧实现，使用相同 WT 基础参数，避免把用户要求统一基础模型误判为分布变化。

| 检查 | 覆盖范围 | 结果 |
| --- | --- | --- |
| 两套幅度函数及两套参数修改函数的公式结构 | 比较 Python 语法树，仅消除新增可选 `rng` 接口及其全局 RNG 等价写法 | 4 个函数全部一致 |
| 幅度抽样 | 每套 84 个合法基因/模式组合，另加 ORI 扰动分支；3 个 seed，每组合连续抽取 32 次 | 共 16,320 次幅度值及 RNG 状态一致 |
| 突变参数修改 | 同样 85 个分支 × 3 个 seed，强度取 `0、1e-4、0.01、0.2、1、2、30` | 共 3,570 次 `p_mut`、覆盖字典及 RNG 状态一致 |
| 模式选择概率 | 27 个模块 × `class/flat/role/s5` × 3 个 seed × 32 次 | 共 10,368 次选择及 RNG 状态一致 |
| 公共功能模块 | 基因目录、角色、模式类别/权重、touch sets、复合突变、L1 网格、54 边拓扑、邻接/距离、收敛对、pair/star 等派生集合 | 与 MA 来源一致 |
| 冲突处理与配额选择器 | 隐式全局/显式局部 RNG 的冲突重抽，QuotaPicker 的普通和加权选择 | 一致 |
| 专用模块 | BAMS 的有向图与派生集合；WT-Blind 独立的 46 边图、强度类别和网格；Fission 模块目录 | 与各自原实现一致 |

机器报告：`scratch/probes/mutation_profiles_audit.json`。其中 MASS 目录的 `equal=true` 表示“只有已声明的统一初值差异”；原始目录是否完全相同另存为 `exact_equal=false`，没有把这处差异隐藏掉。

## 验证中修复的迁移遗漏

1. **Gibbs-Sobol 配额与顺序**：旧入口对每层四舍五入、差额补到 L1，然后按 L1→L5 顺序生成。新配置此前误用 floor + shuffle。已恢复旧逻辑，也恢复 `SOBOL_` 样本名前缀。
2. **MASS Random 层级打乱的随机流**：旧入口使用全局 `np.random.shuffle`，新入口此前使用局部 `RandomState`。虽然打乱后的层级可相同，但全局 RNG 消耗不同，会改变后续参数。已改为旧用法。
3. **Twins A 的 TEM1-ts 初值**：原 Twins 对 MASS 规则增加了显式 `init_Tem1=0`，不能直接把它当作无差别的 MASS 别名。已在专用薄封装中保留该字段。
4. **碰撞救援的调用方 RNG**：Pure Random 原本显式传入局部 RNG；MASS、k-DPP、BAMS 原本使用全局 NumPy RNG。此前公共入口一律传局部 RNG，已按旧调用方式区分；BAMS 工作者中按 sample_id 派生的独立救援流继续显式指定。专项测试在 4 个入口、12 个 seed 上对照一次救援的结果和两条 RNG 状态。

补充回归位于 `tests/test_modular_datasets.py`，本轮 **65 项全部通过**，覆盖这些迁移遗漏，避免以后只验证内核而遗漏调度或调用方状态。日志保存在 `scratch/probes/modular_tests_after_sampling_audit.log`。

## 仍然存在的预期差异

- **模块目录的两处安全初值扩充**：MASS 原目录中 TEM1 只有 `Tem1T、ka15pp`，CDC15 只有 `Cdc15T、kkpnetpp`；新公共目录沿用 MA，分别额外列出 `init_Tem1、init_Cdc15`。其余目录字段相同。MASS 自己的合成参数修改函数和分布没有因此被换成 MA。
- **real 和基础仿真**：Budding 统一使用 MA 的初值/real/ODE；这按用户要求实施，不与旧 MASS 的问题初值保持一致。
- **碰撞政策**：旧代码有丢弃、一次救援后重抽等不同路径；新流程统一有限次数救援。前述 105 组候选比较隔离了这项变化，不代表真实碰撞后的整批序列仍完全相同。旧复合突变描述的解析局限也不作为新流程的兼容目标。
- **并行与失败处理**：BAMS 工作者的稳定 seed、Fission 重复运行网格重建、失败/不完整积分过滤，以及新 NPZ 元数据，仍按整理说明执行。

因此，结论是：**两套采样分布和参数修改实现保持原规则；已核验的采样内核及无碰撞调度在固定状态下逐项一致。完整数据集仍需按明确的政策差异解释，不能承诺所有 seed 的整批 NPZ 字节相同。**

## 复跑

从 `budding_yeast_v2/` 执行：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/users/zsy/.conda/envs/neuralop/bin/python scripts/analysis/audit_sampler_refactor.py \
  --samples 256 --seeds 0 7 42 --output scratch/probes/sampler_parity_audit.json

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/users/zsy/.conda/envs/neuralop/bin/python scripts/analysis/audit_mutation_profiles.py \
  --draws 32 --seeds 0 7 42 --output scratch/probes/mutation_profiles_audit.json
```

验证环境仍为 NumPy 2.2.6、SciPy 1.15.3。精确随机序列结论限定于相同依赖环境和测试状态；语法树对照则补充检查未被少量 seed 触发的公式分支。
