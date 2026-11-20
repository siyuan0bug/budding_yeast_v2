# 2026-11-16：完整候选池数值修复已完成

本次只修求解及监督真值，没有改mutation类型、概率、强度分布、ODE、初值或事件重置，也没有启动正式训练。

## 求解策略

生产入口 `data/systems/budding/simulation.py` 默认BDF改为 `rtol=1e-8、atol=1e-10`。策略独立存放 `solver_policy.py`，版本 `budding_bdf_tight_checked_v1`。输出TEM1明显超过其总量（容差max(0.01,总量1%)，且初值位于合法区间）或存在低于−1e-4的负值时，在同一输入下进一步用1e-10/1e-12复算。求解失败也允许一次精确重试。

每次尝试保留5秒截止，最多两次、总计最多约10秒，低于35秒父进程看门狗。仍失败或总量约束仍明显不满足，返回数值失败；生成器记录失败并继续采样，达到目标有效数才发布。不发布缺额池，达到尝试上限明确失败。已有回归强制一条失败，验证最终仍取得3条有效样本且记录1次失败。

有限负值不作为丢弃条件：明显负值仅触发复核；复核仍有限或复核失败时保留可用有限结果并记录。微小负值保留。没有恢复real×2、MASS上限或其他经验幅值过滤。记录每次真实容差、状态、负值和TEM1异常。

LHS失败补采延续现有多design规则；跨design补采后的最终接受集不能冒称仍严格保持单个全局LHS分层。本次修复11条均在原输入上成功，未重新采样、未引入此问题。

## 冻结数据视图

旧原池保留。新默认入口：

- `/home/users/zsy/mydata/budding_yeast/offline_pools/mutation_100000_seed42_numerical_qc/dataset.json`
- `/home/users/zsy/mydata/budding_yeast/offline_pools/lhs_matched_100000_seed42_numerical_qc/dataset.json`

每池仍100000条。新manifest引用原块，仅11条更正行进入 `dataset.patch.npz`；没有复制20万轨迹，也没有把未修行伪称为重新高精度积分。原generation元数据保留，修复来源、旧manifest哈希、修改字段、ID及求解策略单列于provenance.numerical_repair。

mutation更正0-based行272、38053、49864；LHS更正10191、13667、15153、20781、44998、56394、61238、94061。最严重38053 TEM1峰125.7483→4.1658，总量4.1901。所有11条复算有效，重新计算生物标签、ABC、六分组和分裂次数。原始输入、mutation描述、初值、样本ID和次序逐位保留。

公共修补存储实现 `data/dataset_repairs.py`，生产分析入口 `scripts/analysis/repair_pool_numerical_artifacts.py`。修补前后源文件及字段哈希验证，修补视图支持原有TrajectoryStore和ManifestArchive接口；目标存在时拒绝覆盖。

## 协议与评测参考

新协议 `/home/users/zsy/mydata/budding_yeast/benchmarks/budding_sampling/seed42_a866aef5d988/`。

新正式结果将位于 `/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/seed42_a866aef5d988/`，没有启动或覆盖旧结果。归一化validation-best、train-only逐轮重拟合、5轮预算与epochs、epsilon0.01和无ratio截断保持最新确认配置。

四个输入/split缓存硬链接复用 `seed42_b5bff21cb8de`，90000候选/5000validation/5000test、初始2000及Random顺序完全不变。协议明确固定旧的126株real参考 `references/real_reference_b227aee93901.npz` 及其SHA；CLI优先复用并校验，不能因求解代码hash变化自动更换本轮比较的评测真值。其他新建参考则按实际新容差记录身份，不再错误写成旧容差。

## 旧实验需要重训吗

下面仅指上午 `seed42_02eebd57592f` 的16项，round编号从00开始。11条均不属于initial/validation/test；7条曾被10项实验训练，另外6项没有使用错误监督真值。

|池|方法|最早受影响训练轮|
|---|---|---|
|mutation|LCMD-LL|round_02，第3轮|
|mutation|VeSSAL|round_03，第4轮|
|mutation|Uncertainty/TopK|round_04，第5轮|
|LHS|Random、Uncertainty/TopK、LCMD-LL、VeSSAL|round_02，第3轮|
|LHS|CoreSet-LL|round_03，第4轮|
|LHS|MaxDist-Pred、BAIT-LL|round_04，第5轮|

只考虑数据修复：受影响方法需从首次使用错误标签的训练轮重训，并重拟合归一化；模型相关方法的后续选样也应重算。Random可保持相同ID顺序重训。没有用过这些错误标签的6项，不因本次修复而需要重训；固定模型、val/test和real目标未变，修数据也不会自动改变其旧指标。仅重新eval不能消除错误监督对权重的影响。最严重38053未进入任何旧训练，不能用它单独解释旧实验的整体误差。

另一个独立因素是用户已确认更换为归一化validation-best：该协议改变了首轮权重，完整方法对比仍应按新协议运行；不能把上述数据修复的局部影响范围误当作checkpoint协议迁移的全部影响范围。本次未启动正式训练。

证据位于 `/home/users/zsy/mydata/budding_yeast/analysis/experiment_flow_audit_20261116/`：`numerical_repair_completed.json`、`numerical_repair_verification.json`及此前的严格复算曲线。17项CPU定向回归通过（求解/补采/修补读写/生成恢复/非有限预测与LR日志）。

最终评测继续要求原始预测、逆变换结果和指标都有限。初期有限大loss允许继续；真正NaN/Inf不能通过clamp伪装成正常预测，也不能换零或删除该测试行。
