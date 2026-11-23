# 条件机制覆盖与组合迁移研究对照

状态：2026-11-22完成21项训练（7方法×seed42/123/1024），2026-11-23完成结果/权重核验；属于研究原型，不宣称理论创新或Fission已适配。完整指标及审计见[本批报告](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/conditional_dynamics_v1/report.md)。机制覆盖Real平均MAE比池内随机低3.07%，但只赢1/3 seed；合成测试MAE三seed均改善、均值改善7.31%。组合梯度迁移及机制缺口生成均未证明优于各自随机对照，暂不扩大实验。

## 固定协议

只使用现有mutation完整池，生成seed42不变；实验主seed42/123/1024，复用各自共享split及共同初始2000条。全部最终模型从头训练相同10000条/595epoch/47005步，warmup30epoch，FNO192/4blocks/24modes，batch128，LR5e-4到1e-5，当前无截断Relative eps0.01。来源池合成验证归一化loss选best；统计各自按最终训练集重拟合一次。这是一次性选择机制研究，不冒充历史五轮协议。迁移方法额外pilot100epoch/1600步，单列成本，不从最终47k步中隐去。

从训练侧输入随机保留：反馈512、独立未见背景检查512、随机输入检查512。反馈和未见检查的完整模块/模式/mdt背景彼此隔离，并从全部训练/生成中排除。随机检查的完整输入也被排除。来源validation/test不进入选择；real只在训练best锁定后读取终评。real已长期参与项目开发，不能称新盲测。

## 七个对照

| 代码名 | 实际操作与信息条件 |
|---|---|
| random_coverage_control_v1 | 从相同合法候选随机新增8000条 |
| trajectory_coverage_v1 | 已仿真训练侧轨迹的四时间段均值/RMS压缩；普通facility覆盖 |
| context_coverage_v1 | 相同轨迹描述，但仅同干预类型能互相覆盖，干预类型等权 |
| ours_mechanism_coverage_v1 | 同干预覆盖约束，加入在实际状态上的原生RHS撤销差、初值差、重置代数差及状态/作用交互坐标 |
| ours_composition_transfer_v1 | 初始pilot的输出头Relative梯度128维sketch；反馈完整背景隔离，匹配共享干预在其他背景上的损失加权梯度，分块更新目标残差 |
| random_gap_generate_control_v1 | 相同2万合法新提案的随机顺序；仅使用初始2000条真值，ODE失败补足 |
| ours_mechanism_gap_generate_v1 | 相同2万提案，以最近初始输入锚点的已知状态路径作为代理，按条件机制覆盖排序，再查询真实ODE补足8000成功 |

机制描述不是反事实轨迹：仅在同一个状态上比较RHS。撤销使用原有touch footprint与原生初值准备，mdt保持不变；同基因复合干预整体处理，重叠干预合并。重置描述独立调用原生reset映射，不声称该事件实际发生。32点状态摘要不作跨跳变有限差分。选择描述的符号log压缩不改变模型归一化。

覆盖目标为抽样landmark的同干预相似度最大覆盖均值，每种干预等权，类内随机landmarks；最多32768个目标点上使用epsilon=.05随机贪心，初始2000条预覆盖。代表输入数必须超过新增预算，防止通过选完少量代表点导致目标提前饱和；大矩阵分块构建并检查GPU余量。context/机制两组使用相同求解器和类型约束，普通轨迹组去掉类型约束。只有完整机制超过两个覆盖消融，才有依据归因于机制描述。

组合迁移是梯度匹配的任务适配，不是精确双层训练收益。候选和反馈共享某项干预但完整背景不同，使用归一化头梯度方向、按反馈初始误差加权、每64条更新残差。需要与覆盖和随机实测，不能把一阶评分称为实际训练收益或全模型梯度。

生成版不读取未查询新提案真值、也不读取其他池候选真值排序。最近输入锚点可能不能代表新候选真实动力学，这是显式近似，需要随机生成控制检验。每组最多尝试12000次ODE以补足8000条成功；底层5秒/父级35秒、数值QC及全部生物规则沿用。新增轨迹只写一次分块，累计数据用引用清单。池选择访问所有合格训练真值，不能与生成组宣称相同标签访问成本。

## 入口与产物

批量：`scripts/launch/run_conditional_coverage_study.py --directory <外部batches/批次名>`；子任务CLI：`scripts/analysis/run_conditional_coverage_study.py`。最终训练/损失/逐样本CSV/126张终评图/offline W&B均在标准`fno/mutation/<method>/<profile>/seed_<seed>/`。源ID清单、少量选择特征、pilot与新增轨迹分块位于`shared/pool100k/seed_<seed>/studies/conditional_dynamics_v1_fullcoverage/`，避免跨实验重复保存轨迹。共享pilot是方法额外开销，明确记录于选择provenance。

批次状态/资源/启动日志只在batches；汇总CLI为`scripts/analysis/summarize_conditional_coverage.py`，只读取完成结果，输出summary/conditional_dynamics_v1。0～3号GPU最多每卡2项/共8项，I/O并发4，每生成器8个CPU worker；系统有效可用内存低于约100GiB触发本批进程保护，PSS只监测不设配额。实验开始后冻结sources身份范围，禁止边跑边改源码。
