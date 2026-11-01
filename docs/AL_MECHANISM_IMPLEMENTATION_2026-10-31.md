> **完成更新（2026-10-31 16:34）**：本批四项及Event重跑全部完整完成，实际预算/辅助模型/查询前审计均核对通过。[完整用时和结果](AL_MECHANISM_COMPLETION_2026-10-31.md)。下文启动时“尚无结果”等为历史记录。

# Pattern条件支撑与ODE响应采集：实现与运行

用户已接受两条研究方向。本次实现三项Pattern消融及一项独立物理响应策略；继续使用同一FNO、gene_ranges pool、初始集及full训练协议。下面描述实际实现，不是性能结论或已完成的新颖性论证。

## 文件与调用位置

| 文件 | 职责 |
|---|---|
| `utils/active_learning/auxiliary/pattern_classifier.py` | ExtraTrees辅助分类器、train分组交叉拟合和温度校准 |
| `auxiliary/views.py` | 显式合成train/无标签pool/归一化数据视图 |
| `auxiliary/phenotype.py` | 完整500点真实/预测六分类、119维物理描述 |
| `auxiliary/coverage.py` | 条件稀疏核及对数支撑收益的逐批贪心 |
| `auxiliary/canonical_rhs.py` | 从当前canonical NumPy RHS自动适配的可微Torch RHS |
| `auxiliary/mutation_directions.py` | 从当前mutation规律导出合法连续/整数强度方向及实际初值 |
| `auxiliary/sensitivity.py` | 连续区间的切线/整数割线一致性评分 |
| `auxiliary/artifacts.py` | 原子保存辅助分类器及查询前预测 |
| `strategies/pattern.py` | 三个Pattern策略的统一入口 |
| `strategies/sensitivity.py` | 独立ODE响应采集入口 |
| `registry.py` | CLI与引擎注册，均限定Budding |

辅助分类器不是新的轨迹代理模型，不加入`train.py --model`选择列表，也不替换FNO。其对象实现位于`auxiliary/pattern_classifier.py`；模型文件由每轮追加保存流程落盘。工程入口仍为`run_al_budding.sh → train.py → ActiveLearningEngine → registry → strategy`。

## 已注册的四个方法

| 名称 | 实际选样目标 |
|---|---|
| `ours_pattern_balance` | 朴素对照：候选分类概率除以训练集/本批预期类别计数，动态补充不足类别 |
| `ours_pattern_context` | 结合输入背景和分组匹配，增加真实已查询证据对候选的支撑 |
| `ours_pattern_support` | 同一目标再加入同组内连续轨迹描述，避免“分组对就算覆盖好” |
| `ours_sensitivity` | 在合法突变方向上检查模型响应与当前ODE的一致性，独立于Pattern辅助模型 |

已有`ours_response`继续保留。更复杂的“条件支撑＋已知锚点响应指纹”尚未加入此次前三项，避免将它与条件覆盖本身混为一项消融。物理方向单独验证，不与Pattern排名固定比例融合。

## 辅助分类器如何训练

输入是141维物理参数、39维实际初值及当前catalog合法gene/mode多热编码；Z-score仅拟合累计合成train。标签由`pattern_group_metadata`从合成train实际500点物理轨迹计算，保持六类原定义，不改BIO/ABC。

32棵ExtraTrees、深度14、叶节点最少8、最多2线程、balanced训练权重。train内部按模块集合/环境/事件开关分组，最多3折OOF；在预先给定的温度集合中按OOF NLL选择温度，再用全部train拟合最终树。六类概率有正伪计数，缺类不会报错，但不能声称平滑就保证发现缺失类。

两个独立FNO分别推理、反归一化、提取分组和连续描述；先分类后投票，不先平均轨迹制造事件。连续描述是39变量的log峰值、log终值、log均值及分裂次数/首次分裂时间，共119维；候选使用各成员描述的均值，这是对研究设计中核期望的近似，不是完整贝叶斯联合后验。

当前核中的六类概率来自辅助分类器；FNO的六类投票只保存作诊断，不作为额外的分歧奖励。`context`和`balance`也计算并保存FNO描述用于统一诊断，但只有`support`将连续描述用于选样。

朴素平衡对照每128次选择刷新全池评分并取1024点短名单，短名单内每选一个点即更新预期类别计数。这是明确的微批近似，不声称等同全池逐点精确排序；概率相同时，全池lazy heap会反复刷新百万候选，因此不采用该退化实现。条件支撑目标仍用冻结稀疏核上的精确lazy greedy。

## 条件覆盖的计算与近似

背景分组为模块集合＋mdt＋非参数事件开关；组内使用train归一化后的参数、实际初值及mutation编码做软距离，因而不同突变形式仍有区分。输入/分组核与动力学RBF相乘，训练端使用真实one-hot和实际描述，候选端使用辅助概率和预测描述。

每个候选分别查16个候选近邻和8个训练近邻。两类邻居必须分别查：在百万pool中只找混合近邻，训练点很容易完全进不了邻域。训练近邻形成实际支撑，候选近邻形成稀疏批覆盖核。

冻结本批核和参考集，贪心最大化：

\[
\Delta(A)=\sum_{x\in U}\log\left(1+\frac{\sum_{a\in A}K(x,a)}{0.05+\sum_{z\in D}K(x,z)}\right).
\]

选择同一区域后边际收益递减。此处是非负稀疏代理目标；只对该冻结目标做精确lazy greedy，不声称对真实FNO风险最优。硬模块集合分组和近邻截断限制了跨模块迁移，是需报告的实现边界。初版不加入real比例、real峰值或手选badcase邻域。

## ODE敏感性方法的精确范围

没有复用旧的32通道简化残差，也没有复制一份ODE。适配器读取`systems/budding/ode.py`当前`GK/eqns`源码，只将数组布局、NumPy函数和标量条件改为Torch运算，保留所有39通道与特殊开关的原有代数。源码摘要记录在采集设置中；原ODE、solver和data根目录备份均不改。

每个候选按其稳定ID与轮次，从自己的合法强度方向选一个方向。连续模式覆盖GAL-OE、ts、DAmP、pLOF、db、APC-A和TAB6-1，使用生产mutation规则/CDF曲线及0.01 CDF半步构造模型探针。参数和实际初值都由现有初始化逻辑重新组装，mdt不变。不是随意给KO加一个微小非零值。

连续方向用合法两侧模型预测的有限差分近似S=dY/du，对照当前中心预测处canonical RHS的精确JVP：`dS/dt = f_Y S + f_p dp/du`。初值条件同时比较`S(0)`与实际`dy0/du`。因此模型参数和初值编码的链式影响都进入数值方向，但模型灵敏度仍有有限步长近似误差。

普通MC/GAL-MC只用合法整数拷贝邻点，计算有限割线及两侧RHS的实际差分，不冒称整数拷贝的微分导数。没有可变强度的KO/固定等位模式不伪造方向。

评分屏蔽四个guard、向下重置附近及探针事件计数不同的时段；ORI大幅**正向连续增长**不会仅因幅值大被当重置。最低有效时间占比25%。该实现只声称连续片段的一致性，尚未实现saltation矩阵或真实事件时间灵敏度。非有限/不可观测方向得不到物理利用分数，不因此删除候选或训练样本。

时间导数用500点物理分钟轴的中心差分；对ODE响应率做逐变量相对尺度归一化，并加上train-derived浓度/时间下限，再联合初值一致性。90%预算优先取有限正物理分数，至少10%作私有seed随机探索；方向不足时余量也探索，确保固定模式仍有入选机会。随机探索不是方法的创新主张，完整计费和分支均记录。

探针只做FNO预测，不调用ODE、不修改pool、不加入数据集。真正query的始终是原pool的候选ID。

## 无测试信息的实现边界

`build_view`是唯一可以看旧engine/datamodule的适配器，只读取累计合成train、无标签pool、train-only归一化统计及时间轴；要求`fit_scope=train_synthetic`、非负唯一candidate IDs且train/pool不相交。核心辅助代码不接收datamodule、benchmark名称、指标文件或实验路径。

单元测试以禁止访问test/val字段的datamodule执行，并在冻结模型/train/pool后移除或破坏real/heldout缓存，检查四个策略候选ID不变；另验证真实train标签改变会改变辅助概率。这是防误用的数据边界，不是Python恶意代码安全沙箱。

代码层面隔离不改变历史126 real已用于方法开发的事实。当前real仍是开发评测集，后续独立泛化结论必须增加未参与设计/调参的冻结评测，见研究方案的数据协议。本次不把任何real指标用于辅助分类或选样。

## 保存什么

每轮`acquisitions/after_round_XX/`在原有新增NPZ、selected/report、query_design之外增加：

- Pattern方法：`pattern_classifier.joblib`，包含完整树、编码词表、train Z-score、温度；`pattern_classifier.json`记录训练行数、训练ID摘要、类数量和OOF诊断。
- 四方法：`mechanism_predictions.npz/json`记录**查询前**原candidate ID及预测。Pattern包括分类概率、轨迹投票、描述、边际增益；物理方法包括方向类型、基因/模式、等位分支、坐标/步长、有效时间占比、初值条件误差和分数。
- 这是查询尝试记录，包含可能仿真失败的输入；与真正accepted的`added_samples.npz`按candidate ID连接。不能将失败尝试当成新增有效训练样本。

所有写入先完整编码到临时文件，再原子替换。跨轮数据仍是共享初始块＋新增块＋`dataset.json`，无累计轨迹重复副本。

## 入口和协议

`run_al_budding.sh`默认`AL_SUITE=mechanism`，四项分配GPU0敏感性、GPU1朴素平衡、GPU2条件背景、GPU3条件动力学支撑。保持每卡1项，已有任务占卡则等待；`AL_SUITE=optimized/research/original`可显式复现此前清单。

仍是seed42、同百万pool、初始10000/1250/1250+126、FNO192/4/24、batch512、100/100/100/200 epochs、warmup5/5/5/10、LR1e-3→1e-5，逐轮冷启动和train-only重拟合。三个Pattern策略各使用两个独立FNO成员；物理策略只需主FNO，额外辅助树/JVP的成本应单独报告。W&B强制offline。

结果新后缀为`gene_ranges_mechanism_v1_full`，不覆盖旧结果。四项尚无正式最终指标。

## 验证入口

`tests/test_al_mechanism.py`：数据隔离、修改标签的有效性、辅助模型重载、稀疏目标与完整贪心对照、canonical RHS四种特殊开关及JVP有限差分、合法连续/整数探针、解析灵敏度系统、分裂位移掩码及真实注册入口。

`scripts/analysis/verify_al_mechanism_integration.py`：保存的真实FNO/独立成员、当前pool/初始数据上的四种选择，及可选的原生ODE查询、追加、manifest/标签/dtype/辅助模型落盘检查。此脚本不训练FNO，也不读取real目标评分。

## 2026-10-31 验证与后台启动记录

- 主链路与新方法回归124项通过；更新微批控制后，新方法/训练预算/离线W&B等59项通过，补充参数依赖初值解析校验后新方法23项通过。此前Fission、数据清单、训练预算及批日志等88项通过；这些集合有重叠，不相加当独立测试数。静态编译、shell语法和`git diff --check`通过。
- 真实已训练FNO和独立成员，从当前池抽取2048个候选；四种注册策略各选64条，选样时不调用ODE、不修改pool、无重复ID。敏感性方法64条原生ODE查询全部成功；Pattern support另64条原生查询和追加成功，检查累计训练行数10064、标签、float64参数/初值、float32轨迹、辅助模型及查询前审计保存。
- 2048个候选中，1555个连续方向、446个整数割线、47个无合法方向；1659个有有效正敏感性分数。该诊断证明运行链路有效，不是采样收益结论。
- 百万候选、3万预算且六类概率全部相同的微批平衡压力测试：12.36秒、3万个唯一ID。这是CPU排名开销，不含FNO推理/ODE。
- 条件稀疏核几何压力测试：10万候选、500背景、190万边、1万预算；构图1.35秒、贪心2.04秒，无重复ID。该项使用合成特征，仅测几何实现，不包含分类器、FNO推理和solver成本。
- 验证记录：`/home/users/zsy/mydata/budding_yeast/analysis/al_mechanism_verification_20261031/`。原生临时快照核对后已删除；正式实验的分类器和审计正常长期保存。
- tmux：`al_budding_mechanism_20261031_114217`；启动清单和输入SHA256在`analysis/al_mechanism_launch_20261031_114217/manifest.json`。四项使用neuralop，输入未重新生成，0–3卡有占用时等待，每卡最多一项，W&B offline。新方法尚无完整结果。
- 11:50:56启动检查：GPU1 Pattern balance已训练至round0第6/100 epoch，train loss从1.2036降到0.5336，均有限、无Traceback，offline W&B正确；GPU0/2/3三项仍排队等待。确认当前载入987500个剩余候选、初始10000条train，未重复ODE生成初始集。这是启动健康检查，不是收敛或最终收益结论。
- 11:59用户要求取消GPU0等待：仅结束尚未启动训练的等待worker736805，在同一tmux新增`sensitivity`窗口，按`WAIT_FOR_GPU=0`和`MAX_JOBS_PER_GPU=2`启动同目录`ours_sensitivity`，与Event同卡并行；Event及三个Pattern进程均保留。启动覆盖记录于原manifest的`scheduler_overrides`和`launch_sensitivity_parallel.sh`，原batch因有意结束等待worker最终可能返回非零，不代表新训练失败；应结合替代launcher及每项训练结果检查。已确认敏感性进程启动并加载已有初始集，训练预算/数据/离线W&B配置不变。
