# Budding AL候选池两级均衡（2026-10-28）

> **当前批实验配置**：直接运行现有`run_al_budding.sh`，默认复用已生成的balanced百万池；新balanced初始集按需生成并在16项实验间共享。结果目录的设置段使用`_refit_balanced`，W&B默认开启且强制offline。仅更新原脚本配置，本轮未启动数据准备或训练。

用户确认的方案已接入生产，不再由旧5/40/55层比例和随机模块/模式决定Budding AL候选池。静态数据集及Fission采样不变；原始data根目录代码、旧池、旧初始集及已有实验保留。

## 生成和复用入口

- 生产逻辑：data/systems/budding/al_pool.py；枚举与容量分配公共函数：quotas.py。
- data/generate.py --recipe budding_pool和AL运行时兜底均调用同一逻辑；串行、多进程只改变执行方式，相同seed和配置生成同样的候选及顺序。
- run_al_budding.sh默认池：/home/users/zsy/mydata/budding_yeast/budding_pool_1000000_seed42_balanced.npz。
- 新默认初始集：budding_initial_train10000_seed42_balanced.npz；本轮未生成。之后运行脚本时才会从新池进行KMeans、仿真和固定切分，不复用旧_al初始集。
- 新系统导出的旧配额池会在AL和初始集入口被拒绝。初始集记录源池路径、大小、mtime、契约和catalog身份，源池变化后拒绝复用旧初始缓存。
- 训练仍为40/40/40/100 epoch、warmup4/4/4/8；本轮没有启动训练。

## 数量如何决定

1. 枚举当前22模块、合法基础/复合模式；用TOUCH规则排除跨模块参数冲突，不再用随机重抽决定组合。
2. 22/230/1520个合法L1/L2/L3模块集合尽量等额。25个只有有限输入的集合精确计容，扣除real碰撞及物理等价输入后，容量不足的饱和，余额重新均分。
3. 每个集合内，在最终接受计数最少、仍有容量的合法模式组合中选一个。有限模式耗尽后，剩余配额由其他模式均衡承担；不重复输入填数。
4. 按模块集合/模式/seed/重试号派生独立加扰Sobol流。连续/离散强度按原mutant-aware逆CDF映射；CDC14-ts内部支路、APC-A和TAB6-1等内部均匀抽样也使用该行独立Sobol坐标，不改变赋值公式或分布支持。
5. 小有限支持（如固定KO、单MC加固定背景）使用原MC权重1/N与非GAL环境80/15/5形成条件CDF，不放回选取。接近容量时最终频率必然偏离有放回的原始概率；不会为凑原始概率重复输入。
6. 有GAL仍固定mdt150；其余环境提议80/15/5。全池real排除、唯一去重和有限容量会使最终环境比例与提议比例不同。
7. 连续提议碰到real时使用既有70%软化KO/否则同模块重抽的救援，不改模块集合或mdt。救援结果按实际模式记账；若其模式单元已超额则拒绝并重提。有限支持在枚举时已剔除real，不产生一个待救援的实际候选。
8. 协调器按固定模块顺序做全局effective_inputs唯一性检查（含初值与事件控制）；重复在原模块内补提，缺额明确失败，不跨集合静默填充。最后按seed打乱文件顺序。

百万池：L1=9815，L2=126225，L3=863960。无限容量集合各571或572条；BCK2 L1为88、SBF_MBF L1为3。模块集合等额不等于单模块总出现次数或每种基础模式包含率完全相等。

契约：pool_generation_contract=budding_module_allele_equal_capacity_sobol_v1，写入NPZ顶层和报告；catalog_version沿用当前catalog。generation_config中的level_ratios是实际模块配额推导结果，不再伪报5/40/55。

## 正式生成和审计状态

已使用48个CPU worker、seed42生成100万条。文件无轨迹/轨迹标签，保留完整参数、精确初值、mutation和来源元数据；详细字段见DATASET_SAMPLE_CONTRACT.md。

生成报告：effective输入唯一1000000；全局重复0；局部重复265已重提；无模式耗尽造成的缺额。第一次百万提议在报告汇总阶段遇到Counter字符串字段错误，未发布文件；已修复为字典式元数据写入，新增完整生成—导出测试后重新生成。失败与成功日志都保存在机械盘审计目录。

Top20来自既有twenty_badcase_evidence.csv，不按新池结果重选。compare_balanced_pool_coverage.py复用历史参数盒判定：±25%/±50%，最小宽度为5%WT，严格零参数保持零且mdt相同；另统计允许1%WT残余的零参数邻域。补充联合初值匹配（rtol1e-6、atol1e-8），最近距离对两个池统一使用旧池p_std标准化。此审计不做轨迹仿真、不重训，不将输入邻近直接解释成动力学或预测相似。

输出目录：/home/users/zsy/mydata/budding_yeast/analysis/balanced_pool_2026-10-28。

固定Top20覆盖对比已完成：严格25%邻域2例增加/9例不变/9例减少，合计155→38；近零放宽后183→54。均衡并非这些case的局部覆盖最优，完整20例与限制见[BALANCED_AL_POOL_COVERAGE_2026-10-28.md](BALANCED_AL_POOL_COVERAGE_2026-10-28.md)。实际产物随机抽取5000条通过AL加载器的输入/初值/real碰撞检查，配额来源保留。
