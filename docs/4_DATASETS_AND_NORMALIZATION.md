# 数据、seed与归一化

## 当前完整池

两个池各100000条完整ODE轨迹，生成seed均为42；当前使用数值修补视图，原始池不覆盖。
完整轨迹只保存在外部`/home/users/zsy/mydata/budding_yeast/offline_pools/`，每个实验不得复制累计轨迹。
mutation池按当前合法模块/模式规则随机生成，LHS按已审核混合边际规则生成；不能把旧Sobol或集合配额规则写成当前池设置。
具体字段见[样本契约](reference/DATASET_SAMPLE_CONTRACT.md)，生成入口见[生成指南](reference/DATA_GENERATION_QUICKSTART.md)。

## 共享划分与随机流

共享协议在`experiments/budding_sampling/shared/pool100k/seed_<实验主seed>/`，包含协议、输入缓存、划分和seed记录。
每池默认90000训练侧候选、5000验证、5000测试；普通初始训练2000从训练侧随机选取。
同一有效输入跨池出现时应保持角色一致；不能在一池训练而在另一池当独立测试。
生成seed固定42；实验主seed独立配置，按用途派生split/initial/model/selection随机流。当前正式结果主要是主seed42，2026/123属于计划，不得写成已完成。
普通同协议方法共享初始集；HGGS等原生初始化不同，另报真实查询成本与历史曝光。

## 编码与存储

参数逐维Z-score，轨迹变量log1p后逐变量Z-score，统计只由合成训练数据拟合。
普通逐轮冷启动可按累计训练集重拟合，持续训练方法按其原生协议冻结；加载checkpoint必须恢复其保存统计。
物理参数与实际初值可存float64，模型缓存编码为float32。字段精度不能随意变更。
`dataset.json`是来源、哈希和有序行号引用，不是新的轨迹副本。使用`data/dataset_archive.py`读取，不能对JSON直接np.load。
新增生成轨迹分块写一次，当前样本配置、轨迹、标签与mutation来源必须对应。

## 生物与求解约束

未授权不得改变mutation、初值、ODE、事件和标签；data根原始代码继续保护。
数值成功、完整积分、非有限与已知总量约束遵循当前生产政策；已知数值异常同输入精确复算，失败补采并记账。
有限微小负值按已确认规则保留，不恢复real最大值两倍过滤，不改变既定超时保护。
