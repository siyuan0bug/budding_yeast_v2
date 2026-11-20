# Baseline出处更正与复现程度核查

> 后续实现更新：四类方法已接入明确命名的回归/动态/生成适配入口，详细机制和未继承的原文假设见[新实现说明](REGRESSION_SAMPLING_BASELINES_2026-11-17.md)。下方“尚未完成”是更改前盘点，不能替代最新状态；尚无新方法正式性能成绩。

本次为原文与当前生产代码的只读核查，以及文档更正。没有修改算法、生物规则或启动实验。“能运行”“核心公式对应”“原论文完整复现”是不同状态。

## Uncertainty的正确出处

AL4PDE《Active Learning for Neural PDE Solvers》（ICLR 2025）是benchmark，不是QBC首创论文。原文第4.2节明确采用Seung、Opper、Sompolinsky的《Query by committee》（COLT 1992），Eq.5用ensemble成员预测与均值之间的平方差作分歧评分；第4.3节再以Top-K选择高分输入。

本项目应记录为：**QBC / ensemble variance + Top-K；思想来源Seung et al. 1992，当前回归评分/聚合与实现参考Musekamp et al. 2025（AL4PDE）**。不能把1992原理论模型与本项目独立FNO ensemble说成完全相同实现。2025是采用框架的年份，不能当作Uncertainty诞生年份，也不能用它满足“新增2025方法”数量。

依据：[AL4PDE第4.2/4.3节](https://arxiv.org/html/2408.01536v2)、[QBC DOI](https://doi.org/10.1145/130385.130417)。

## 新增五类实现究竟对应到哪里

|方法|当前代码对应|明确差异/未验收事项|准确表述|
|---|---|---|---|
|Clustering Sensitivity|Algorithm1的中心loss＋距离z次项，按分数独立有放回抽样，线性逆概率权重|参数＋初值Z-score作为嵌入；用最近真实样本代替不可查询中心；z=2、lambda=.1、代表点MSE为项目设定；未证明本任务Hölder假设|核心算法的论文重实现及回归任务适配；不承诺原理论保证|
|Clustering Sensitivity WOR|同样的基础分数|改为无放回且不带原重要性权重|额外适配，不能称原Algorithm1的无偏估计器|
|Lev|块内杠杆分数、分步选择和更新|原文文本嵌入换成参数＋初值；分块整数配额/初始化为项目协议|作者机制的输入特征适配；不是LLM预训练实验复现|
|Physics-PRE|Eq.5平均绝对残差及最近已查询真轨迹参考残差归一化|PDE换ODE；最近邻用训练Z-score距离，参考floor=1e-6；预测事件mask±4，不检查完整跳跃映射|事件ODE任务适配，不宣称原文处理了本系统全部事件|
|OED-Temporal|线性高斯后验更新、递减探索UCB、逐个真实反馈|随机共享初始集替代原LHS；97维字典、时间方差log1p＋高频比例、新合法bank内近似优化为项目定义|核心采集机制重实现，信息量与搜索域适配；不是论文原PDE全复现|
|OGAS-Round|v1条件扩散/CFG/EMA/FIFO/密度比/反馈|分轮冷启动、项目Relative难度、合法投影；不是原连续在线|分轮适配版本|
|OGAS-Online|依据v2的持续训练反馈、异步求解、后台DDPM、密度比稳定化、replay|完整轨迹、CPU width128而非论文1024、合法投影、冻结初始统计、项目validation-best；未复现Melissa速度限制器/Unc变体|异步在线任务适配，非作者系统完整复现|

现有公式及小规模实现/恢复测试可支持上述有限声明，不能替代对作者全部配置、数据和大规模结果的验收。参考源码不在运行时import/exec；独立实现不等于可以自行改变核心选样规则。

## 尚未完成的四项，不等于不适合或决定排除

|方法|正式来源|当前实际缺口|后续必须保留|
|---|---|---|---|
|PEAKS|Gurbuz等，ICML2025|尚未实现轨迹回归评分、流式cache及阈值/refresh完整链路；作者主实验分类，但原文明确数学分析也适用于回归|误差×kernel影响、增量流和刷新；回归应按实际损失导数推导，不能拿Pattern当类别冒充原轨迹任务|
|HGGS|Rao等，**AAAI2025**；本地PDF为2024预印本|只有residual_topk，尚无完整GF＋MGS；原输出振荡频率，本任务输出39变量轨迹|边界/梯度过滤、遗传多网格生成、查询反馈及费用；生成只调用既有合法规则/初值/ODE|
|ERACS|Cheng等，NeurIPS2025|尚未实现训练期间梯度SNR和OMP带权重选回调；不是固定五轮只追加|训练中检查和动态替换，记录历史唯一曝光/访问真值，不能只按最终驻留数计费。论文Alg1条件SNR<threshold与实验段“exceeds”文字不一致，实施时需对照公式/源码确认|
|Low-Rank|Cohen-Addad等，ICML2026|Algorithm1需要basis-loss；本任务在任意线性基方向上没有已定义的物理真轨迹标签，定义尚未落实|行子集/低秩投影残差、basis-loss、原敏感度与有放回权重；不能只做SVD分数后换名|

优先补齐HGGS、PEAKS和ERACS；Low-Rank先明确loss定义及其适用假设。生成/动态流程不同不是拒绝理由，但要分别计ODE尝试、访问真值、历史训练曝光和训练步数。所有实现及调参继续隔离real/test。

## 年份修正

此前按本地文件名把HGGS写成2024，遗漏了正式AAAI2025发表，现更正。它是符合2025近期要求的候选，但尚未实现完整流程，不能计作已有结果。

- [HGGS AAAI2025](https://ojs.aaai.org/index.php/AAAI/article/view/34920)
- [PEAKS ICML2025](https://proceedings.mlr.press/v267/gurbuz25a.html)
- [ERACS NeurIPS2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/92a035beb61e68f7a688467e5103a17c-Abstract-Conference.html)
- [Low-Rank ICML2026](https://proceedings.mlr.press/v306/cohen-addad26a.html)

OGAS等既有实现的详细接口/验证记录见[此前核查说明](BASELINE_FIDELITY_AND_OGAS_ONLINE_2026-11-17.md)。本次没有新增测试运行或正式实验。
