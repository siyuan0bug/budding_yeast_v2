# OGAS采样公式与来源标记修复

日期：2026-11-18。对应[原文／源码一致性审计](../../../papers/baselines/sampling/BASELINE_SOURCE_FIDELITY_AUDIT_2026-11-18.md)第8节。只修复已确认的两处非必要差异，以及它们需要的来源标记与恢复链路；不修改生物规则、采样强度分布、ODE、已有候选池或历史实验。

## 1. 其他baseline也存在适配，不能说只有OGAS不完全一致

|方法|已核实的情况|本轮处理|
|---|---|---|
|Random、Power、LCMD选择内核|固定池随机；Power/LCMD同输入选择分别90/45组一致|保持；表示、训练协议仍需注明|
|HGGS|论文Eq.4是距离比平方，作者源码是未平方距离比；项目遵循论文|保留论文版本，明确源码差异|
|PEAKS|作者公开主代码分类；项目用参考Relative L2梯度内积做回归评分|保留限定名称，不称官方回归等价版|
|Low-Rank|采用梯度平方能量方向目标，不能直接继承原轨迹loss保证|保留GradientEnergy名称及理论边界|
|Physics-PRE|预测残差／参考归一化结构保留，ODE事件屏蔽为适配|保持事件适配说明|
|OGAS|CFG参数记号混用；密度上限混淆两种比例|本轮修复，并验证完整链路|

## 2. CFG：相同配置值现在具有作者定义的含义

对照作者Melissa提交`be91967d3af39da5d1a56b58f8eeb7b89763a42c`中的`active_sampling/models/ddpm.py`。

```text
修复前：epsilon = (1+w)*conditional - w*unconditional
修复后：epsilon = unconditional + w*(conditional-unconditional)
```

修复前设置`w=5`等效于作者公式的6；现在5就是作者的5。`w=0`产生无条件预测，`w=1`产生条件预测。该修复作用于共用DDPM核心，分轮版和在线版均使用。

这会改变原始生成目标及后续选中的合法样本；仅凭公式不能保证real指标必然改善，效果需要使用修复后的版本重新实验。

## 3. 密度校正：分类先验和实际参考比例分开

两个比例分别是：

- `discriminator_reference_prior`：判别器训练时reference类别占比。当前reference和buffer各一半，因此固定0.5。
- `buffer_reference_fraction`：本次生成器反馈小批中，来自随机参考提案的实际比例。根据保存的来源标记计算，不使用固定探索率代替。

固定为0.5的只有第一项。第二项需要随数据实际变化。

```text
log_ratio = reference_logit + log((1-class_prior)/class_prior)
pi_u = max(actual_reference_count / batch_size, 1/batch_size)
raw_weight = min(exp(clip(log_ratio, -5, 5)), 1/pi_u)
weight = raw_weight / mean(raw_weight)
```

`1/batch_size`是作者对全生成批次的保护。这个上限约束均值归一化之前的权重；归一化后的单个权重可以高于该值。

例：100个反馈中20个来自随机参考提案，上限为5；50个来自参考提案，上限为2。判别器训练这两个批次时仍然各使用一半reference、一半buffer。

此前审计的三点数值反例忽略了小批次的`1/m`下限，修订后的报告采用四点、参考占比0.25的合法例子；差异结论不变。

## 4. 来源标记贯穿仿真、反馈、缓存与恢复

- 初始共同随机训练集标为reference。
- 生成器引导查询标为非reference。
- 随机探索标为reference。
- 生成器未就绪而随机补位的样本也标为reference，不能误记为生成器引导。
- 分轮版保留每次选择的探索标记，在线版在提交仿真前把标记写入候选metadata。
- 成功仿真按实际返回的候选metadata入库；失败样本不进入反馈，因此不会因失败／乱序错配标记。
- `candidate_metadata_json`保存`ogas_is_reference`；在线ODE账本记录`is_reference`。
- replay返回`online_is_reference`；训练反馈与参数、初值、难度一起传入后台线程。
- FIFO淘汰、后台待处理反馈、controller和replay恢复均保留对齐的布尔标记。

只增加每条反馈／缓存样本的布尔来源信息，不重复保存轨迹，不增加仿真调用或实验并发。

## 5. 旧结果与恢复

新controller状态标记`implementation_version=author_cfg_and_reference_provenance_v2`。旧状态缺少正确来源信息，直接拒绝恢复，不默认把全部旧反馈当reference。项目已有源码指纹参与结果目录身份，因此重新启动会使用新的运行身份；不覆盖旧结果或把新旧采集过程接在一起。

旧OGAS指标仍是旧实现的结果，修复不会追溯改变它们。若要将OGAS作为修正后的正式baseline，需要重新运行对应OGAS实验；本轮没有启动正式实验，也没有重生成完整候选池。

## 6. 验证范围

新增作者公式约定、实际DDPM采样路径、不同类别先验／参考比例、全生成批次下限、FIFO淘汰、异步待处理反馈恢复、replay覆盖恢复及随机补位／失败乱序来源测试。

同时运行已有小规模真实canonical ODE→FNO→评测→中断恢复、分轮生成、独立采样及其他回归baseline测试。**最终69项全部通过**；另检查了实际写出的在线／分轮NPZ，来源布尔标记已写入`candidate_metadata_json`。82个Budding生物／data根目录原文件SHA前后变化0，其他采样方法源码未改。记录见临时证据目录`scratch/probes/ogas_alignment_fix_20261118/verification.json`。

修复后仍是明确的酵母任务适配：完整轨迹反馈、合法域投影、生成器容量、归一化与checkpoint协议等，与作者PDE全套实验不同。此次修复不将其升级表述为原样全复现。
