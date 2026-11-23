# 干预阶数分布校准：有停止门槛的诊断

结果状态：seed42筛选阴性，后续确认按门槛跳过，独立审计/正式test/real未开放。三个最终模型完整结束，没有运行错误。分层结果及解释见[报告](/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/summary/intervention_order_calibration_v1/report.md)。未获得有效新算法的证据。

目标是检验“代表原候选池”与“学习不同阶数干预”是否冲突。固定L配额本身不声称新算法；只有基础假设得到独立证据，才研究自适应跨阶数预算分配。不能反复查看real来挑比例或改门槛。

原创性边界：自动优化数据混合比例本身已有[DoReMi（NeurIPS2023）](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dcba6be91359358c2355cd920da3fcbd-Abstract-Conference.html)及[DoGE（ICML2024）](https://proceedings.mlr.press/v235/fan24e.html)等工作。不能把文本domain换成L分组就宣称新算法；若本轮阳性，下一阶段还必须证明干预阶数间的机制依赖/迁移结构提供了通用重加权无法解释的额外收益。

## 三个对照

保留源pool、mutation种类、强度分布、初值、ODE、seed共享划分和共同初始2000条。只改变最终10000条中各实际模块数量L的分配；同一基因复合mutation只算一个模块，mdt不是基因模块。层内均匀无放回，容量不足不重复凑数。

| 方法 | 配额规则 |
|---|---|
| calibration_pool_random_v1 | 按合格源池比例无放回随机 |
| calibration_order_balanced_v1 | 各L最终尽量等额；受初始数量和有限唯一输入容量约束，剩余分给尚未满额的层 |
| calibration_low_order_first_v1 | 初始2000条保留，先用完合格L1，再分给L2，最后才给L3 |

三组最终都从头训练595epoch/47005步，batch128、FNO192/4blocks/24modes、LR5e-4、warmup30、minLR1e-5。训练Relative/eps0.01、不截断、梯度裁剪、逐训练集拟合log1p/Zscore均沿用。best仍由来源合成验证归一化loss选择，不用real或研究反馈挑epoch。不同训练集的自身归一化loss不直接比较；反馈比较统一使用该seed共同初始2000条的统计。

## 信息隔离

仅从训练侧候选INPUT的模块/模式/mdt完整背景分组。排除初始2000条出现过的完整背景，然后每L独立随机保留反馈与审计背景，各取不超过剩余背景15%、最多64种、每背景最多8条。反馈与审计的完整背景不进入任何训练集，也从checkpoint验证loader中排除。原共享protocol及其validation/test划分文件不改，研究工作目录记录验证loader实际子集及被排除数。

研究反馈用于选择quota和决定是否继续；独立审计只有确认通过后开放一次。两者按L1/L2/L3各占1/3计算风险，不使用real的L比例。记录共同初始坐标的逐样本逐变量Relative，以及物理MAE、MSE、当前物理Relative。正式test与real在筛选/确认/审计决策中不可见。

## 顺序与预先声明的门槛

1. 输入准备支持42/123/1024，先仅训练seed42的三个对照。
2. 每个quota相对随机：共同坐标均衡Relative改善至少5%，均衡物理MAE/Relative均不得变差超过5%，任何L的共同坐标Relative不得变差超过10%。从通过者中选择共同坐标风险最低者并写入不可变selection_decision；没有通过者立即停止。
3. 只将这一个预选quota与随机在123/1024上确认，不根据后两个seed重新换quota。合计最多7个最终模型。
4. 三seed确认：平均共同坐标均衡Relative改善至少5%，至少2个seed改善，平均均衡物理MAE与Relative都不得变差，任何L平均共同坐标Relative不得变差超过10%。失败则停止，独立审计仍不开放。
5. 确认通过后锁定六个checkpoint哈希，再开放独立审计一次，使用相同严格门槛。只有审计通过，才给已锁定的quota与随机作三seed正式test/real终评和绘图。不依据终评改变规则或重新挑epoch。

这些是筛选研究价值的门槛，不是已证明统计显著、强baseline优势或论文创新。若假设阴性，记录负结果，不继续扩大同一方案、不为了得到阳性结果修改门槛。

## 代码与产物

- 纯输入分组/配额：utils/sampling/ours/intervention_calibration.py。
- 工作流适配：utils/sampling/benchmark/calibration_study.py，复用公共fit_round、归一化和评测。
- 通用依赖调度：utils/experiments/study_queue.py，复用BatchMemoryGuard；只管理本批进程。
- 阶段CLI：scripts/analysis/run_intervention_calibration.py；批量CLI：scripts/launch/run_intervention_calibration.py。
- 共享来源/角色/有序ID：shared/pool100k/seed_<seed>/studies/intervention_order_calibration_v1/；不复制轨迹。
- 单项训练与反馈：fno/mutation/calibration_<method>_v1/<profile>/seed_<seed>/。
- 汇总及不可变选择：summary/intervention_order_calibration_v1/。阴性结果没有real图是预期设计，不是漏评测。
- 批次状态/资源/启动日志：batches/。0～3卡，单卡最多2项，I/O并发4，系统有效可用内存保留约100GiB；W&B只离线。实际上首轮仅3项训练，确认最多4项，资源不用于增加无关分支。

## 统计来源的配对控制

原配额筛选阴性结论保持不变。补充诊断只复用seed42随机组与低阶优先组的有序10000条训练ID，各新增一个模型：两者统一使用共同初始2000条训练样本拟合的参数及轨迹统计。与原有各自10000条拟合统计的模型组成2×2对照。训练预算、模型seed、损失、验证ID和checkpoint选择规则均不变；不生成新样本，不重新抽配额，不读取独立审计/正式test/real。

它检验的是整体归一化政策的影响，包括输入编码、目标编码及validation-best选择，并非仅隔离损失分母。默认正式流程仍按当前累计训练集拟合统计；这里的冻结统计只是显式诊断。公共训练适配器通过已存在的preprocessing_state恢复接口替换统计，保持当前数据及划分契约，并将统计来源ID和指纹写入checkpoint身份。

两项均训练595epoch/47005次更新，与原对照相同。若共享统计后低阶优先仍更差，停止固定L配额方向；若表现反转，只能说明原归因不充分，不能宣称得到有效或创新采样算法。该对照帮助排除混杂因素，本身不是mutant-aware sampling的创新。

入口：`scripts/launch/run_normalization_control.py`，复用通用资源调度与训练服务；0/1号卡各一项，系统有效可用内存保留100GiB，W&B离线。结果位于`fno/mutation/calibration_{pool_random,low_order_first}_fixednorm_v1/<profile>/seed_42/`；比较报告位于`summary/statistics_only_control_v1/`。批次仅保存任务状态和资源记录。
