# 2026-10-31 mechanism批次与Event重跑完成检查

截至2026-10-31 16:34（Asia/Shanghai），四项mechanism新实验及Event重跑全部正常完成。逐项训练进程退出码均为0；检查实际checkpoint和数据，不仅依据终端“完成”字样。

四项新实验11:42:57开始，16:30:27最后一项结束，整批跨度4小时47分30秒；Event此前10:21:49启动、13:59:39结束。多GPU并行，不能将单项耗时相加当成批次墙钟时间。

## 完整用时与最终real指标

均为FNO192/4/24、batch512、gene_ranges池、seed42、7万合成train、126株real，采用实际最后一轮权重。完整用时从任务START到END，不含候选池/初始集预生成及任务启动前的GPU等待。

| 方法 | 完整用时 | MAE | MSE | RelL2 | Corr |
|---|---:|---:|---:|---:|---:|
| ours_sensitivity | 4小时31分12秒 | 0.113218 | 2.020053 | 0.176110 | 0.960775 |
| ours_pattern_balance | 3小时15分51秒 | 0.134670 | 2.908884 | 0.193806 | 0.954820 |
| ours_pattern_context | 3小时17分48秒 | 0.087077 | 0.945067 | 0.145252 | 0.973121 |
| ours_pattern_support | 3小时16分59秒 | 0.092608 | 0.979995 | 0.153649 | 0.970543 |
| ours_event | 3小时37分50秒 | 0.115966 | 2.217988 | 0.177114 | 0.960494 |

## 核对证据

- 五项均四轮完整，历史epoch序列分别1..100/100/100/200，合计500个主模型epoch；末轮确实训练200epoch。
- 每轮checkpoint实际global_step核对为1900/3900/7800/27200，总40,800次更新，与batch512、drop_last=True一致，没有少跑或提前停止。
- 合成train分别10000/20000/40000/70000；三次有效追加10000/20000/30000，共60000条。候选ID没有重复，也没有与初始train/val/test候选交叉。
- 固定val/test/real为1250/1250/126，跨轮及四项新方法的样本身份/顺序一致，切分不重叠；manifest来源SHA校验通过。没有重新逐点审查所有大轨迹。
- 归一化均为train_synthetic/log1p，fit数量与当轮train匹配、统计量有限且尺度为正；轨迹float32。
- Pattern三个方法及Event，前三轮独立ensemble成员checkpoint训练步数匹配，额外训练合计300epoch；sensitivity按注册能力只使用主模型，不应要求其存在ensemble checkpoint。分析工具已改为读取registry能力，避免误报缺文件。
- 四项新方法每次追加的查询前预测文件齐全；accepted候选均有对应查询前ID、没有重复尝试ID。Pattern概率有限、非负、归一化，连续描述有限。辅助分类器三次训练行数10000/20000/40000，训练ID摘要与相应轮次实际train完全匹配。物理方法保存分数有限。
- 每轮126幅real预测图、loss图、物理指标JSON齐全；real 126/synthetic test 1250，汇总指标与逐轮结果一致、数值有限。
- 完整日志检索未发现Traceback、CUDA OOM或本工具检索的失败/超时重建信息。

## 为什么看起来快

| 方法 | 主模型训练 | 额外ensemble训练 | 候选评分/选样 | ODE进程池 |
|---|---:|---:|---:|---:|
| ours_sensitivity | 1小时26分27秒 | 0小时00分00秒 | 2小时03分31秒 | 0小时38分08秒 |
| ours_pattern_balance | 1小时23分50秒 | 0小时29分40秒 | 0小时15分20秒 | 0小时43分46秒 |
| ours_pattern_context | 1小时23分42秒 | 0小时28分22秒 | 0小时17分28秒 | 0小时39分35秒 |
| ours_pattern_support | 1小时21分14秒 | 0小时30分33秒 | 0小时17分19秒 | 0小时40分12秒 |
| ours_event | 1小时33分06秒 | 0小时28分21秒 | 0小时21分05秒 | 0小时55分42秒 |

以上stage为日志中的分段累计，不含初始加载/校验、归一化、checkpoint、绘图/评测和数据写入，故与完整用时存在合理差额。百万pool仅存参数/初值；每项在线新增6万条ODE，并非对100万候选逐个仿真。FNO主训练约81–93分钟/项，Pattern查询前推理/选样约15–17分钟，敏感性JVP/模型探针明显更贵，约2小时03分。输入已预生成，多CPU ODE、共享初始块和增量数据存储也避免了重复工作。

## 总启动器exit=1的解释

原四项启动器返回1，是11:59按用户要求有意终止GPU0的等待worker造成。该worker当时尚未启动训练；其余三个Pattern进程没有中止。随后同一tmux的独立sensitivity窗口以WAIT_FOR_GPU=0与Event同卡并行，独立启动器退出0，sensitivity训练进程也退出0。原始退出记录保留，不篡改为成功；应结合manifest.scheduler_overrides、替代启动器和逐项结果判断。

## 结论范围

正常完成训练预算不等于严格收敛；四项新实验convergence.json仍标记still_changing，诊断没有触发早停。当前全部按既定预算完成，无需因“太快”重跑。性能上context在本批四项中最强，但不能据此声称全面胜过全部baseline，或仅用seed42给出稳定泛化结论。126 real参与过方法开发的历史继续披露。

## 可复核输出

- `analysis/al_mechanism_completion_20261031/integrity.json`、`completion.csv`：四项完整性、实际训练预算、分段耗时及最终指标。
- `analysis/al_eventfix_completion_20261031/integrity.json`、`completion.csv`：Event同口径检查。
- 分析入口：`scripts/analysis/audit_al_batch_completion.py --manifest <launch/manifest.json> --output <analysis/output>`。
