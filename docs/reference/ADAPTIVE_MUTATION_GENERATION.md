# 三种反馈生成原型

入口复用`run_sampling_budding.sh generate`，实现位于`utils/sampling/ours/adaptive_generation/`；合法单轴重抽编译器位于`utils/sampling/generation/interventions.py`。不是四角生成，不改任何mutation类型、合法强度域、环境、初值或ODE。

| 方法 | 实际原型 |
|---|---|
| ours_response_generate | 同一突变背景/环境的已查询样本局部响应插值与当前预测之差；有多个同路径锚点时使用正则化局部线性响应 |
| ours_event_generate | 用原系统guard方向提取预测穿越次数、首次/末次时刻、事件顺序转移，与已查询事件特征的局部插值比较；不读取Pattern |
| ours_progress_generate | Response分数及生成锚点权重乘以历史区域序贯误差变化权重；无足够反馈时中性处理 |

## 采样及反馈

- 新提案约80%由已查询合成锚点单模块合法强度重抽，至少20%为新背景的原生提案；不支持变化的固定原子、碰撞或重复导致缺额时，有界补入原生探索提案。
- 同模块复合模式按原合法规则顺序重放；触及重叠模块或未声明参数足迹时跳过，不用手工参数插值代替合法生成。
- mdt保持锚点值。重新调用原初值处理，保留float64物理参数/初值；模型输入仍由公共Predictor处理float32。
- 批次选样约20%随机探索；其它点按分数及批内输入冗余惩罚选取。提案探索比例与查询探索比例是两个不同环节，不能相加解释为最终随机样本比例。
- 查询前保存预测，ODE返回后按同一个预测计算误差，不能用训练后的预测替换。失败样本清理等待记录，继续补足有效预算。
- 策略checkpoint保存RNG、轮次、反馈和待揭示预测，恢复时不重复增加同目标轮次。轨迹仍只由公共生成数据分块存储一次。

## 研究边界

- 当前只支持Budding mutation生成；LHS明确拒绝。事件描述适配器可供后续Fission使用，但Fission的合法生成编译和端到端流程尚未实现，不能声称已支持。
- Response误差用固定初始合成训练集RMS缩放的asinh轨迹特征，不改变训练log1p归一化或Relative loss。此分数不是已认证真实误差；局部插值也可能把正常的大响应当作困难点。
- Event记录的是guard穿越代理，并不保证每次预测穿越都对应求解器实际执行的event/reset。
- Progress是不同新查询点的区域序贯误差变化，不是独立探针上的因果学习收益；独立探针/交叉拟合校准仍需后续消融，当前不得将其称为已验证的泛化收益估计。
- 输出prequery_score_error_spearman及实际查询前误差，用于检验分数是否有信息。不能使用real/test表现来拟合分数或挑阈值。
- 三方法是可运行研究原型，不是已证明胜过baseline的新算法，不作论文新颖性认证。

## 批量入口

```bash
bash run_sampling_budding.sh extensions --suite adaptive \
  --protocol /home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/shared/pool100k/seed_42 \
  --directory /home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/adaptive_three_multiseed \
  --seed 42 --seeds 42 123 1024 \
  --gpus 0 1 2 3 --initial-jobs 8 --max-jobs 8 --jobs-per-gpu 2 \
  --io-concurrency 2 --mechanism-workers 2 --mechanism-concurrency 4 \
  --dry-run
```

该suite只包含以上三种方法，mutation、FNO；上述命令联合排9项（seed42/123/1024），每卡最多2项，总计最多8项。去掉dry-run才启动正式实验，沿用冻结协议训练配置。结果位于`experiments/budding_sampling/fno/mutation/<上述方法>/<profile>/seed_<主seed>/`，不会覆盖旧Closure/Transport等结果。W&B仅离线。

多seed联合调度校验每个共享协议的pool、real引用和训练超参一致，任务使用各自主seed的协议与输出目录。明确交接时可用`--adopt-from <旧批次> --extend-adoption`保留旧任务、仅补入新seed；单一调度器负责全部9项，避免重复启动。旧任务保留原I/O锁域，新任务统一另一个锁域；本次交接各2个加载槽，总计最多4个。系统可用内存保留约100GiB。

所有轮次保留指标和loss图，仅最终round生成real逐样本预测图；沿用已审核的I/O压力宽限与只存一次轨迹规则。生成/持续训练入口自身写train.log时，调度器将stdout写批次日志，避免对同一文件重复tee。已交接旧进程不热改日志实现，旧日志中的重复行保留为原始运行证据。
