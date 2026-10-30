> **2026-10-30 存储格式更新**：新AL改为共享初始数据＋新增块＋`rounds/round_XX/dataset.json`，不再重复写累计轨迹。训练/评测/相关分析支持新旧格式，原始轨迹精度不变；本批14项已完成逐字段验证转换，173.13→56.65GiB（含共享初始），释放116.48GiB；首末轮真实checkpoint重载通过。[布局、读取和验证](AL_DATASET_STORAGE_LAYOUT.md)。下方空间数字为转换前审计快照。

# AL收敛与实验存储审计

检查日期2026-10-30，状态刷新17:10，空间快照17:14。当前14项中8完成、6失败，ours_event也已完成。本轮只读取结果并保存小型分析产物，没有删除、硬链接、重压缩、迁移正式数据或改动训练代码。

产物目录：`/home/users/zsy/mydata/budding_yeast/analysis/al_storage_convergence_20261030/`。分析入口`scripts/analysis/audit_al_storage.py`，输入为experiments目录和本批启动manifest，输出包含全部219个有config的实验目录及未归属文件的清单、14项分类、NPZ尺寸、50个完成轮次的变化率和布局估算。大小按实际文件分配块计数，不跟随符号链接；全树总量跨硬链接去重，目录块未计入，故与du有少量差异。单位GiB=2^30字节。

## 100/200 epochs的收敛证据

每个数为末10epoch对11个log(loss)值拟合得到的每epoch下降百分比，正数为下降。监控只使用合成训练loss与合成验证MSE，不用real/test来决定停止。

|训练轮次|训练量|epochs|训练loss下降趋势范围|验证MSE下降趋势范围|
|---|---:|---:|---:|---:|
|round_00|10k|100|0.0861%/epoch|0.0315%/epoch|
|round_01|20k|100|0.1240%–0.1521%/epoch|0.0239%–0.0717%/epoch|
|round_02|40k|100|0.1704%–0.1973%/epoch|0.0126%–0.1010%/epoch|
|round_03，8项完整结果|70k|200|0.0520%–0.1068%/epoch|−0.0102%–0.0631%/epoch|

round_00的14项数据、主模型seed完全相同，不能当成14份独立实验。20k/40k的100epochs仍有清晰的训练loss下降；验证大多已较平，但不能说训练优化已经全部结束。最终200epochs显著更接近当前LR计划下的平台。

|最终轮方法|训练趋势%/epoch|验证趋势%/epoch|连续平台检查|
|---|---:|---:|---:|
|Random|0.0758|0.0454|1/3|
|MaxDist|0.0520|0.0025|1/3|
|marginal|0.0666|0.0631|1/3|
|event|0.1068|0.0293|0/3|
|feedback|0.0671|0.0438|1/3|
|error|0.0863|−0.0102|1/3|
|interaction|0.0716|0.0071|1/3|
|coverage|0.0902|0.0114|1/3|

当前经验判据要求训练/验证趋势绝对值都低于0.1%/epoch，连续3次每10epoch的检查。8项均未通过完整规则，event最后一次仍稍高于训练阈值。正确结论是“多数最终轮接近平台，尚未确认所有轮次稳定收敛”，不能把still_changing解释为严格数学证明未收敛，也不能承诺加到某个固定epoch就必然收敛。LR降到底同样会缩小变化率。

100/200可作为统一计算预算作方法比较，但论文不宜写“所有方法充分收敛”。若需进一步确认，应固定现有合成数据与验证集，选择代表性40k和70k checkpoint作短段优化续训检验，明确optimizer/LR的恢复规则；不重新选样、不用real作停止依据。本次没有执行续训或改变预算。增加epoch主要增加时间，当前每轮只留一个主checkpoint，磁盘大头不会随epoch成比例增长。

## 管理员额度与当前磁盘

`/home/users/zsy/mydata`是指向`/data/zsy`的符号链接；项目内experiments同样链接到机械盘。

|范围|空间|
|---|---:|
|/data整块文件系统容量|14.44 TiB（df显示15T）|
|当前普通用户可用空间|40.33 GiB（df显示41G、使用率100%为取整）|
|个人/data/zsy数据树|约859 GiB|
|全部experiments文件分配块|516.66 GiB|
|其中al_budding全部历史|396.14 GiB|
|本次14项长程目录|173.13 GiB|

整个文件系统容量和剩余容量不是个人分配额度。系统没有quota命令；对当前UID1027查询/data及/home/users的用户quota，均返回ESRCH（该类quota未启用）。数据目录project id为0，挂载选项没有usrquota/prjquota。因此没有查到可确认管理员给zsy多少额度的配置；口头/登记约定无法由这些结果推断。不能说15T全给了当前用户，也不能说目前只占859G就还有十几T可以使用。inode仅约2%使用，问题是字节容量。

40GiB不足以再启动一批当前布局的14项实验；单项完整目录约13–14GiB，约3项就几乎吃满余量。/home/users当前另有共享空间，但不是本用户的确认配额，本轮没有把实验自动迁回项目盘。

## 每项当前空间

失败目录同样保留了数据，不能按退出码推断它们只占很少空间。

|方法|状态|GiB|
|---|---|---:|
|Random|完成|12.92|
|MaxDist|完成|13.00|
|marginal|完成|13.25|
|event|完成|13.24|
|feedback|完成|13.72|
|error|完成|13.58|
|interaction|完成|13.17|
|coverage|完成|13.65|
|Uncertainty|失败|7.42|
|Power|失败|7.39|
|LCMD|失败|12.79|
|BAIT|失败|12.83|
|原版ours_response|失败|13.08|
|invariance|失败|13.09|

全部静态/历史/失败实验的逐目录清单见all_experiment_sizes.csv，按占用降序列出，避免只统计本批而漏掉旧结果。

## 空间为什么大

本批分类汇总：累计dataset快照107.60GiB，追加added_samples42.87GiB，initial副本9.67GiB，主checkpoint4.08GiB，ensemble checkpoint2.45GiB，图片4.71GiB，反馈等其他NPZ1.38GiB，W&B0.081GiB，其余约0.29GiB。前三种轨迹文件共占92.5%。

Random单项示例：

|内容|GiB|
|---|---:|
|data/initial_dataset.npz|0.691|
|round_00/dataset.npz|0.691|
|round_01/dataset.npz|1.238|
|round_02/dataset.npz|2.331|
|round_03/dataset.npz|3.972|
|三次acquisitions/added_samples.npz合计|3.281|
|全部主checkpoint|0.326|
|图片/PDF|0.373|

初始轨迹反复进入各轮累计文件，新增轨迹也既保存在追加文件、又进入之后每份累计文件。NPZ已经是压缩的float32轨迹；float64参数和初值很小，删除它们或降低精度不能解决主要问题，还会破坏精确来源。W&B只占本批约83MiB，关闭它几乎不能缓解空间问题。

## 建议的无损布局（未实施）

```text
shared/initial/<内容哈希>.npz      # 同一初始数据只存一次，固定为只读
shared/pools/<内容哈希>.npz        # 候选池只引用，不复制到每个实验
<experiment>/
  data/manifest.json             # 初始集、pool、schema、哈希及固定切分
  acquisitions/after_round_XX/
    added_samples.npz            # 每次新轨迹只保存一份
    selected.csv / report.json   # 含查询与失败/消耗ID的原有审计
    query_design/feedback文件    # 维持策略复现所需证据
  rounds/round_XX/
    dataset.json                # 引用初始集与此前追加块、精确行顺序/切分
    model.ckpt / preprocessing.pt
    training_history.csv / convergence.* / loss图 / eval/
  summary/                      # 保留汇总指标、四轮loss大图
  wandb/                        # 保留本地监控
```

该方案保留所有params、float64实际IC、mutation/override元数据、轨迹、时间轴、L、生物/ABC/六分类标签、每轮样本身份和所有图片/模型；变更的是物理存储方式。

按已有压缩文件大小估计，本批173.13GiB可降至约55.86GiB，加少量manifest，约节省117.27GiB（67.7%）。完整单项可从13–14GiB降至约4.0–4.8GiB。估算共享initial已在外部保留，不重复收费；不假定各方法新增轨迹之间可以去重。另一种互斥方案是只存最终NPZ、各轮保存精确行视图，预计节省107.60GiB；不能与增量块方案的节省量叠加。

低侵入第一步可考虑初始副本及round_00去重：15份initial候选与14份round_00候选分别具有一致ZIP中央目录CRC/size签名，两组的首尾代表文件SHA256均相同。预计可合并约18.65GiB，但不是所有候选都已计算SHA，真正变更前仍须逐文件完整哈希验证。内容哈希只读共享文件可在同一分区用硬链接保留原路径；不能把未来会被原地覆盖的可变源文件直接作为共享源。

**当前不能直接删各轮dataset.npz**：eval.py会自动寻找checkpoint旁边的NPZ，DataModule又核对checkpoint中保存的完整数组哈希、行序及切分。直接给早期checkpoint传最终NPZ会拒绝加载。需要先实现兼容旧NPZ的manifest loader，精确还原每轮视图，逐轮检查数组/标签/切分/哈希和checkpoint评测一致，再作受控迁移。剩余40GiB也要求迁移逐项验证，避免先复制全批造成空间耗尽。本轮没有实施该接口改造或删除历史实验。

分析器已通过实际目录扫描，以及临时小型硬链接/符号链接/分配块计数校验。所有估算只计无损结构压缩，没有通过删除badcase、裁剪大值、丢失标签或float16化来节省空间。
