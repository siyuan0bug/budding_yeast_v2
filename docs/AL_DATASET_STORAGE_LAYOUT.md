# 主动学习数据的无损存储

2026-10-30 起，新 AL 实验默认采用共享初始数据、只追加新样本、每轮保存数据清单的方式。采样、ODE、训练预算和每轮重新拟合归一化的规则不变。

## 每条轨迹只存一次

```text
experiments/al_budding/
├── shared/datasets/<初始文件SHA256>.npz
└── fno/<loss_strategy>/<配置>/seed_42/
    ├── data/pool_manifest.json
    ├── acquisitions/
    │   ├── after_round_00/added_samples.npz  # 新增1万条
    │   ├── after_round_01/added_samples.npz  # 新增2万条
    │   └── after_round_02/added_samples.npz  # 新增3万条
    └── rounds/
        ├── round_00/dataset.json            # 引用初始数据
        ├── round_01/dataset.json            # 引用初始数据＋第一批新增
        ├── round_02/dataset.json            # 引用初始数据＋前两批新增
        └── round_03/dataset.json            # 引用初始数据＋全部新增
```

初始文件为不可写的独立内容寻址副本，不和可被重新生成的源文件硬链接；同一初始输入的模型、策略共用这一份。`AL_SHARED_DATA_DIR` 可显式覆盖共享位置，默认位于对应 `al_budding/al_fission` 家族目录下。

清单记录源文件的相对路径和 SHA256、每个字段的形状/类型/校验值，以及精确行顺序。必要的小型差异元数据保存在同目录的 `dataset.metadata.<hash>.npz`，例如累计样本编号和各切分的行索引。参数、实际初值、突变规则、L、全部标签、候选来源均保留。固定验证/测试/real 行通过引用复用。

读取后仍是该轮完整数据：10k/20k/40k/70k 合成训练行，外加1250验证、1250合成测试、126 real。读取清单会检查源文件及字段的校验值；缺块、改值或顺序错误会报错。

## 训练、评测与分析接口

`train.py` 和 AL engine 写入 `dataset.json`。`YeastDataModule` 自动读取 NPZ 或清单，每轮仍在累计合成训练集重拟合归一化。`eval.py` 自动找到 checkpoint 同目录的清单或旧 NPZ；checkpoint 的数据哈希、切分及统计检查保持有效。

Python 中读取两种格式统一使用：

```python
from budding_yeast_v2.data.dataset_archive import open_dataset

with open_dataset("rounds/round_03/dataset.json") as dataset:
    params = dataset["params"]
    initials = dataset["y_init"]
    trajectories = dataset["data"]
```

旧配置中的 `.../dataset.npz` 路径在原 NPZ 已压缩迁移、旁边存在 `dataset.json` 时也可通过此接口读取。直接 `np.load` 仅适用于真实 NPZ，不能用于 JSON；已有轨迹极值、badcase、响应配对等分析入口已接入统一读取。分块扫描可用 `dataset_field_chunks`，指定行可用 `read_dataset_rows`。

移动或归档实验时，需要同时保留它引用的共享初始文件、新增块和差异元数据；单独复制清单并不包含轨迹。旧的独立静态数据集仍可原样读取。

## 已有实验的转换

`scripts/analysis/compact_al_datasets.py` 默认只列计划；`--apply` 对每个旧快照逐字段检查形状、dtype及字节完全一致，通过后才移除冗余 NPZ。原始 initial/pool 输入、增量轨迹、checkpoint、评测图片、日志和 W&B 离线记录保留。转换支持中断后再次执行；已完成文件不重复转换。

```bash
python scripts/analysis/compact_al_datasets.py \
  --manifest /path/to/launch/manifest.json \
  --apply --workers 6 --output /path/to/conversion.json
```

本次范围为 `al_full_launch_20261030_022728/manifest.json` 列出的14个长程实验，包括已有数据的失败实验。更早的历史 AL 和静态实验未批量迁移。转换和 checkpoint 重载报告保存在 `/home/users/zsy/mydata/budding_yeast/analysis/al_storage_compaction_20261030/`。

## float16 试算与精度选择

不是只能用 float32。但 float16 是有损保存，读取时转回 float32 不能恢复丢失位数。对本次 Random 最后一轮全部72,626条轨迹逐值试算：

|范围|float16舍入造成的整体RelL2|最大绝对误差|非零变成零的数值数|溢出数|
|---|---:|---:|---:|---:|
|全量|0.0002080，约0.0208%|1.998535|15,748,562|0|
|126株real|0.0002114，约0.0211%|0.249878|56,498|0|

原数据范围约为 −1.57e−6 至5471.00。这个试算仅衡量存储舍入，不代表重训后的 real 误差，也没有证明标签重算完全一致。因此本次仍保存 float32 轨迹、float64 实际初值及物理参数；通过去掉重复快照节省空间，训练端 float32/bf16 不变。

验证包括141项相关回归（含新增读取/迁移测试、原 AL 流程、真实 spawn ODE 入库、Fission 标签与冷启动重载、查询策略及大值/badcase 分析），以及真实 Random 的 round_00/round_03 FNO checkpoint 重载、数据哈希/切分/归一化严格匹配与 CPU 推理。

## 本次迁移实测结果

14个长程实验已全部转换，54份累计轨迹快照和14份初始副本逐字段形状/dtype/字节完全一致后移除。68份清单及其95个唯一源块的SHA256复核全部通过。

|项目|转换前|转换后|
|---|---:|---:|
|14项实验占用（转换后计入一份共享初始）|173.13 GiB|56.65 GiB|
|机械盘普通用户可用空间（本次快照）|40.33 GiB|156.81 GiB|

净释放116.48 GiB，约67.28%。14项运行目录本身合计55.96 GiB，共享初始0.69 GiB。checkpoint、ensemble、图片、离线W&B及反馈数组的分配空间与转换前逐类一致。原始输入文件和更早的静态/AL实验未批量改写。

逐实验前后大小见分析目录的`per_run_savings.csv`，转换逐文件记录为`random_conversion.json`和`remaining_conversion.json`，最终统计为`summary.json`，checkpoint实测为`replay_round00.json`及`replay_round03.json`。
