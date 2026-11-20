# 酵母ODE代理模型与采样实验

本项目预测参数、实际初值及时间对应的多变量ODE轨迹，并比较数据选择与合法新输入生成方法。
开发与文件归属规则见[AGENTS.md](AGENTS.md)，使用说明从[文档导航](docs/README.md)进入。

## 环境

```bash
export PYTHONPATH=/home/users/zsy/projects/budding_yeast_cell_cycle:$PYTHONPATH
/home/users/zsy/.conda/envs/neuralop/bin/python train.py --help
```

使用已有neuralop环境。正式训练前核对任务配置、数据、GPU和资源限制；W&B仅离线保存。

## 常用入口

- `train.py`、`eval.py`、`tune.py`：单项训练、评测、调参。
- `run_pipeline.sh`、`run_tune.sh`：模型实验与调参批处理。
- `run_sampling_budding.sh`：Budding采样实验；无参为默认固定池batch。
- `run_sampling_fission.sh`：Fission原在线流程，不能把更名理解成已支持Budding完整池新协议。

选定baseline与ours批量任务预览：

```bash
bash run_sampling_budding.sh extensions --suite all \
  --directory /home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/batches/preview \
  --dry-run
```

## 代码与结果

- `models/`、`data/`、`utils/`为运行库；`scripts/`为命令行工具；`tests/`为正式测试；`scratch/`为临时探针。
- 归档代码保留在`utils/archive/`但不作为运行依赖；在线在役功能在`utils/sampling/online/`。
- 完整池在`/home/users/zsy/mydata/budding_yeast/offline_pools/`。
- 采样结果在`/home/users/zsy/mydata/budding_yeast/experiments/budding_sampling/<model>/<pool>/<method>/<profile>/seed_<主seed>/`。
- 共享协议、批次与汇总分别在该根的`shared/`、`batches/`、`summary/`；`experiments`为外部结果根的软链接。
- 不在仓库根、scripts或docs保存生成的轨迹、权重、图片、CSV与运行报告。

历史方案和整理前说明在docs/history，不能当成当前默认设置。迁移、归档及清理有对应外部审计记录。
