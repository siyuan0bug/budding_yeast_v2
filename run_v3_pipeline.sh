#!/bin/bash
# ============================================================
# V3 数据集生成 + 实验流水线
#
# 用法:
#   前台运行 (会阻塞终端):
#     bash run_v3_pipeline.sh
#
#   后台运行 (推荐, 长时间任务):
#     nohup bash run_v3_pipeline.sh > /dev/null 2>&1 &
#     tail -f logs/v3_pipeline_<timestamp>.log
# ============================================================

set -u  # 未定义变量报错
# 注意: 不用 set -e, 因为 tee 会吞掉退出码, 改用手动检查 PIPESTATUS

# ---------- 路径配置 ----------
DATA_DIR="/home/users/zsy/trae_workspace/budding_yeast_v2/data"
PIPELINE_DIR="/home/users/zsy/trae_workspace/budding_yeast_v2"
PYTHON_BIN="/home/users/zsy/.conda/envs/neuralop/bin/python"
PYTHONPATH_DIR="/home/users/zsy/trae_workspace/mydata/budding_yeast"
DATASET_PATH="${PYTHONPATH_DIR}/lhs_cheat_origin_v3_210min_500steps_dual_labels.npz"

# ---------- 日志配置 ----------
LOG_DIR="${PIPELINE_DIR}/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/v3_pipeline_${TIMESTAMP}.log"

# ---------- 工具函数 ----------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

fail() {
    log "[ERROR] $*"
    log "===== 流水线中断 ====="
    exit 1
}

# ---------- 开始 ----------
log "===== V3 数据集生成 + 实验流水线 ====="
log "日志文件: $LOG_FILE"
log ""

# ============================================================
# 阶段 1: 生成 V3 数据集 (执行目录: data/)
# ============================================================
log "===== [1/2] 生成 V3 数据集 ====="
log "执行目录: $DATA_DIR"

if [ ! -f "${DATA_DIR}/lhs_v1_origin_v3.py" ]; then
    fail "未找到 lhs_v1_origin_v3.py"
fi

cd "$DATA_DIR" || fail "无法进入 $DATA_DIR"
PYTHONPATH="${PYTHONPATH_DIR}:${PYTHONPATH:-}" "$PYTHON_BIN" lhs_v1_origin_v3.py 2>&1 | tee -a "$LOG_FILE"
DATA_STATUS=${PIPESTATUS[0]}

if [ "$DATA_STATUS" -ne 0 ]; then
    fail "V3 数据集生成失败 (exit=$DATA_STATUS)"
fi

# 二次确认: 数据集文件确实生成
if [ ! -f "$DATASET_PATH" ]; then
    fail "数据集文件未生成: $DATASET_PATH"
fi

DATASET_SIZE=$(du -h "$DATASET_PATH" | cut -f1)
log "[OK] V3 数据集生成成功"
log "  路径: $DATASET_PATH"
log "  大小: $DATASET_SIZE"
log ""

# ============================================================
# 阶段 2: 运行实验管道 (执行目录: budding_yeast_v2/)
# ============================================================
log "===== [2/2] 运行 run_pipeline.sh ====="
log "执行目录: $PIPELINE_DIR"

if [ ! -f "${PIPELINE_DIR}/run_pipeline.sh" ]; then
    fail "未找到 run_pipeline.sh"
fi

cd "$PIPELINE_DIR" || fail "无法进入 $PIPELINE_DIR"
bash run_pipeline.sh 2>&1 | tee -a "$LOG_FILE"
PIPELINE_STATUS=${PIPESTATUS[0]}

if [ "$PIPELINE_STATUS" -ne 0 ]; then
    fail "run_pipeline.sh 执行失败 (exit=$PIPELINE_STATUS)"
fi

# ---------- 完成 ----------
log ""
log "===== 全部完成 ====="
log "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
log "完整日志: $LOG_FILE"
exit 0
