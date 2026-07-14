#!/bin/bash
# ==========================================
# 批量运行 v2-v6 LHS 数据生成脚本
# 用法:
#   ./run_lhs_batch.sh           # 运行 v2,v3,v4,v5,v6
#   ./run_lhs_batch.sh 2 4       # 仅运行 v2,v4
#   SIM_TF=300 ./run_lhs_batch.sh  # 自定义模拟时长
# ==========================================

# === 可配置参数 (通过环境变量覆盖) ===
SIM_TF=${SIM_TF:-210}
SIM_STEPS=${SIM_STEPS:-500}
SCRIPT_DIR="/home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/data"
LOG_DIR="/home/users/zsy/mydata/budding_yeast/logs"

# === 限制 BLAS 线程数为 1, 防止 fork + OpenBLAS 多线程死锁 ===
# ProcessPoolExecutor 用 fork 创建子进程时, OpenBLAS 线程池的锁状态不一致会导致死锁
# 设为 1 后每个工作进程单线程做线性代数, 16 进程 × 1 线程 = 16 核, 匹配 safe_cores
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# === 版本列表 (命令行参数或默认 v2-v6) ===
if [ $# -gt 0 ]; then
    VERSIONS=("$@")
else
    VERSIONS=(5 6 4 3 2)
fi

# === 创建日志目录 ===
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/batch_lhs_${TIMESTAMP}.log"

# === 辅助函数: 同时输出到终端和日志文件 ===
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# === 开始 ===
log "========================================"
log "批量运行 LHS 数据生成"
log "版本: ${VERSIONS[*]}"
log "SIM_TF=${SIM_TF}, SIM_STEPS=${SIM_STEPS}"
log "日志文件: ${LOG_FILE}"
log "========================================"

TOTAL_START=$(date +%s)
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_VERSIONS=()

for v in "${VERSIONS[@]}"; do
    SCRIPT="${SCRIPT_DIR}/lhs_v1_origin_v${v}.py"

    if [ ! -f "$SCRIPT" ]; then
        log "✗ v${v}: 脚本不存在 -> ${SCRIPT}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_VERSIONS+=("$v")
        continue
    fi

    log ""
    log "----------------------------------------"
    log "▶ v${v} 开始: ${SCRIPT}"
    log "----------------------------------------"

    START=$(date +%s)

    # python -u 确保实时输出 (不缓冲), 2>&1 合并 stderr, tee 同时写终端和日志
    SIM_TF=$SIM_TF SIM_STEPS=$SIM_STEPS python -u "$SCRIPT" 2>&1 | tee -a "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}

    END=$(date +%s)
    DURATION=$((END - START))
    MINS=$((DURATION / 60))
    SECS=$((DURATION % 60))

    if [ $EXIT_CODE -eq 0 ]; then
        log "✓ v${v} 成功 (耗时 ${MINS}m${SECS}s)"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        log "✗ v${v} 失败 (耗时 ${MINS}m${SECS}s, 退出码 ${EXIT_CODE})"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_VERSIONS+=("$v")
    fi
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
TOTAL_MINS=$((TOTAL_DURATION / 60))
TOTAL_SECS=$((TOTAL_DURATION % 60))

log ""
log "========================================"
log "批量运行结束"
log "成功: ${SUCCESS_COUNT}/${#VERSIONS[@]}  失败: ${FAIL_COUNT}/${#VERSIONS[@]}"
if [ ${FAIL_COUNT} -gt 0 ]; then
    log "失败版本: ${FAILED_VERSIONS[*]}"
fi
log "总耗时: ${TOTAL_MINS}m${TOTAL_SECS}s"
log "日志文件: ${LOG_FILE}"
log "========================================"
