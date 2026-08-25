#!/bin/bash
# ==============================================================================
# 历史归档脚本: 早期遗留实验配置归档 (Archive of Legacy LHS & MASS v1 Experiments)
# ==============================================================================
# 本脚本归档了项目早期的历史实验配置（供随时查阅与复现）：
#   1. LHS V5 数据集上的 Batch Size / LR 边界探测
#   2. LHS 早期主动学习池化实验 (RGS, US, IS, WRS, VESSAL, PIAL, HGGS, PatchTST)
#   3. LHS 时代 AL4PDE v2.3 外层重训实验
#   4. LHS 早期 H1 / PINO 物理损失验证实验
#   5. LHS V2 ~ V6 数据集演进与对比实验 (V2, V3, V4, V5, V6, nofilter, guaranteed)
#   6. MASS v1 初代生化图算子与机制算子探索实验
# ==============================================================================

cd "$(dirname "$0")" || exit

export NCCL_P2P_DISABLE=1
export PYTHONPATH=$(pwd)/..:$PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON_BIN="/home/users/zsy/.conda/envs/neuralop/bin/python"

mkdir -p logs_v3 logs
NO_WANDB=1

# ================= 历史 LHS 数据集路径 =================
DATA_DIR="/home/users/zsy/mydata/budding_yeast"
DATASET_CHEAT="${DATA_DIR}/lhs_cheat_origin_210min_500steps_dual_labels.npz"
DATASET_BLIND="${DATA_DIR}/lhs_blind_210min_500steps_dual_labels.npz"
DATASET_V2="${DATA_DIR}/lhs_cheat_origin_v2_210min_500steps_dual_labels.npz"
DATASET_V3="${DATA_DIR}/lhs_cheat_origin_v3_210min_500steps_dual_labels.npz"
DATASET_V4="${DATA_DIR}/lhs_cheat_origin_v4_210min_500steps_dual_labels.npz"
DATASET_V5="${DATA_DIR}/lhs_cheat_origin_v5_210min_500steps_dual_labels.npz"
DATASET_V5_BOUNDARY="${DATA_DIR}/lhs_cheat_origin_v5_boundary_210min_500steps_dual_labels.npz"
DATASET_V5_NO_TARGET="${DATA_DIR}/lhs_cheat_origin_v5_no_target_210min_500steps_dual_labels.npz"
DATASET_V6="${DATA_DIR}/lhs_cheat_origin_v6_210min_500steps_dual_labels.npz"
DATASET_nofilter="${DATA_DIR}/lhs_cheat_origin_no_filter_210min_500steps_dual_labels.npz"
DATASET_guaranteed="${DATA_DIR}/lhs_cheat_origin_guaranteed_210min_500steps_dual_labels.npz"
ATLAS_CSV="./eval_result/old_hyper_fno_none_mse_only_210m_500s_modes32_seed42/mutant_atlas_merged.csv"

SEEDS=(42)

EXPERIMENTS=(
    # 格式 (17列，空格分隔):
    # MODEL LOSS_TYPE USE_ADJ DATASET_PATH DATASET_TAG MODES WIDTH BATCH_SIZE EPOCHS AL_STRATEGY LR WARMUP AL_NUM_ADD AL_PERTURBATION AL_MAE_THRESHOLD ROLLOUT_TYPE GPU_ID
    
    # ====== 1. LHS V5 数据集 Batch Size / LR OOM 边界探测 (GPU 0-3) ======
    #"hyper_fno mse_only false ${DATASET_V5} v5_probe_bs512 32 32 512 100 none 1.7e-3 10 0 0 0 one_time 0"
    #"hyper_fno mse_only false ${DATASET_V5} v5_probe_bs1024 32 32 1024 100 none 2.4e-3 10 0 0 0 one_time 1"
    #"hyper_fno mse_only false ${DATASET_V5} v5_probe_bs2048 32 32 2048 100 none 3.39e-3 10 0 0 0 one_time 2"
    #"hyper_fno mse_only false ${DATASET_V5} v5_probe_bs4096 32 32 4096 100 none 4.8e-3 10 0 0 0 one_time 3"

    # ====== 2. 早期 LHS 主动学习与池化采样实验 ======
    #"hyper_fno mse_only false ${DATASET_CHEAT} 32 32 64 200 rgs 6e-4 10 500 0.05 0.1 one_time 0"
    #"hyper_fno mse_only false ${DATASET_CHEAT} 32 32 64 200 us 6e-4 10 500 0 0 one_time 1"
    #"hyper_fno mse_only false ${DATASET_CHEAT} 32 32 64 200 is 6e-4 10 500 0 0. one_time 2"
    #"hyper_fno mse_only false ${DATASET_CHEAT} 32 32 64 200 wrs 6e-4 10 500 0 0. one_time 3"
    #"hyper_fno mse_only false ${DATASET_CHEAT} 32 32 64 200 vessal 6e-4 10 500 0 0. one_time 4"
    #"hyper_fno mse_only false ${DATASET_CHEAT} one_time 32 32 64 200 none 6e-4 10 0 0 0 one_time 5"
    #"cross_fno mse_only true ${DATASET_CHEAT} 32 16 96 200 rgs 9e-4 10 5000 0.1 0.1 one_time 6"
    #"cross_fno mse_only true ${DATASET_CHEAT} 32 16 32 120 random 3e-4 10 5000 0 0 one_time 7"
    #"cross_fno mse_only true ${DATASET_CHEAT} hggs 3e-4 10 5000 0 0 one_time 0"
    #"hyper_fno mse_only false ${DATASET_CHEAT} 210m_500s 32 32 64 200 none 6e-4 10 0 0 0 one_time 1"

    # ====== 3. 时序模型 (PatchTST) 与自回归探索 ======
    #"patchtst mse_only false ${DATASET_CHEAT} auto 32 128 64 200 none 6e-4 10 0 0 0 one_time 0"
    #"hyper_fno mse_only false ${DATASET_CHEAT} auto 32 32 32 200 none 6e-4 10 0 0 0 autoregressive 1"
    #"hyper_fno pinn_residual false ${DATASET_CHEAT} one 32 32 64 200 none 6e-4 10 0 0 0 one_time 2"
    #"hyper_fno mse_only false ${DATASET_CHEAT} one 32 32 64 200 pial 6e-4 10 500 0 0 one_time 3"

    # ====== 4. LHS 时代 AL4PDE 外层重训实验 (v2.3) ======
    #"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 100 al4pde_random 6e-4 10 1000 0 0 one_time 0"
    #"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 100 al4pde_uncertainty 6e-4 10 1000 0 0 one_time 1"
    #"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 100 al4pde_diversity 6e-4 10 1000 0 0 one_time 2"
    #"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 100 maal 6e-4 10 1000 0 0 one_time 3"
    #"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 100 al4pde_boundary 6e-4 10 1000 0 0 one_time 0"
    #"hyper_fno mse_only false ${DATASET_CHEAT} v1_al4pde 32 32 64 1024 al4pde_mass 2.4e-3 10 1000 0 0 one_time 1"

    # ====== 5. 早期 H1 / PINO 损失实验 ======
    #"hyper_fno h1 false ${DATASET_CHEAT} v1_h1 32 32 64 200 none 3e-4 10 0 0 0 one_time 1"
    #"hyper_fno pino false ${DATASET_CHEAT} v1_pino 32 32 64 200 none 3e-4 10 0 0 0 one_time 2"

    # ====== 6. LHS V2 ~ V6 数据集演进与对照实验 ======
    #"hyper_fno mse_only false ${DATASET_V5_NO_TARGET} v5_no_target 32 32 64 200 none 6e-4 10 0 0 0 one_time 3"
    #"hyper_fno mse_only false ${DATASET_V5_BOUNDARY} v5_boundary 32 32 64 200 none 6e-4 10 0 0 0 one_time 1"
    #"hyper_fno mse_only false ${DATASET_nofilter} nofilter 32 32 64 200 none 6e-4 10 0 0 0 one_time 3"
    #"hyper_fno mse_only false ${DATASET_guaranteed} guaranteed 32 32 64 200 none 6e-4 10 0 0 0 one_time 0"
    #"hyper_fno mse_only false ${DATASET_V6} v6 32 32 64 200 mutant_aware 6e-4 10 5000 0.05 0.1 one_time 0"
    #"hyper_fno mse_only false ${DATASET_V6} v6 32 32 64 200 none 6e-4 10 0 0 0 one_time 3"
    #"hyper_fno mse_only false ${DATASET_V5} v5 32 32 64 200 mutant_aware 6e-4 10 5000 0.05 0.1 one_time 0"
    #"hyper_fno mse_only false ${DATASET_V5} v5 32 32 64 200 none 6e-4 10 0 0 0 one_time 0"
    #"hyper_fno mse_only false ${DATASET_V4} v4 32 32 64 200 mutant_aware 6e-4 10 5000 0.05 0.1 one_time 1"
    #"hyper_fno mse_only false ${DATASET_V4} v4 32 32 64 200 none 6e-4 10 0 0 0 one_time 2"
    #"hyper_fno mse_only false ${DATASET_V3} v3 32 32 64 200 mutant_aware 6e-4 10 5000 0.05 0.1 one_time 1"
    #"hyper_fno mse_only false ${DATASET_V3} v3 32 32 64 200 none 6e-4 10 0 0 0 one_time 2"
    #"hyper_fno mse_only false ${DATASET_V2} v2 32 32 64 200 none 6e-4 10 0 0 0 one_time 3"

    # ====== 7. MASS v1 早期生化图与机制探索归档 (DATASET_MASS) ======
    #"multiscale_diffgraph hybrid_mse_rel false ${DATASET_MASS} mass_multiscale_diffgraph 32 64 512 100 none 1.7e-3 10 0 0 0 one_time 0"
    #"dualpath_spectralgraph hybrid_mse_rel false ${DATASET_MASS} mass_dualpath_spectralgraph 32 64 512 100 none 1.7e-3 10 0 0 0 one_time 1"
    #"bio_statespace_graph hybrid_mse_rel false ${DATASET_MASS} mass_bio_statespace 32 64 512 100 none 1.7e-3 10 0 0 0 one_time 2"
    #"gated_graph_resnet_film hybrid_mse_rel false ${DATASET_MASS} mass_gated_graph_hybrid 32 64 512 100 none 1.7e-3 10 0 0 0 one_time 3"
    #"bio_hetero_causal_graph hybrid_mse_rel false ${DATASET_MASS} mass_bio_hetero 32 64 512 100 none 1.7e-3 10 0 0 0 one_time 5"
    #"dyrfo hybrid_mse_rel false ${DATASET_MASS} mass_dyrfo_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 1"
    #"st_hyperno hybrid_mse_rel false ${DATASET_MASS} mass_st_hyperno_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 2"
    #"pc_bio_operator hybrid_mse_rel false ${DATASET_MASS} mass_pc_bio_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 3"
    #"biom_full hybrid_mse_rel false ${DATASET_MASS} mass_biom_full_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 7"
    #"biom_titration hybrid_mse_rel false ${DATASET_MASS} mass_biom_titration_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 6"
    #"biom_event hybrid_mse_rel false ${DATASET_MASS} mass_biom_event_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 5"
    #"biom_pathway hybrid_mse_rel false ${DATASET_MASS} mass_biom_pathway_width128 32 128 512 100 none 1.7e-3 10 0 0 0 one_time 4"
)

# 执行入口
USE_HIERARCHICAL=1
EXP_ROOT="./experiments"
mkdir -p "$EXP_ROOT" logs_v3

PIDS=()
for SEED in "${SEEDS[@]}"; do
    for EXP in "${EXPERIMENTS[@]}"; do
        read -r MODEL LOSS_TYPE USE_ADJ DATASET DATASET_TAG MODES WIDTH BATCH_SIZE EPOCHS AL_STRATEGY LR WARMUP AL_NUM_ADD AL_PERTURBATION AL_MAE_THRESHOLD ROLLOUT_TYPE GPU_ID <<< "$EXP"
        if [ -z "$GPU_ID" ]; then GPU_ID=0; fi
        if [ -z "$ROLLOUT_TYPE" ]; then ROLLOUT_TYPE="one_time"; fi
        ADJ_PARAM=$([ "$USE_ADJ" = "true" ] && echo "--adj_matrix_path mydata/yeast_signed_adjacency_matrix.npy" || echo "")

        AL_PARAMS=""
        if [ "$AL_STRATEGY" != "none" ]; then
            TRIGGER_EPOCH=10
            if [ "$AL_STRATEGY" = "pial" ]; then TRIGGER_EPOCH=50; fi
            if [ "$AL_STRATEGY" = "mutant_aware" ]; then TRIGGER_EPOCH=100; fi
            if [[ "$AL_STRATEGY" == al4pde_* || "$AL_STRATEGY" = "maal" ]]; then TRIGGER_EPOCH=30; fi
            AL_PARAMS="--al_strategy $AL_STRATEGY --al_trigger_epoch $TRIGGER_EPOCH --al_num_add $AL_NUM_ADD --al_perturbation $AL_PERTURBATION --al_mae_threshold $AL_MAE_THRESHOLD"
            if [ "$AL_STRATEGY" = "mutant_aware" ]; then AL_PARAMS="$AL_PARAMS --atlas_csv $ATLAS_CSV"; fi
            if [[ "$AL_STRATEGY" == al4pde_* || "$AL_STRATEGY" = "maal" ]]; then AL_PARAMS="$AL_PARAMS --al_pool_dataset $DATASET_V5 --al_initial_train_size 2000"; fi
        else
            AL_PARAMS="--al_strategy none"
        fi

        LR_STR=$(( $(echo "$LR >= 0.001" | bc -l 2>/dev/null || echo 0) ) && printf "%.4g" "$LR" || printf "%.1e" "$LR" | sed 's/-0/-/g')
        DATASET_NAME=$(basename "$DATASET" .npz)
        RUN_NAME="${MODEL}_${AL_STRATEGY}_${LOSS_TYPE}_${DATASET_TAG}_modes${MODES}_seed${SEED}"

        HPARAM_SIG="w${WIDTH}_m${MODES}_b${BATCH_SIZE}_ep${EPOCHS}_lr${LR_STR}"
        if [ "$USE_ADJ" = "true" ]; then HPARAM_SIG="${HPARAM_SIG}_adj"; fi
        if [ "$ROLLOUT_TYPE" = "autoregressive" ]; then HPARAM_SIG="${HPARAM_SIG}_ar"; fi
        if [ -n "$WARMUP" ] && [ "$WARMUP" -ne 10 ] && [ "$WARMUP" -gt 0 ]; then HPARAM_SIG="${HPARAM_SIG}_wm${WARMUP}"; fi

        EXP_DIR="${EXP_ROOT}/${DATASET_NAME}/${MODEL}/${LOSS_TYPE}_${AL_STRATEGY}/${HPARAM_SIG}/seed_${SEED}"
        CKPT_DIR="${EXP_DIR}/checkpoints"
        EVAL_OUT_DIR="${EXP_DIR}/eval"
        LOG_FILE="${EXP_DIR}/train.log"
        mkdir -p "$CKPT_DIR" "$EVAL_OUT_DIR"
        ln -sf "$(pwd)/${LOG_FILE}" "logs_v3/${RUN_NAME}.log" 2>/dev/null

        echo "🚀 启动归档实验 -> Seed: $SEED | 模型: $MODEL | GPU: $GPU_ID | 目录: $EXP_DIR"
        {
            CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_BIN train.py \
                --model $MODEL --loss_type $LOSS_TYPE --dataset_name "$DATASET" \
                --modes $MODES --width $WIDTH --batch_size $BATCH_SIZE --max_epochs $EPOCHS \
                --seed $SEED --name $RUN_NAME --lr $LR --warmup_epochs $WARMUP --devices 1 \
                $AL_PARAMS $ADJ_PARAM --rollout_type $ROLLOUT_TYPE --exp_dir $EXP_DIR --dataset_tag $DATASET_NAME --no_wandb > "$LOG_FILE" 2>&1
        } &
        PIDS+=($!)
    done
done

for pid in "${PIDS[@]}"; do wait $pid; done
echo "🎉 归档实验执行完毕!"
