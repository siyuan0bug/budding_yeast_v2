#!/bin/bash
# 自动 cd 到脚本所在目录，无论终端 cwd 在哪都能正确运行
cd "$(dirname "$0")/.." || exit

export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=$(pwd)/..:$PYTHONPATH
export WANDB_API_KEY="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"

# ==========================================
# 全局开关：设为 1 启用 W&B 离线模式（日志保存在本地）
# ==========================================
NO_WANDB=0

WANDB_FLAG=""
if [ "$NO_WANDB" = "1" ]; then
    WANDB_FLAG="--no_wandb"
fi

echo "==========================================================="
echo "🔥 启动 AR+AL 训练流水线 (AL4PDE 架构)"
echo "==========================================================="

/home/users/zsy/.conda/envs/neuralop/bin/python scripts/run_ar_al.py \
    --config configs/ar_al_default.yaml \
    --model hyper_fno \
    --num_al_iter 5 \
    --num_epochs 100 \
    --num_train_steps 2 \
    --noise 0.0 \
    --noise_mode channel_std_rel \
    --batch_size 64 \
    --acquisition lcmd \
    --seed 42 \
    $WANDB_FLAG

echo "✅ 训练结束"

# ==========================================
# 自动评估 (Eval) 说明：
# 上方的 run_ar_al.py 脚本运行到最后一步时，会自动触发 eval_ar_al.py 进行测试集预测和画图！
# 如果您想单独/手动运行评估，可以参考以下命令：
# python scripts/eval_ar_al.py \
#     --ckpt train_result_ar/hyper_fno_ar_2steps_al_random_seed42/checkpoints/model_al_4.pt \
#     --data_path /home/users/zsy/mydata/budding_yeast/lhs_blind_210min_500steps_dual_labels.npz \
#     --save_dir train_result_ar/hyper_fno_ar_2steps_al_random_seed42/eval_result
# ==========================================
