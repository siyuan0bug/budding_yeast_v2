import os
import sys
import torch
from pathlib import Path

# 将项目根目录添加到系统路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from torchview import draw_graph
from models.components.resnet_film import ResNetFiLM

def visualize_resnet_film():
    save_dir = Path(__file__).resolve().parent
    print(f"[*] 正在生成 ResNetFiLM 模型结构图，保存路径: {save_dir}")

    # 1. 实例化模型
    model = ResNetFiLM(num_vars=39, param_dim=141, width=64, n_blocks=4, d_lambda=128, d_attn=16)
    model.eval()

    # 2. 构造模拟输入 (Batch Size = 2, 39 个变量, 2 个特征通道, 100 个时间点) 和 141 维动力学参数
    ic_time_grid = torch.randn(2, 39, 2, 100)
    params = torch.randn(2, 141)

    # 3. 生成宏观级别结构图 (depth=1: 最清晰的顶层流程图)
    print("  -> 生成 depth=1 宏观架构图...")
    graph_d1 = draw_graph(
        model,
        input_data=[ic_time_grid, params],
        depth=1,
        graph_name="ResNetFiLM_Macro",
        expand_nested=True,
        save_graph=True,
        directory=str(save_dir),
        filename="resnet_film_macro"
    )

    # 4. 生成模块级别结构图 (depth=2: 展开到 ResNetFiLMBlock 内部子模块，如 FiLMLayer, Attention, Conv)
    print("  -> 生成 depth=2 模块详细架构图...")
    graph_d2 = draw_graph(
        model,
        input_data=[ic_time_grid, params],
        depth=2,
        graph_name="ResNetFiLM_Detailed",
        expand_nested=True,
        save_graph=True,
        directory=str(save_dir),
        filename="resnet_film_detailed"
    )

    print(f"[✓] 生成完毕！结构图已保存在: {save_dir}")

if __name__ == "__main__":
    visualize_resnet_film()
