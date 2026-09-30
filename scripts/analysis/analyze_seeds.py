import os
import json
import numpy as np

def analyze_seeds(eval_root="/home/users/zsy/budding_yeast_v2/eval_result"):
    # 按照前缀对目录进行分组 (例如: cross_fno_mse_only_210m_500s_modes32)
    groups = {}
    
    for folder in os.listdir(eval_root):
        if not os.path.isdir(os.path.join(eval_root, folder)): continue
        
        # 假设目录名格式为: xxx_seed42
        if "_seed" in folder:
            prefix = folder.split("_seed")[0]
            if prefix not in groups: groups[prefix] = []
            groups[prefix].append(os.path.join(eval_root, folder, "global_metrics_denorm.json"))

    print("📊 开始计算多 Seed 统计学指标...")
    for prefix, json_paths in groups.items():
        if len(json_paths) < 2: continue
        
        all_metrics = []
        for p in json_paths:
            with open(p, 'r') as f:
                all_metrics.append(json.load(f)['all_39_vars'])
        
        # 转为 numpy 进行统计
        keys = all_metrics[0].keys()
        stats = {k: [m[k] for m in all_metrics] for k in keys}
        
        print(f"\n🚀 模型配置: {prefix} (共 {len(json_paths)} 个 Seed)")
        print("-" * 50)
        for k, v in stats.items():
            mean_val = np.mean(v)
            std_val = np.std(v)
            print(f"{k.ljust(15)} : {mean_val:8.4f} ± {std_val:8.4f}")
        print("-" * 50)

if __name__ == "__main__":
    analyze_seeds()