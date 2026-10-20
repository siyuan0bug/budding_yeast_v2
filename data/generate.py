"""Unified dataset CLI; all original generator scripts remain unchanged.

Example: python -m budding_yeast_v2.data.generate --recipe budding_kdpp
         --samples 100 --seed 42 --workers 4 --output /tmp/kdpp_new.npz
"""
import argparse
from dataclasses import asdict,replace
import json
import os
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from budding_yeast_v2.data.generation.contracts import GenerationConfig
from budding_yeast_v2.data.generation.runner import generate_dataset
from budding_yeast_v2.data.recipes import RECIPES,resolve_config,output_filename


def main(argv=None):
    parser=argparse.ArgumentParser(description='模块化 Budding / Fission 数据集生成；保留全部旧代码')
    parser.add_argument('--list',action='store_true',help='列出可用数据集配置')
    parser.add_argument('--recipe',choices=tuple(RECIPES))
    parser.add_argument('--samples',type=int,help='合成样本请求数；默认继承对应旧 CLI')
    parser.add_argument('--seed',type=int,help='默认继承对应旧 CLI')
    parser.add_argument('--workers',type=int,default=min(60,os.cpu_count() or 1))
    parser.add_argument('--output',help='新的 NPZ 输出路径')
    parser.add_argument('--paired-output',default='',help='Twins 的另一套输出路径（mass 方案）')
    parser.add_argument('--source',default='',help='KO mirror / BAMS ablation 的源 NPZ')
    parser.add_argument('--data_dir',help='输出目录，可与自动文件名配合使用')
    parser.add_argument('--mutation-profile',choices=('mass','mutant_aware'))
    parser.add_argument('--distribution-profile',choices=('mass','mutant_aware'))
    parser.add_argument('--level-ratios',type=float,nargs='+')
    parser.add_argument('--mode-freq',choices=('class','flat','role','s5'))
    parser.add_argument('--temperature',type=float,default=.8)
    parser.add_argument('--curv-thresh',type=float,default=.01)
    parser.add_argument('--flat-prob',type=float,default=.20)
    parser.add_argument('--ko-interval-max',type=float,default=1e-4)
    parser.add_argument('--pilot-samples',type=int)
    parser.add_argument('--no-bisection','--no_bisection',action='store_true',help='关闭 BAMS 二分细化')
    parser.add_argument('--nan-cap-frac',type=float,default=.05)
    parser.add_argument('--backlog-thresh',type=float,default=.20)
    parser.add_argument('--max-attempts-mult',type=int,default=30)
    parser.add_argument('--wt-fold-dist',choices=('flat','tier2'),default='tier2')
    parser.add_argument('--max-attempts',type=int,help='每个碰撞候选的最大救援次数（默认 30）；WT-Blind 也用作候选总尝试上限')
    parser.add_argument('--no-real',action='store_true')
    parser.add_argument('--parameter-only',action='store_true')
    parser.add_argument('--overwrite',action='store_true',help='显式允许覆盖输出，默认拒绝')
    parser.add_argument('--show-config',action='store_true',help='只显示配置，不仿真或写文件')
    args=parser.parse_args(argv)
    if args.list:
        for name,recipe in RECIPES.items():
            filename=output_filename(GenerationConfig(name))
            print(f'{name:30s} samples={recipe.default_samples:<7d} seed={recipe.default_seed:<9d} file={filename}')
        return
    if args.recipe is None:
        parser.error('--recipe is required unless --list is used')
    output=args.output
    if output is None and not args.data_dir and not args.show_config:
        parser.error('Specify --output or --data_dir')
    config=GenerationConfig(recipe=args.recipe,n_samples=args.samples,seed=args.seed,workers=args.workers,
        output=output or '',mutation_profile=args.mutation_profile,distribution_profile=args.distribution_profile,
        level_ratios=tuple(args.level_ratios) if args.level_ratios else None,include_real=not args.no_real,
        parameter_only=args.parameter_only,mode_freq=args.mode_freq or RECIPES[args.recipe].mode_freq,temperature=args.temperature,
        curv_thresh=args.curv_thresh,flat_prob=args.flat_prob,ko_interval_max=args.ko_interval_max,
        pilot_samples=args.pilot_samples,max_attempts=args.max_attempts,overwrite=args.overwrite,
        paired_output=args.paired_output,source=args.source,wt_fold_dist=args.wt_fold_dist,
        enable_bisection=not args.no_bisection,nan_cap_frac=args.nan_cap_frac,
        backlog_thresh=args.backlog_thresh,max_attempts_mult=args.max_attempts_mult)
    config=resolve_config(config)
    if output is None and args.data_dir:
        config=replace(config,output=str(Path(args.data_dir)/output_filename(config)))
    if args.show_config:
        print(json.dumps({'recipe':asdict(RECIPES[args.recipe]),'run':asdict(config)},ensure_ascii=False,indent=2))
        return config
    return generate_dataset(config)


if __name__=='__main__':
    main()
