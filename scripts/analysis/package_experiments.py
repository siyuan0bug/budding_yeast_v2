#!/usr/bin/env python3
"""
Package experiments directory, excluding checkpoints, images, and wandb files.
Generates a compressed zip or tar.gz archive for easy local download.
Can be executed from ANY directory.
"""

import os
import sys
import argparse
import zipfile
import tarfile
import time
from pathlib import Path

# 默认排除的文件夹名（小写匹配）
EXCLUDE_DIRS = {
    "wandb",
    "checkpoints",
    "__pycache__",
    ".git",
}

# 默认排除的文件扩展名（小写匹配，带点）
EXCLUDE_EXTENSIONS = {
    # 检查点/模型权重
    ".ckpt", ".pt", ".pth", ".bin", ".safetensors", ".onnx",
    # 图像文件
    ".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp", ".bmp", ".tiff",
    # wandb 与缓存
    ".wandb", ".pyc", ".pyo", ".DS_Store",
}


def is_excluded_dir(dir_name: str) -> bool:
    name_lower = dir_name.lower()
    return name_lower in EXCLUDE_DIRS or "checkpoint" in name_lower or "wandb" in name_lower


def is_excluded_file(file_name: str) -> bool:
    ext = os.path.splitext(file_name)[1].lower()
    if ext in EXCLUDE_EXTENSIONS:
        return True
    if file_name.startswith("."):
        return True
    return False


def collect_files(source_dir: Path):
    matched_files = []
    skipped_count = 0
    total_size_bytes = 0

    for root, dirs, files in os.walk(source_dir):
        # 剪枝：避免遍历被排除的子目录
        dirs[:] = [d for d in dirs if not is_excluded_dir(d)]

        for file in files:
            if is_excluded_file(file):
                skipped_count += 1
                continue

            full_path = Path(root) / file
            if full_path.is_symlink() and not full_path.exists():
                continue

            rel_path = full_path.relative_to(source_dir)
            file_size = full_path.stat().st_size
            matched_files.append((full_path, rel_path, file_size))
            total_size_bytes += file_size

    return matched_files, skipped_count, total_size_bytes


def format_size(bytes_num: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_num < 1024.0:
            return f"{bytes_num:.2f} {unit}"
        bytes_num /= 1024.0
    return f"{bytes_num:.2f} TB"


def package_to_zip(matched_files, output_path: Path):
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for full_path, rel_path, _ in matched_files:
            zf.write(full_path, arcname=str(rel_path))


def package_to_tar(matched_files, output_path: Path):
    with tarfile.open(output_path, "w:gz") as tf:
        for full_path, rel_path, _ in matched_files:
            tf.add(full_path, arcname=str(rel_path))


def main():
    # 自动定位项目根目录（基于脚本真实物理路径，不受当前工作目录影响）
    script_dir = Path(__file__).resolve().parent
    default_src = script_dir / "experiments"
    if not default_src.exists():
        default_src = Path("/home/users/zsy/projects/budding_yeast_cell_cycle/budding_yeast_v2/experiments")
    
    default_dst = default_src.parent / "experiments_clean.zip"

    parser = argparse.ArgumentParser(description="Package experiment logs & configs, skipping models, images and wandb.")
    parser.add_argument("--source", "-s", type=str, default=str(default_src), help="Source experiments directory")
    parser.add_argument("--output", "-o", type=str, default=str(default_dst), help="Output archive path (.zip or .tar.gz)")
    args = parser.parse_args()

    src_path = Path(args.source).resolve()
    out_path = Path(args.output).resolve()

    if not src_path.exists() or not src_path.is_dir():
        print(f"[Error] Source directory does not exist: {src_path}")
        sys.exit(1)

    print(f"==================================================")
    print(f"📦 开始扫描实验目录: {src_path}")
    print(f"🚫 排除目录: wandb, checkpoints, __pycache__ 等")
    print(f"🚫 排除文件: *.png, *.jpg, *.ckpt, *.pt, *.pth, *.wandb 等")
    print(f"==================================================")

    start_time = time.time()
    matched_files, skipped_count, total_size_bytes = collect_files(src_path)
    scan_time = time.time() - start_time

    print(f"🔍 扫描完成 (耗时 {scan_time:.2f}s):")
    print(f"   - 符合条件的文件数: {len(matched_files)}")
    print(f"   - 过滤/跳过的文件数: {skipped_count}")
    print(f"   - 待打包文件原始总大小: {format_size(total_size_bytes)}")
    print(f"--------------------------------------------------")

    if not matched_files:
        print("[Warning] 未找到需要打包的文件！")
        return

    print(f"⏳ 正在压缩打包到: {out_path} ...")
    pack_start = time.time()

    if str(out_path).endswith(".tar.gz") or str(out_path).endswith(".tgz"):
        package_to_tar(matched_files, out_path)
    else:
        package_to_zip(matched_files, out_path)

    pack_time = time.time() - pack_start
    archive_size = out_path.stat().st_size

    print(f"✅ 打包完成！(压缩耗时 {pack_time:.2f}s)")
    print(f"   - 压缩包文件: {out_path}")
    print(f"   - 压缩后大小: {format_size(archive_size)}")
    print(f"==================================================")
    print(f"💡 下载到本地电脑的方法：")
    print(f"1. 【VS Code 界面下载】在左侧文件浏览器中找到并右键 '{out_path.name}' -> 点击 'Download...' (下载)")
    print(f"2. 【SCP 命令（在本地终端执行）】:")
    print(f"   scp <user>@<host>:{out_path} ./")
    print(f"3. 【Rsync 命令（在本地终端执行）】:")
    print(f"   rsync -avzP <user>@<host>:{out_path} ./")
    print(f"==================================================")


if __name__ == "__main__":
    main()
