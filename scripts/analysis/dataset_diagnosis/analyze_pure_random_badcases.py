#!/usr/bin/env python3
"""Reproducible multi-metric diagnosis for the pure-random benchmark.

This script deliberately does *not* load ``data`` from the compressed NPZ: that
array is several GB once decompressed.  It uses the authoritative per-mutant
evaluation CSVs, plus the lightweight names/labels arrays and (optionally) the
already-generated genotype-coverage table.

It answers three questions which a single top-k ranking cannot answer:
  1. Which real mutants are bad on several metrics in the selected reference run?
  2. Which ones stay bad across different model architectures?
  3. Is a bad case associated with absent/sparse genotype coverage?

The output is intentionally a CSV first, so it can be joined with future runs
without changing the scientific decision rule.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENTS = ROOT / "experiments/mutant_aware_pure_random_100000_210m_500s"
DEFAULT_DATASET = Path("/home/users/zsy/mydata/budding_yeast/mutant_aware_pure_random_100000_210m_500s.npz")
DEFAULT_REFERENCE = (
    DEFAULT_EXPERIMENTS
    / "biom_refiner_plus/triple_dynamic_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/per_mutant_metrics.csv"
)
_legacy_cov = ROOT / "data/dataset_analysis/real_mutants_synthetic_coverage.csv"
_archived_cov = ROOT / "archive/dataset_analysis_reports/real_mutants_synthetic_coverage.csv"
DEFAULT_COVERAGE = _legacy_cov if _legacy_cov.exists() else _archived_cov


def mutant_id(name: str) -> str:
    """Return the stable numeric prefix, preserving decimal IDs such as 2.1."""
    match = re.match(r"^(\d+(?:\.\d+)?)_", str(name))
    return match.group(1) if match else str(name)


def choose_one_run(paths: list[Path]) -> Path:
    """Choose one evaluation per model without silently averaging ablations.

    A model can have normalization/loss ablations.  Prefer the normal directory
    and, within it, the shortest path.  The reference run is reported separately.
    """
    normal = [p for p in paths if "asinh_" not in str(p) and "旧版本" not in str(p)]
    return sorted(normal or paths, key=lambda p: (len(str(p)), str(p)))[0]


def collect_model_metrics(experiments: Path) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for model_dir in sorted(p for p in experiments.iterdir() if p.is_dir()):
        candidates = list(model_dir.glob("**/eval/per_mutant_metrics.csv"))
        if not candidates:
            continue
        path = choose_one_run(candidates)
        frame = pd.read_csv(path)
        needed = {"Mutant", "MAE", "MSE", "Relative_L2", "Correlation"}
        if not needed.issubset(frame.columns):
            continue
        frame = frame[[c for c in ["Mutant", "Pattern", "MAE", "MSE", "Relative_L2", "Correlation"] if c in frame]]
        frame["model"] = model_dir.name
        frame["metrics_path"] = str(path)
        records.append(frame)
    if not records:
        raise FileNotFoundError(f"No usable per_mutant_metrics.csv below {experiments}")
    return pd.concat(records, ignore_index=True)


def load_coverage(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["id"])
    coverage = pd.read_csv(path)
    name_col = "Mutant_Name" if "Mutant_Name" in coverage else "Mutant"
    coverage["id"] = coverage[name_col].map(mutant_id)
    keep = [
        "id", "Synth_Exact_Count", "Synth_Gene_Subset_Count", "Num_Mutations",
        "Mutated_Genes", "Genotype_Modes", "Carbon",
    ]
    return coverage[[c for c in keep if c in coverage]].drop_duplicates("id")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE,
                        help="The one run used for the primary bad-case decision.")
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "scripts/analysis/outputs/pure_random_badcase_diagnosis")
    parser.add_argument("--mae", type=float, default=0.15)
    parser.add_argument("--rel-l2", type=float, default=0.25)
    parser.add_argument("--corr", type=float, default=0.75)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.reference.exists():
        raise FileNotFoundError(args.reference)

    # Lightweight dataset audit: never touch z["data"].
    with np.load(args.dataset, allow_pickle=True) as dataset:
        names = dataset["mutant_names"].astype(str)
        patterns = dataset["pattern_labels"].astype(str)
        labels = dataset["labels"].astype(str)
    real_mask = np.fromiter((bool(re.match(r"^\d+(?:\.\d+)?_", n)) for n in names), dtype=bool)
    dataset_summary = {
        "dataset": str(args.dataset),
        "n_total": int(len(names)),
        "n_real": int(real_mask.sum()),
        "n_synthetic": int((~real_mask).sum()),
        "real_pattern_counts": pd.Series(patterns[real_mask]).value_counts().to_dict(),
        "synthetic_pattern_counts": pd.Series(patterns[~real_mask]).value_counts().to_dict(),
        "real_label_counts": pd.Series(labels[real_mask]).value_counts().to_dict(),
    }

    all_models = collect_model_metrics(args.experiments)
    all_models["id"] = all_models["Mutant"].map(mutant_id)
    all_models["bad_mae"] = all_models["MAE"] > args.mae
    all_models["bad_rel_l2"] = all_models["Relative_L2"] > args.rel_l2
    all_models["bad_corr"] = all_models["Correlation"] < args.corr
    all_models["bad_any"] = all_models[["bad_mae", "bad_rel_l2", "bad_corr"]].any(axis=1)

    cross = all_models.groupby("id", as_index=False).agg(
        models=("model", "nunique"),
        median_mae=("MAE", "median"), median_mse=("MSE", "median"),
        median_rel_l2=("Relative_L2", "median"), median_corr=("Correlation", "median"),
        fraction_models_bad=("bad_any", "mean"),
        fraction_models_low_corr=("bad_corr", "mean"),
    )

    reference = pd.read_csv(args.reference)
    required = {"Mutant", "MAE", "MSE", "Relative_L2", "Correlation"}
    if not required.issubset(reference.columns):
        raise ValueError(f"Reference metrics missing required columns: {args.reference}")
    reference["id"] = reference["Mutant"].map(mutant_id)
    reference["bad_mae"] = reference["MAE"] > args.mae
    reference["bad_rel_l2"] = reference["Relative_L2"] > args.rel_l2
    reference["bad_corr"] = reference["Correlation"] < args.corr
    reference["bad_metric_count"] = reference[["bad_mae", "bad_rel_l2", "bad_corr"]].sum(axis=1)
    reference["bad_any"] = reference["bad_metric_count"] > 0

    coverage = load_coverage(args.coverage)
    report = reference.merge(cross, on="id", how="left", suffixes=("", "_cross")).merge(coverage, on="id", how="left")
    report["persistent_cross_model_badcase"] = report["fraction_models_bad"] >= 0.50
    report["coverage_risk"] = np.select(
        [report.get("Synth_Exact_Count", pd.Series(np.nan, index=report.index)).eq(0),
         report.get("Synth_Exact_Count", pd.Series(np.nan, index=report.index)).lt(10)],
        ["zero exact genotype", "<10 exact genotypes"], default="not flagged",
    )
    report = report.sort_values(
        ["persistent_cross_model_badcase", "bad_metric_count", "fraction_models_bad", "MAE"],
        ascending=[False, False, False, False],
    )
    report.to_csv(args.output_dir / "per_real_mutant_joint_diagnosis.csv", index=False)
    all_models.to_csv(args.output_dir / "all_models_per_mutant_metrics.csv", index=False)
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    persistent = report[report["persistent_cross_model_badcase"]]
    primary = report[report["bad_any"]]
    markdown = [
        "# Pure-random real-mutant bad-case diagnosis",
        "",
        f"Reference run: `{args.reference}`  ",
        f"Models included: {all_models['model'].nunique()}; real mutants: {len(report)}.",
        "",
        f"Primary bad-case rule: MAE > {args.mae}, Relative L2 > {args.rel_l2}, or correlation < {args.corr}.",
        f" It flags **{len(primary)}** mutants; **{len(persistent)}** are bad in at least half of the evaluated architectures.",
        "",
        "## Persistent joint bad cases",
        "",
        "| Mutant | Pattern | MAE | MSE | relL2 | corr | bad models | exact coverage |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in persistent.head(30).iterrows():
        exact = row.get("Synth_Exact_Count", "unknown")
        markdown.append(
            f"| {row['Mutant']} | {row.get('Pattern', '')} | {row['MAE']:.4f} | {row['MSE']:.4f} | "
            f"{row['Relative_L2']:.4f} | {row['Correlation']:.4f} | {row['fraction_models_bad']:.0%} | {exact} |"
        )
    markdown.extend([
        "",
        "`Synth_Exact_Count` comes from the supplied coverage file; it is evidence about sampling coverage, not a model metric. "
        "Recreate that table whenever the dataset generator or random seed changes.",
    ])
    (args.output_dir / "REPORT.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir / 'per_real_mutant_joint_diagnosis.csv'}")
    print(f"Primary bad cases: {len(primary)}; persistent cross-model bad cases: {len(persistent)}")


if __name__ == "__main__":
    main()
