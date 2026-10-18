#!/usr/bin/env python3
"""Read-only loss/sampling-contract diagnostics, not a training experiment.

Fixed synthetic fixtures test current loss implementations. Frozen real outputs
are used only to locate errors in the loss's temporal support. No sampler,
training configuration, model weight, or benchmark metric is changed.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT.parent))
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_pure_random_evidence import (
    DEFAULT_DATASET, REFERENCE, VARIABLES, split_indices,
)
from budding_yeast_v2.utils.losses import RelativeL2Loss, TripleDynamicLoss
from budding_yeast_v2.utils.metrics import relative_l2_loss


def reference_soft_dtw(x, y, gamma):
    """Independent scalar recurrence, with no shared vectorized indexing."""
    n = x.shape[-1]
    inf = x.new_full((x.shape[0],), float("inf"))
    previous = [x.new_zeros(x.shape[0])] + [inf] * n
    for i in range(n):
        current = [inf]
        for j in range(n):
            choices = torch.stack([previous[j], previous[j + 1], current[j]], -1)
            value = (x[:, i] - y[:, j]) ** 2 - gamma * torch.logsumexp(-choices / gamma, -1)
            current.append(value)
        previous = current
    return previous[-1].mean()


def loss_probes(output_dir):
    criterion = TripleDynamicLoss()

    def dtw(p, y):
        a = p[:, criterion.core_indices, ::criterion.sub_stride].flatten(0, 1)
        b = y[:, criterion.core_indices, ::criterion.sub_stride].flatten(0, 1)
        return criterion._soft_dtw_anti_diag(a, b)

    target = torch.zeros(1, 39, 500)
    base_dtw, base_total = float(dtw(target, target)), float(criterion(target, target))
    rows = []
    for label, variable, begin, end in (
        ("late_CLN2", 1, 481, 500),
        ("unsampled_interior_CLN2", 1, 10, 11),
        ("whole_ORI", 33, 0, 500),
        ("whole_Vi20", 36, 0, 500),
        ("sampled_early_CLN2", 1, 0, 20),
    ):
        pred = target.clone()
        pred[:, variable, begin:end] = 1.
        rows.append({"fixture": label, "variable": VARIABLES[variable], "begin": begin, "end": end,
                     "mse": float(((pred - target) ** 2).mean()),
                     "dtw_change": float(dtw(pred, target)) - base_dtw,
                     "total_loss_change": float(criterion(pred, target)) - base_total})
    pd.DataFrame(rows).to_csv(output_dir / "temporal_channel_support_probes.csv", index=False)

    relative_rows = []
    for length in (125, 500, 1000):
        y = torch.zeros(1, 1, length)
        p = torch.full_like(y, .01)
        relative_rows.append({"time_steps": length, "constant_error": .01,
                              "training_relative_term": float(RelativeL2Loss()(p, y)),
                              "evaluation_relative_metric": float(relative_l2_loss(p, y))})
    pd.DataFrame(relative_rows).to_csv(output_dir / "relative_floor_probes.csv", index=False)
    y = torch.zeros(1, 1, 500)
    p = torch.full_like(y, .2, requires_grad=True)
    capped = RelativeL2Loss()(p, y)
    capped_grad = torch.autograd.grad(capped, p)[0]
    uncapped = RelativeL2Loss(max_clamp=None)(p, y)
    uncapped_grad = torch.autograd.grad(uncapped, p)[0]

    # Validate the suspected mechanism against an independent implementation,
    # rather than treating a negative Soft-DTW value as an implementation bug.
    t = torch.linspace(0, 1, 25)
    x = torch.stack([torch.sin(2 * torch.pi * t), torch.cos(4 * torch.pi * t)]).requires_grad_()
    y = torch.stack([torch.sin(2 * torch.pi * t + .2), torch.cos(4 * torch.pi * t - .1)])
    vectorized = criterion._soft_dtw_anti_diag(x, y)
    reference = reference_soft_dtw(x, y, criterion.gamma)
    gv = torch.autograd.grad(vectorized, x, retain_graph=True)[0]
    gr = torch.autograd.grad(reference, x)[0]

    # The full objective contains nonsmooth L2 norms: a nonzero chosen autograd
    # subgradient at equality alone does NOT prove that equality is not optimal.
    t = torch.linspace(0, 1, 500)
    y = torch.stack([torch.sin(2 * torch.pi * (1 + j % 3) * t + .1 * j) + .2 * t
                     for j in range(39)])[None]
    p = y.clone().requires_grad_()
    exact_loss = criterion(p, y)
    g = torch.autograd.grad(exact_loss, p)[0]
    with torch.no_grad():
        candidate = p - .05 * g / g.norm()
        candidate_loss = criterion(candidate, y)

    return {"scope": "fixed synthetic fixtures only; no model optimization",
            "dtw_core_variables": [VARIABLES[i] for i in criterion.core_indices],
            "dtw_sampled_times": list(range(0, 500, criterion.sub_stride)),
            "zero_target_self_dtw": base_dtw,
            "independent_recurrence_value_difference": float(abs(vectorized - reference)),
            "independent_recurrence_gradient_max_difference": float((gv - gr).abs().max()),
            "relative_cap_probe": {"uncapped_value": float(uncapped.detach()),
                                   "capped_value": float(capped.detach()),
                                   "uncapped_gradient_norm": float(uncapped_grad.norm()),
                                   "capped_gradient_norm": float(capped_grad.norm()),
                                   "note": "Only the relative term is capped, not the total training objective"},
            "exact_prediction_probe": {"loss": float(exact_loss.detach()), "autograd_norm": float(g.norm()),
                                       "fixed_output_step_l2": .05,
                                       "loss_after_fixed_output_step": float(candidate_loss),
                                       "mse_after_step": float(((candidate - y) ** 2).mean()),
                                       "note": "Nonzero subgradient is not proof of a nonoptimal full objective; this fixed step increases loss"}}


def source_provenance(checkpoint):
    config_path = checkpoint.parent.parent / "config.json"
    config = json.loads(config_path.read_text())
    recorded_commit = config["git_commit"]
    old = ast.parse(subprocess.check_output(["git", "show", f"{recorded_commit}:utils/losses.py"], cwd=ROOT, text=True))
    current = ast.parse((ROOT / "utils/losses.py").read_text())
    matching = {}
    for name in ("RelativeL2Loss", "HybridMSERelLoss", "TripleDynamicLoss"):
        selected = []
        for tree in (old, current):
            node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
            for function in node.body:
                if isinstance(function, ast.FunctionDef):
                    function.args.kwarg = None  # Added API compatibility, not a numeric change.
            selected.append(ast.dump(node))
        matching[name] = selected[0] == selected[1]
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    from budding_yeast_v2.models.ours.bio_refiner_plus import BioRefinerPlus
    hp = ckpt["hyper_parameters"]
    if hp["model_name"] != "biom_refiner_plus":
        raise ValueError("Architecture-contract check supports the reference BioRefinerPlus only")
    model = BioRefinerPlus(num_vars=hp["num_vars"], param_dim=hp["param_dim"],
                           modes=hp["modes"], width=hp["width"], n_blocks=hp["n_blocks"])
    state = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    architecture = {"strict_model_state_load": True,
                    "coarse_model_class": type(model.stage1_coarse).__name__,
                    "coarse_use_event": model.stage1_coarse.use_event,
                    "coarse_event_trigger_is_none": model.stage1_coarse.event_trigger is None,
                    "coarse_use_titration": model.stage1_coarse.use_titration,
                    "coarse_use_pathway": model.stage1_coarse.use_pathway,
                    "all_model_weights_finite": all(bool(torch.isfinite(v).all()) for v in state.values())}
    bias = model.stage2_refiner.graph_attn.graph_bias.detach().cpu()
    graph_path = ROOT / "data/yeast_signed_adjacency_matrix.npy"
    raw_graph = np.load(graph_path)
    architecture["graph_contract"] = {
        "checkpoint_graph_bias_offdiagonal_nonzeros": int(torch.count_nonzero(bias - torch.diag(bias.diag()))),
        "checkpoint_graph_bias_identity_max_difference": float((bias - torch.eye(bias.shape[0])).abs().max()),
        "configured_explicit_adj_matrix_path": config["args"].get("adj_matrix_path"),
        "default_relative_path": "budding_yeast_v2/data/yeast_signed_adjacency_matrix.npy",
        "default_path_exists_from_repo_root": (ROOT / "budding_yeast_v2/data/yeast_signed_adjacency_matrix.npy").exists(),
        "actual_repo_graph": str(graph_path),
        "actual_graph_offdiagonal_nonzeros": int(np.count_nonzero(raw_graph - np.diag(raw_graph.diagonal()))),
        "note": "Identity structural bias does not remove learned dense QK attention; no frozen buffer is replaced",
    }
    callbacks = []
    for key, state in ckpt.get("callbacks", {}).items():
        if "ModelCheckpoint" in str(key):
            score = state.get("best_model_score")
            callbacks.append({"monitor": state.get("monitor"), "best_score": float(score) if score is not None else None,
                              "best_model_path": state.get("best_model_path")})
    return {"checkpoint": str(checkpoint), "config": str(config_path),
            "recorded_commit": recorded_commit, "checkpoint_loss_type": ckpt["hyper_parameters"]["loss_type"],
            "checkpoint_epoch": int(ckpt["epoch"]), "normalization_in_config": config["args"]["norm_type"],
            "loss_class_ast_equal_except_kwargs": matching, "checkpoint_callbacks": callbacks,
            "architecture_contract": architecture,
            "note": "This compares recorded-commit code, not an unavailable snapshot of uncommitted training files"}


def sampling_support(dataset, seed, output_dir):
    with np.load(dataset, allow_pickle=True) as z:
        names, patterns, p = z["mutant_names"].astype(str), z["pattern_labels"].astype(str), z["params"]
    real, synth, train, _, _ = split_indices(names, patterns, seed)
    records = []
    for split, indices in (("synth_all", synth), ("synth_train", train), ("test_real", real)):
        for idx in indices:
            match = re.search(r"_L(\d+)_\d+$", names[idx])
            records.append({"split": split, "level": int(match.group(1)) if match else -1,
                            "Pattern": patterns[idx]})
    table = pd.DataFrame(records).groupby(["split", "level", "Pattern"]).size().rename("samples").reset_index()
    table.to_csv(output_dir / "level_pattern_counts.csv", index=False)
    varying = p[train].std(0) > 1e-6
    constant_mismatches = np.abs(p[real][:, ~varying] - p[train[0], ~varying]) > 1e-7
    return {"synthetic_train_size": len(train), "varying_parameter_coordinates_std_gt_1e-6": int(varying.sum()),
            "total_parameter_coordinates": p.shape[1],
            "exactly_constant_training_coordinates": int((np.ptp(p[train], axis=0) == 0).sum()),
            "real_mutants_with_constant_coordinate_mismatch_gt_1e-7": int(constant_mismatches.any(1).sum()),
            "synthetic_pattern_counts": pd.Series(patterns[synth]).value_counts().to_dict(),
            "real_pattern_counts_for_posthoc_comparison_only": pd.Series(patterns[real]).value_counts().to_dict(),
            "note": "Nominal mutation level is not a balanced dynamics class or a count of changed parameter coordinates"}


def frozen_tail_support(predictions, output_dir):
    with np.load(predictions) as z:
        names, pred, target = z["names"].astype(str), z["predictions"].astype(float), z["targets"].astype(float)
    last_sampled = list(range(0, target.shape[-1], TripleDynamicLoss().sub_stride))[-1]
    squared_error = (pred - target) ** 2
    rows = []
    for j, name in enumerate(names):
        rows.append({"Mutant": name,
                     "sse_after_last_dtw_sample_fraction": squared_error[j, :, last_sampled + 1:].sum() / max(squared_error[j].sum(), 1e-30),
                     "cln2_sse_after_last_dtw_sample_fraction": squared_error[j, 1, last_sampled + 1:].sum() / max(squared_error[j, 1].sum(), 1e-30)})
    pd.DataFrame(rows).to_csv(output_dir / "frozen_prediction_tail_support.csv", index=False)
    return {"scope": "post-hoc error localization, not evidence that other loss terms ignore this tail",
            "last_dtw_index": last_sampled, "last_dtw_time_min": 210 * last_sampled / (target.shape[-1] - 1),
            "unobserved_tail_points": target.shape[-1] - last_sampled - 1, "mutants": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=REFERENCE.parent.parent / "checkpoints/best-epoch=91-val_loss_lhs=0.0348.ckpt")
    parser.add_argument("--comparison-checkpoint", type=Path, action="append", default=[], help="Additional frozen BioRefinerPlus source/architecture checks; no inference or selection")
    parser.add_argument("--predictions", type=Path, default=ROOT / "scripts/analysis/outputs/pure_random_current_audit/triple_dynamic_predictions.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "scripts/analysis/outputs/pure_random_current_audit/training_contracts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {"scope": "diagnostic only; production training and sampling code unchanged",
              "source": source_provenance(args.checkpoint),
              "comparison_sources": [source_provenance(p) for p in args.comparison_checkpoint],
              "loss_probes": loss_probes(args.output_dir),
              "sampling_support": sampling_support(args.dataset, args.seed, args.output_dir),
              "frozen_tail_support": frozen_tail_support(args.predictions, args.output_dir)}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
