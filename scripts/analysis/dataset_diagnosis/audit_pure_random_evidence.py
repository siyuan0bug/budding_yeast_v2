#!/usr/bin/env python3
"""Post-hoc audit of the actual pure-random NPZ and frozen checkpoints.

This is an evaluator, never a sampler or training-data selector. Benchmark
parameters/trajectories are used only to explain existing evaluation failures.
No archived coverage CSV or manually inferred gene-mode parser is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT.parent))
VARIABLES = "MASS CLN2 CLB2 CLB5 SIC1 CDC6 C2 C5 F2 F5 SIC1P C2P C5P CDC6P F2P F5P SWI5T SWI5 IEP CDC20T CDC20 CDH1T CDH1 CDC14T CDC14 NET1T NET1 RENT TEM1 CDC15 PPX PDS1 ESP1 ORI BUD SPN Vi20 lte1 BUB2".split()
DEFAULT_DATASET = Path("/home/users/zsy/mydata/budding_yeast/mutant_aware_pure_random_100000_210m_500s.npz")
EXP = ROOT / "experiments/mutant_aware_pure_random_100000_210m_500s"
REFERENCE = EXP / "biom_refiner_plus/triple_dynamic_none/w192_m24_b256_ep100_lr8.8e-4_wm8/seed_42/eval/per_mutant_metrics.csv"


def split_indices(names, patterns, seed):
    """Mirror the current non-LHS DataModule split, including subgroup order."""
    if any("_LHS_" in n for n in names):
        raise ValueError("This audit expects the non-LHS pure-random dataset")
    real = np.flatnonzero([bool(re.match(r"^\d+(?:\.\d+)?_", n)) for n in names])
    synth = np.setdiff1d(np.arange(len(names)), real)
    rng = np.random.RandomState(seed)
    groups = {}
    for idx in synth:
        groups.setdefault(str(patterns[idx]).split(":")[0], []).append(idx)
    train, val, test = [], [], []
    for idx in groups.values():
        rng.shuffle(idx)
        n = len(idx)
        qv = qt = math.ceil(n * .1)
        if n <= 2:
            qv = qt = 0
        elif n <= 4:
            qt = 0
        qt = min(qt, n - qv)
        val.extend(idx[:qv]); test.extend(idx[qv:qv + qt]); train.extend(idx[qv + qt:])
    for idx in (train, val, test):
        rng.shuffle(idx)
    return real, synth, np.array(train), np.array(val), np.array(test)


def data_chunks(path, chunk_size=512):
    """Stream data.npy once; avoid materializing the 7.8 GB trajectory array."""
    with zipfile.ZipFile(path) as archive, archive.open("data.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"Unsupported NPY version: {version}")
        if fortran or dtype.hasobject:
            raise ValueError("Expected a numeric C-order trajectory array")
        stride = int(np.prod(shape[1:])) * dtype.itemsize
        for start in range(0, shape[0], chunk_size):
            count = min(chunk_size, shape[0] - start)
            block = stream.read(count * stride)
            if len(block) != count * stride:
                raise EOFError("Truncated data.npy")
            yield start, np.frombuffer(block, dtype=dtype).reshape(count, *shape[1:])


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_geometry(p, names, patterns, param_names, real, synth, train, val, test):
    # This distance is in the legacy/default parameter representation actually
    # used by the baseline: global-synthetic z-score, not min-max or gene space.
    scale = p[synth].std(axis=0, ddof=1) + 1e-6
    ps = p[train]
    train_min, train_max = ps.min(0), ps.max(0)
    rows = []
    for i in real:
        distances = np.linalg.norm((ps - p[i]) / scale, axis=1)
        nearest = np.argsort(distances)[:20]
        zeros = p[i] == 0
        outside = (p[i] < train_min - 1e-9) | (p[i] > train_max + 1e-9)
        rows.append({
            "Mutant": names[i], "Pattern": patterns[i],
            "distance_min": distances[nearest[0]], "distance_top20_mean": distances[nearest].mean(),
            "nearest_train_name": names[train[nearest[0]]],
            "nearest_train_pattern": patterns[train[nearest[0]]],
            "same_pattern_fraction_top20": np.mean(patterns[train[nearest]] == patterns[i]),
            "train_same_zero_face": int(np.all(ps[:, zeros] == 0, axis=1).sum()),
            "train_exact_parameter_count": int(np.all(ps == p[i], axis=1).sum()),
            "range_oob_parameters": ",".join(param_names[outside]),
        })
    # Parameter equality is only a duplicate *candidate*: different ICs can
    # distinguish trajectories. Label that limitation explicitly in the output.
    _, inv = np.unique(p[synth], axis=0, return_inverse=True)
    lookup = np.full(len(p), -1, dtype=int); lookup[synth] = inv
    train_groups = np.unique(lookup[train])
    duplicate_info = {
        "synthetic_unique_parameter_vectors": int(np.unique(inv).size),
        "validation_parameter_matches_training": int(np.isin(lookup[val], train_groups).sum()),
        "test_parameter_matches_training": int(np.isin(lookup[test], train_groups).sum()),
        "duplicate_scope": "parameter vectors only; equality of initial states must also be checked",
    }
    return pd.DataFrame(rows), duplicate_info


def audit_complete_input_duplicates(path, p, names, real, train, val, test, output_dir):
    """Check full (P, IC) equality and entire target-trajectory equality.

    Equality before the shared normalization proves equality of actual network
    inputs; further collisions caused by rounding after normalization are not
    counted. This therefore provides a lower bound on duplicate inputs.
    """
    ic = np.zeros((len(p), len(VARIABLES)), dtype=np.float32)
    digests = np.empty(len(p), dtype="S32")
    for start, block in data_chunks(path):
        stop = start + len(block)
        ic[start:stop] = block[:, :, 0]
        digests[start:stop] = [hashlib.sha256(memoryview(y).cast("B")).digest() for y in block]
        if start % 10240 == 0:
            print(f"Full input/label duplicates: {stop:,} samples read", flush=True)
    physical_p = np.nan_to_num(p.astype(np.float32), nan=0, posinf=1e4, neginf=0)
    ic = np.log1p(np.maximum(np.nan_to_num(ic, nan=0, posinf=1e4, neginf=0), 0))
    features = np.concatenate((physical_p, ic), axis=1)
    _, group = np.unique(features, axis=0, return_inverse=True)
    training_labels = {}
    for idx in train:
        training_labels.setdefault(int(group[idx]), set()).add(digests[idx].tobytes())
    summary = {"scope": "exact equality of float32 P and transformed IC; same time grid for every sample",
               "lower_bound_due_to_normalization_rounding": True, "splits": {}}
    matched_rows = []
    for split, indices in (("val_synth", val), ("test_synth", test), ("test_real", real)):
        same_input = same_label = 0
        for idx in indices:
            labels = training_labels.get(int(group[idx]), set())
            has_input = bool(labels)
            has_target = digests[idx].tobytes() in labels
            same_input += has_input; same_label += has_target
            if has_input:
                matched_rows.append({"split": split, "index": idx, "Mutant": names[idx],
                                     "input_group": group[idx], "identical_target_in_training": has_target})
        summary["splits"][split] = {"total": len(indices), "identical_input_in_training": same_input,
                                      "identical_input_and_full_target_in_training": same_label}
    pd.DataFrame(matched_rows).to_csv(output_dir / "duplicate_inputs.csv", index=False)
    (output_dir / "duplicate_inputs_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def load_trajectory_stats(path, real, synth, cache_path, need_full):
    identity = {"path": str(path.resolve()), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    if cache_path.exists():
        with np.load(cache_path) as cache:
            if json.loads(str(cache["identity"])) == identity:
                return cache["real_y"], cache["mean_y"], cache["std_y"]
    real_blocks, sums, squares, nsteps = [], None, None, 0
    for start, block in data_chunks(path):
        stop = start + len(block)
        selected = real[(real >= start) & (real < stop)] - start
        if len(selected):
            real_blocks.append(block[selected].copy())
        if need_full:
            selected_s = synth[(synth >= start) & (synth < stop)] - start
            if len(selected_s):
                transformed = np.log1p(np.maximum(block[selected_s], 0)).astype(np.float64)
                s = transformed.sum((0, 2)); sq = (transformed ** 2).sum((0, 2))
                sums = s if sums is None else sums + s
                squares = sq if squares is None else squares + sq
                nsteps += transformed.shape[0] * transformed.shape[2]
            if start % 10240 == 0:
                print(f"Trajectory statistics: {stop:,} samples read", flush=True)
        elif stop > real.max():
            break
    real_y = np.concatenate(real_blocks)
    if not need_full:
        return real_y, None, None
    mean_y = (sums / nsteps).astype(np.float32).reshape(1, -1, 1)
    var = (squares - sums ** 2 / nsteps) / (nsteps - 1)
    std_y = (np.sqrt(np.maximum(var, 0)).astype(np.float32) + 1e-6).reshape(1, -1, 1)
    np.savez_compressed(cache_path, real_y=real_y, mean_y=mean_y, std_y=std_y, identity=json.dumps(identity))
    return real_y, mean_y, std_y


def replay_checkpoint(ckpt_path, names, p, synth, real, truth, mean, std, out_dir, batch_size):
    import torch
    from budding_yeast_v2.models.ours.bio_refiner_plus import BioRefinerPlus
    from budding_yeast_v2.utils.metrics import calculate_metrics

    torch.set_num_threads(4)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = checkpoint["hyper_parameters"]
    if hp["model_name"] != "biom_refiner_plus":
        raise ValueError("This replay currently supports the frozen BioRefinerPlus checkpoints")
    config_path = ckpt_path.parent.parent / "config.json"
    config = json.loads(config_path.read_text())["args"]
    norm = hp.get("norm_type", config.get("norm_type"))
    if norm != "log1p":
        raise ValueError(f"Refusing to replay {norm} with log1p statistics")
    model = BioRefinerPlus(num_vars=hp.get("num_vars", 39), param_dim=hp.get("param_dim", 141),
                           width=hp["width"], modes=hp["modes"], n_blocks=hp.get("n_blocks", 4))
    state = {k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    model.eval()
    physical_p = torch.as_tensor(p, dtype=torch.float32)
    pm = physical_p[synth].mean(0, keepdim=True)
    ps = physical_p[synth].std(0, keepdim=True) + 1e-6
    pnorm = (physical_p[real] - pm) / ps
    mean_t, std_t = torch.from_numpy(mean), torch.from_numpy(std)
    y = torch.from_numpy(truth.copy())
    ynorm = (torch.log1p(y.clamp(min=0)) - mean_t) / std_t
    ic = ynorm[:, :, :1].expand(-1, -1, truth.shape[2])
    grid = torch.linspace(0, 1, truth.shape[2])[None, None].expand_as(ic)
    x = torch.stack((ic, grid), dim=2)
    predictions, coarse_predictions = [], []
    with torch.inference_mode():
        for start in range(0, len(real), batch_size):
            prednorm, coarsenorm = model(x[start:start + batch_size], pnorm[start:start + batch_size], return_coarse=True)
            pred = torch.expm1((prednorm * std_t + mean_t).clamp(-10, 15)).clamp(min=0)
            predictions.append(pred)
            coarse_predictions.append(torch.expm1((coarsenorm * std_t + mean_t).clamp(-10, 15)).clamp(min=0))
    predictions = torch.cat(predictions)
    coarse_predictions = torch.cat(coarse_predictions)
    run_tag = config["loss_type"]
    rows, per_channel = [], []
    for j, i in enumerate(real):
        metrics = calculate_metrics(predictions[j], y[j])
        rows.append({"Mutant": names[i], **metrics})
        a = predictions[j].double().numpy(); b = truth[j].astype(np.float64)
        err = a - b; ac = a - a.mean(-1, keepdims=True); bc = b - b.mean(-1, keepdims=True)
        mse = (err ** 2).mean(-1)
        corr = (ac * bc).sum(-1) / (np.sqrt((ac ** 2).sum(-1) * (bc ** 2).sum(-1)) + 1e-8)
        for v, label in enumerate(VARIABLES):
            per_channel.append({"Mutant": names[i], "variable": label, "MAE": np.abs(err[v]).mean(),
                                "MSE": mse[v], "mse_fraction_in_mutant": mse[v] / max(mse.sum(), 1e-30),
                                "Relative_L2": np.sqrt((err[v] ** 2).sum() / max((b[v] ** 2).sum(), .01 * len(b[v]))),
                                "corr_float64": corr[v], "target_range": np.ptp(b[v]),
                                "target_max": b[v].max(), "pred_max": a[v].max(),
                                "target_dynamic": np.ptp(b[v]) > 1e-7})
    current = pd.DataFrame(rows).rename(columns={"Relative L2": "Relative_L2"})
    saved_path = ckpt_path.parent.parent / "eval/per_mutant_metrics.csv"
    saved = pd.read_csv(saved_path)
    combined = current.merge(saved, on="Mutant", suffixes=("_replay", "_saved"), validate="one_to_one")
    differences = {m: float(np.max(np.abs(combined[m + "_replay"] - combined[m + "_saved"])))
                   for m in ("MAE", "MSE", "Relative_L2", "Correlation")}
    pd.DataFrame(per_channel).to_csv(out_dir / f"{run_tag}_per_channel.csv", index=False)
    current.to_csv(out_dir / f"{run_tag}_replay_metrics.csv", index=False)
    # Initial-state identity is needed to verify the parameter-only duplicate audit.
    np.savez_compressed(out_dir / f"{run_tag}_predictions.npz", predictions=predictions.numpy(), targets=truth, names=names[real])
    stage_rows = []
    for j, i in enumerate(real):
        coarse_metrics = calculate_metrics(coarse_predictions[j], y[j])
        final_metrics = calculate_metrics(predictions[j], y[j])
        row = {"Mutant": names[i], **{k + "_coarse": v for k, v in coarse_metrics.items()},
               **{k + "_final": v for k, v in final_metrics.items()}}
        for v in (1, 33, 36):
            row[VARIABLES[v] + "_peak_coarse"] = float(coarse_predictions[j, v].max())
            row[VARIABLES[v] + "_peak_final"] = float(predictions[j, v].max())
        row["tail_sse_fraction_final"] = float(
            ((predictions[j, :, -50:] - y[j, :, -50:]) ** 2).sum() /
            ((predictions[j] - y[j]) ** 2).sum().clamp(min=1e-30))
        stage_rows.append(row)
    pd.DataFrame(stage_rows).to_csv(out_dir / f"{run_tag}_stage_comparison.csv", index=False)
    print(f"Replay {run_tag}: max differences from saved CSV {differences}", flush=True)
    return {"checkpoint": str(ckpt_path), "checkpoint_sha256": fingerprint(ckpt_path),
            "config": str(config_path), "norm": norm, "strict_model_state_load": True,
            "max_abs_difference_from_saved_metrics": differences,
            "pooled_metrics": calculate_metrics(predictions, y),
            "coarse_pooled_metrics": calculate_metrics(coarse_predictions, y),
            "refiner_scale_tanh_gamma": float(torch.tanh(model.gamma).detach()),
            "macro_metrics": {m: float(current[m].mean()) for m in ("MAE", "MSE", "Relative_L2", "Correlation")}}


def plot_diagnostic_trajectories(out_dir):
    """An explanatory scientific figure; these IDs never affect training."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(out_dir / "triple_dynamic_predictions.npz") as saved:
        truth, pred, names = saved["targets"], saved["predictions"], saved["names"]
    panels = [("15", "CLN2"), ("15", "Vi20"), ("22", "ORI"),
              ("124", "ORI"), ("42", "ORI"), ("76", "ORI")]
    t = np.linspace(0, 210, truth.shape[-1])
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), constrained_layout=True)
    for ax, (mid, variable) in zip(axes.flat, panels):
        i = next(j for j, n in enumerate(names) if str(n).startswith(mid + "_"))
        v = VARIABLES.index(variable)
        ax.plot(t, truth[i, v], label="ODE target", color="#1b4f72", linewidth=1.7)
        ax.plot(t, pred[i, v], label="Frozen BioRefiner+ (triple)", color="#a93226", linestyle="--", linewidth=1.4)
        ax.set(title=f"Mutant {mid}: {variable}", xlabel="Time (min)", ylabel="Model state (a.u.)")
        ax.grid(alpha=.15)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.savefig(out_dir / "badcase_trajectories.png", dpi=180)
    plt.close(fig)


def audit_ori_consistency(out_dir, parameters, param_names):
    """Fixed equation probe, with a target-driven control used only for audit.

    The predicted driver uses only model outputs, query parameters, and IC.
    The oracle driver is a numerical-accuracy control, not a predictor.
    """
    with np.load(out_dir / "triple_dynamic_predictions.npz") as saved:
        target = saved["targets"].astype(np.float64)
        pred = saved["predictions"].astype(np.float64)
        names = saved["names"]
    ix = {k: j for j, k in enumerate(param_names)}
    dt = 210 / (target.shape[-1] - 1)
    kd = parameters[:, ix["kdori"]]
    decay = np.exp(-kd * dt)
    gain = np.divide(-np.expm1(-kd * dt), kd, out=np.full_like(kd, dt), where=kd != 0)

    def integrate(driver):
        source = parameters[:, ix["ksori"], None] * (
            parameters[:, ix["eorib5"], None] * driver[:, 3] +
            parameters[:, ix["eorib2"], None] * driver[:, 2])
        ori = np.zeros_like(driver[:, 33])
        ori[:, 0] = target[:, 33, 0]  # This is the given input IC.
        cyclins = driver[:, 2] + driver[:, 3]
        for t in range(1, ori.shape[-1]):
            ori[:, t] = decay * ori[:, t - 1] + gain * .5 * (source[:, t - 1] + source[:, t])
            event2 = (cyclins[:, t - 1] > .2) & (cyclins[:, t] <= .2)
            ori[event2, t] = 0
        return ori

    probe, oracle = integrate(pred), integrate(target)
    table = pd.DataFrame({"Mutant": names,
                          "original_ori_mse": ((pred[:, 33] - target[:, 33]) ** 2).mean(-1),
                          "predicted_driver_ori_mse": ((probe - target[:, 33]) ** 2).mean(-1),
                          "oracle_control_ori_mse": ((oracle - target[:, 33]) ** 2).mean(-1)})
    table.to_csv(out_dir / "ori_consistency_probe.csv", index=False)
    return {"scope": "post-hoc equation-consistency probe; oracle control is not a deployable model",
            "original_mean_ori_mse": float(table.original_ori_mse.mean()),
            "predicted_driver_mean_ori_mse": float(table.predicted_driver_ori_mse.mean()),
            "oracle_control_mean_ori_mse": float(table.oracle_control_ori_mse.mean()),
            "mutants_improved": int((table.predicted_driver_ori_mse < table.original_ori_mse).sum())}


def audit_attractor_input():
    """Check IC preservation with identifiable values independent of weights."""
    import torch
    from budding_yeast_v2.models.ours.bio_refiner_attractor import EquilibriumRelaxationStream

    x = torch.zeros(1, 39, 2, 5)
    x[:, :, 0, :] = torch.arange(1, 40)[None, :, None]
    x[:, :, 1, :] = torch.linspace(0, 1, 5)
    with torch.no_grad():
        y = EquilibriumRelaxationStream()(x.reshape(1, 78, 5), torch.zeros(1, 141))
    expected, actual = x[0, :, 0, 0], y[0, :, 0]
    return {"wrong_initial_channels": int(((expected - actual).abs() > 1e-5).sum()),
            "expected_first_six": expected[:6].tolist(), "actual_first_six": actual[:6].tolist()}


def audit_causal_reference(p, synth, param_names, output_dir):
    """WT-only probe: no benchmark parameter or trajectory is needed."""
    import torch
    from budding_yeast_v2.data.mutant_aware import get_default_params
    from budding_yeast_v2.models.ours.bio_refiner_causal import WT_VALUES_141

    physical = get_default_params()
    physical.update(init_CDH1T=1., init_CDH1=.9304992, init_BUB2=.2)
    wt = torch.tensor([physical[k] for k in param_names], dtype=torch.float32)
    ref = torch.tensor(WT_VALUES_141, dtype=torch.float32)
    values = torch.tensor(p[synth], dtype=torch.float32)
    normalized = (wt - values.mean(0)) / (values.std(0) + 1e-6)
    table = pd.DataFrame({"parameter": param_names, "physical_wt": wt.numpy(),
                          "gate_reference": ref.numpy(), "normalized_wt_input": normalized.numpy(),
                          "implemented_delta_for_wt": (normalized - ref).numpy()})
    table.to_csv(output_dir / "causal_wt_reference_probe.csv", index=False)
    result = {"scope": "WT and synthetic normalization statistics only",
              "physical_wt_reference_max_difference": float((wt - ref).abs().max()),
              "normalized_wt_minus_physical_reference_l2": float(torch.linalg.vector_norm(normalized - ref))}
    (output_dir / "causal_wt_reference_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def audit_dedup_alias(output_dir):
    """Explain an observed overlap; never use this probe as a sampling filter."""
    from budding_yeast_v2.data import mutant_aware as engine

    wt = engine.get_default_params()
    wt.update(init_CDH1T=1., init_CDH1=.9304992, init_BUB2=.2)
    candidate = {}
    engine.apply_gene_mutation("MAD2", "CSL", 1., wt, wt.copy(), candidate)
    candidate["mdt"] = 90.
    without_alias = {k: v for k, v in candidate.items() if k != "Vi20"}
    result = {"scope": "post-hoc reproduction of dictionary/ODE canonicalization mismatch",
              "candidate": candidate,
              "duplicate_filter_with_alias": bool(engine.is_duplicate_real_mutant(candidate, wt)),
              "duplicate_filter_without_alias": bool(engine.is_duplicate_real_mutant(without_alias, wt)),
              "warning": "Do not turn this real-dictionary probe into a blind training-data selector"}
    (output_dir / "dedup_alias_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def summarize_frozen_evidence(output_dir, reference_path):
    """One row for every real mutant, with observed facts rather than causal tags.

    Dynamic-channel correlations are supplementary diagnostics. They do not
    replace the saved benchmark metrics or determine a model/sampling choice.
    """
    with np.load(output_dir / "triple_dynamic_predictions.npz") as saved:
        pred = saved["predictions"].astype(np.float64)
        target = saved["targets"].astype(np.float64)
        names = saved["names"].astype(str)
    err = pred - target
    channel_mse = np.mean(err ** 2, axis=-1)
    pc = pred - pred.mean(-1, keepdims=True)
    tc = target - target.mean(-1, keepdims=True)
    corr = (pc * tc).sum(-1) / (np.sqrt((pc ** 2).sum(-1) * (tc ** 2).sum(-1)) + 1e-8)
    ranges = np.ptp(target, axis=-1)
    dynamic = ranges > 1e-7
    # Fixed sensitivity threshold, not selected to optimize a particular mutant.
    substantial = ranges > 1e-3 * np.maximum(1., np.abs(target).max(-1))
    rows = []
    for j, name in enumerate(names):
        order = np.argsort(channel_mse[j])[::-1]
        sse = channel_mse[j].sum()
        rows.append({"Mutant": name,
                     "leading_sse_channel": VARIABLES[order[0]],
                     "leading_channel_sse_fraction": channel_mse[j, order[0]] / max(sse, 1e-30),
                     "ori_sse_fraction": channel_mse[j, 33] / max(sse, 1e-30),
                     "mse_excluding_ori": np.delete(channel_mse[j], 33).mean(),
                     "tail10_sse_fraction": (err[j, :, -50:] ** 2).sum() / max((err[j] ** 2).sum(), 1e-30),
                     "constant_target_channels": int((~dynamic[j]).sum()),
                     "corr_dynamic_float64": float(corr[j, dynamic[j]].mean()) if dynamic[j].any() else np.nan,
                     "substantially_dynamic_channels": int(substantial[j].sum()),
                     "corr_substantial_float64": float(corr[j, substantial[j]].mean()) if substantial[j].any() else np.nan})
    table = pd.read_csv(reference_path).merge(pd.DataFrame(rows), on="Mutant", validate="one_to_one")
    if len(table) != len(names):
        raise ValueError("Reference metrics do not match all frozen real-mutant predictions")
    table["flag_mae"] = table.MAE > .15
    table["flag_mse"] = table.MSE > 1.
    table["flag_rel_l2"] = table.Relative_L2 > .25
    table["flag_corr"] = table.Correlation < .75
    table["flag_any_metric"] = table[["flag_mae", "flag_mse", "flag_rel_l2", "flag_corr"]].any(axis=1)
    optional = [
        (output_dir / "per_mutant_geometry.csv",
         ["Mutant", "distance_min", "same_pattern_fraction_top20", "train_same_zero_face", "train_exact_parameter_count"]),
        (output_dir / "triple_dynamic_stage_comparison.csv",
         ["Mutant", "MSE_coarse", "MSE_final", "CLN2_peak_coarse", "CLN2_peak_final"]),
    ]
    for path, columns in optional:
        if path.exists():
            table = table.merge(pd.read_csv(path)[columns], on="Mutant", validate="one_to_one")
    events_path = output_dir / "ode/predicted_event_consistency.csv"
    if events_path.exists():
        event_table = pd.read_csv(events_path)
        keep = ["Mutant"] + [c for c in event_table if c.endswith(("_missing", "_extra", "_ordinal_time_mae"))]
        table = table.merge(event_table[keep], on="Mutant", validate="one_to_one")
        count_cols = [c for c in keep if c.endswith(("_missing", "_extra"))]
        table["any_event_count_error"] = table[count_cols].sum(axis=1) > 0
    duplicates_path = output_dir / "input_duplicate_check/duplicate_inputs.csv"
    if duplicates_path.exists():
        duplicates = pd.read_csv(duplicates_path)
        duplicates = duplicates[(duplicates.split == "test_real") & duplicates.identical_target_in_training]
        table["identical_input_and_target_in_training"] = table.Mutant.isin(duplicates.Mutant)
    table.to_csv(output_dir / "per_mutant_diagnostic_summary.csv", index=False)
    table[table.flag_any_metric].to_csv(output_dir / "flagged_mutant_diagnostic_summary.csv", index=False)
    summary = {"scope": "post-hoc descriptive evidence, not causal attribution or model selection",
               "mutants": len(table), "flagged_by_fixed_four_metric_rule": int(table.flag_any_metric.sum()),
               "dynamic_correlation_threshold": "target range > 1e-7",
               "substantial_correlation_threshold": "target range > 1e-3 * max(1, max(abs(target)))",
               "dedup_alias_probe": audit_dedup_alias(output_dir)}
    stats_path = output_dir / "normalization_cache.npz"
    if stats_path.exists():
        with np.load(stats_path) as stats:
            scale = stats["std_y"].astype(np.float64)
        encoded_error = (np.log1p(pred) - np.log1p(np.maximum(target, 0))) / scale
        physical_sse = (err ** 2).sum((0, 2))
        encoded_sse = (encoded_error ** 2).sum((0, 2))
        pd.DataFrame({"variable": VARIABLES, "physical_sse": physical_sse,
                      "physical_sse_fraction": physical_sse / physical_sse.sum(),
                      "reencoded_sse": encoded_sse,
                      "reencoded_sse_fraction": encoded_sse / encoded_sse.sum()}).to_csv(
                          output_dir / "physical_vs_encoded_error.csv", index=False)
        summary["error_space_comparison"] = {
            "scope": "same postprocessed nonnegative predictions reencoded; not raw training loss or gradient shares",
            "ori_physical_sse_fraction": float(physical_sse[33] / physical_sse.sum()),
            "ori_reencoded_sse_fraction": float(encoded_sse[33] / encoded_sse.sum()),
        }
    (output_dir / "joined_evidence_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay-ckpt", type=Path, action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--plot", action="store_true", help="Plot the replayed triple-dynamic diagnostic trajectories")
    parser.add_argument("--input-duplicates", action="store_true", help="Stream the entire NPZ to check (P, IC, full trajectory) duplicates")
    parser.add_argument("--summary-only", action="store_true", help="Join existing frozen prediction, geometry, stage and event outputs without model replay")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "scripts/analysis/outputs/pure_random_current_audit")
    args = parser.parse_args()
    if args.plot and not any("triple_dynamic" in str(p) for p in args.replay_ckpt):
        parser.error("--plot needs the frozen triple_dynamic checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        print(json.dumps(summarize_frozen_evidence(args.output_dir, args.reference), indent=2))
        return
    with np.load(args.dataset, allow_pickle=True) as dataset:
        names = dataset["mutant_names"].astype(str); p = dataset["params"]
        patterns = dataset["pattern_labels"].astype(str); param_names = dataset["param_names"].astype(str)
    real, synth, train, val, test = split_indices(names, patterns, args.seed)
    geometry, duplicates = audit_geometry(p, names, patterns, param_names, real, synth, train, val, test)
    full_duplicates = audit_complete_input_duplicates(args.dataset, p, names, real, train, val, test, args.output_dir) if args.input_duplicates else None
    reference = pd.read_csv(args.reference)
    geometry.merge(reference.drop(columns="Pattern"), on="Mutant", validate="one_to_one").to_csv(args.output_dir / "per_mutant_geometry.csv", index=False)
    truth, mean, std = load_trajectory_stats(args.dataset, real, synth, args.output_dir / "normalization_cache.npz", bool(args.replay_ckpt))
    np.savez_compressed(args.output_dir / "frozen_real_targets.npz", targets=truth, names=names[real])
    ranges = np.ptp(truth, axis=-1)
    pd.DataFrame({"Mutant": names[real], "constant_channels": (ranges <= 1e-7).sum(-1),
                  "nonconstant_channel_fraction": (ranges > 1e-7).mean(-1)}).to_csv(args.output_dir / "correlation_dynamic_channels.csv", index=False)
    replays = [replay_checkpoint(path, names, p, synth, real, truth, mean, std, args.output_dir, args.batch_size) for path in args.replay_ckpt]
    ori_probe = None
    if any("triple_dynamic" in str(path) for path in args.replay_ckpt):
        ori_probe = audit_ori_consistency(args.output_dir, p[real], param_names)
    if args.plot:
        plot_diagnostic_trajectories(args.output_dir)
    if any("triple_dynamic" in str(path) for path in args.replay_ckpt):
        summarize_frozen_evidence(args.output_dir, args.reference)
    metadata = {"scope": "post-hoc diagnostic only; never used to select training samples",
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "dataset": str(args.dataset), "dataset_size": args.dataset.stat().st_size,
                "reference": str(args.reference), "reference_sha256": fingerprint(args.reference),
                "seed": args.seed, "n_real": len(real), "n_synth": len(synth),
                "n_train": len(train), "n_val": len(val), "n_test": len(test),
                "synthetic_patterns": pd.Series(patterns[synth]).value_counts().to_dict(),
                "real_patterns": pd.Series(patterns[real]).value_counts().to_dict(),
                "parameter_duplicates": duplicates,
                "complete_input_duplicates": full_duplicates,
                "reference_mse_top2_fraction": float(reference.nlargest(2, "MSE").MSE.sum() / reference.MSE.sum()),
                "attractor_ic_probe": audit_attractor_input(),
                "causal_wt_reference_probe": audit_causal_reference(p, synth, param_names, args.output_dir),
                "ori_consistency_probe": ori_probe,
                "replays": replays}
    (args.output_dir / "audit.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k not in ("replays",)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
