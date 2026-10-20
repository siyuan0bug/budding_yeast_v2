#!/usr/bin/env python3
"""Post-hoc ODE label reproduction and numerical-tolerance audit.

Reads the frozen benchmark to test its numerical consistency. It does not
generate training data, propose parameter regions, or select model weights.
Only scalar diagnostic metrics and solver traces are written.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT.parent))
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_pure_random_evidence import (
    DEFAULT_DATASET, REFERENCE, data_chunks,
)


def compare(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    e = a - b
    return {"mae": float(np.abs(e).mean()), "mse": float((e * e).mean()),
            "relative_l2": float(np.sqrt((e * e).sum(-1) / np.maximum((b * b).sum(-1), .01 * b.shape[-1])).mean()),
            "max_abs": float(np.abs(e).max())}


def _timeout(signum, frame):
    raise TimeoutError("ODE diagnostic time limit exceeded")


def worker(task):
    mid, name, target, tolerances, timeout = task
    from budding_yeast_v2.data.systems.budding import simulation as engine
    from budding_yeast_v2.data.systems.budding.real_mutants import REAL_MUTANTS
    original_solver = engine.solve_ivp
    results = []
    original_y = None
    for rtol in tolerances:
        calls = []

        def traced_solver(*args, **kwargs):
            kwargs["rtol"] = rtol
            kwargs["atol"] = rtol * .01
            sol = original_solver(*args, **kwargs)
            event_ids = [j + 1 for j, ts in enumerate(sol.t_events) if len(ts)]
            calls.append({"status": int(sol.status), "t_end": float(sol.t[-1]),
                          "nfev": int(sol.nfev), "events": event_ids})
            return sol

        engine.solve_ivp = traced_solver
        start = time.monotonic()
        row = {"id": mid, "Mutant": name, "rtol": rtol, "atol": rtol * .01}
        signal.signal(signal.SIGALRM, _timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            _, y, label, pattern = engine.simulate_mutant(name, dict(REAL_MUTANTS[mid][1]), tf=210, ns=500)
            if y is None:
                row.update(status="simulation_failed", label=label, pattern=pattern)
            else:
                if original_y is None:
                    original_y = y
                row.update(status="ok", label=label, pattern=pattern)
                row.update({"stored_" + k: v for k, v in compare(y, target).items()})
                row.update({"control_" + k: v for k, v in compare(y, original_y).items()})
        except Exception as exc:
            row.update(status="exception", error=repr(exc))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            engine.solve_ivp = original_solver
        row.update(elapsed_seconds=time.monotonic() - start,
                   solve_calls=len(calls),
                   final_integrated_time=calls[-1]["t_end"] if calls else None,
                   failed_solver_segments=sum(c["status"] < 0 for c in calls),
                   event_count=sum(c["status"] == 1 for c in calls),
                   event_sequence="-".join(str(e) for c in calls for e in c["events"]),
                   solver_trace=json.dumps(calls))
        results.append(row)
    return results


def audit_predicted_events(predictions_path, control_path, output_dir):
    """Apply the four published event surfaces to frozen predicted curves.

    Matching is by event type and ordinal occurrence. Missing/extra events are
    counted separately, since timing averages alone would conceal their absence.
    """
    with np.load(predictions_path) as saved:
        predictions = saved["predictions"].astype(np.float64)
        names = saved["names"].astype(str)
    control = pd.read_csv(control_path)
    control = control[control.rtol == 1e-4].set_index("Mutant")
    if control.empty:
        raise ValueError("Event audit requires the original rtol=1e-4 control rows")
    dt = 210 / (predictions.shape[-1] - 1)
    rows = []
    for name, y in zip(names, predictions):
        truth_events = []
        for segment in json.loads(control.loc[name, "solver_trace"]):
            truth_events.extend((e, segment["t_end"]) for e in segment["events"])
        predicted_events = []
        surfaces = [(y[2], .3, False), (y[2] + y[3], .2, False),
                    (y[33], 1., True), (y[35], 1., True)]
        for ei, (v, threshold, up) in enumerate(surfaces, 1):
            crossings = ((v[:-1] < threshold) & (v[1:] >= threshold)) if up else (
                (v[:-1] > threshold) & (v[1:] <= threshold))
            for j in np.flatnonzero(crossings):
                t = (j + (threshold - v[j]) / (v[j + 1] - v[j])) * dt
                predicted_events.append((ei, float(t)))
        row = {"Mutant": name,
               "true_events": json.dumps(truth_events),
               "predicted_events": json.dumps(sorted(predicted_events, key=lambda e: e[1]))}
        for ei in range(1, 5):
            actual = [t for e, t in truth_events if e == ei]
            inferred = [t for e, t in predicted_events if e == ei]
            n = min(len(actual), len(inferred))
            row.update({f"event{ei}_true_count": len(actual),
                        f"event{ei}_pred_count": len(inferred),
                        f"event{ei}_missing": max(0, len(actual) - len(inferred)),
                        f"event{ei}_extra": max(0, len(inferred) - len(actual)),
                        f"event{ei}_first_t_true": actual[0] if actual else None,
                        f"event{ei}_first_t_pred": inferred[0] if inferred else None,
                        f"event{ei}_ordinal_time_mae": float(np.mean(np.abs(np.array(actual[:n]) - inferred[:n]))) if n else None})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "predicted_event_consistency.csv", index=False)
    summary = {"scope": "post-hoc threshold-crossing diagnostic, not a trained event classifier",
               "predictions": str(predictions_path), "mutants": len(table), "events": {}}
    for ei in range(1, 5):
        summary["events"][str(ei)] = {
            "mutants_with_missing_events": int((table[f"event{ei}_missing"] > 0).sum()),
            "mutants_with_extra_events": int((table[f"event{ei}_extra"] > 0).sum()),
            "mutants_with_wrong_event_count": int((table[f"event{ei}_true_count"] != table[f"event{ei}_pred_count"]).sum()),
            "mean_ordinal_time_mae_when_matched": float(table[f"event{ei}_ordinal_time_mae"].mean()),
        }
    (output_dir / "predicted_event_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "scripts/analysis/outputs/pure_random_current_audit/ode")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=5.)
    parser.add_argument("--future-timeout", type=float, default=35., help="Parent-side deadline when retrieving a worker result")
    parser.add_argument("--rtol", type=float, nargs="+", default=[1e-4, 1e-6, 1e-8])
    parser.add_argument("--predictions", type=Path, help="Optional frozen prediction NPZ for event-surface checks")
    parser.add_argument("--events-only", action="store_true", help="Reuse the existing ODE audit CSV")
    args = parser.parse_args()
    if (any(not np.isfinite(t) or t <= 0 for t in args.rtol)
            or not np.isfinite(args.timeout) or args.timeout <= 0
            or not np.isfinite(args.future_timeout) or args.future_timeout <= 0 or args.workers < 1):
        parser.error("Tolerances/timeouts must be finite and positive; workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.events_only:
        if args.predictions is None:
            parser.error("--events-only requires --predictions")
        print(json.dumps(audit_predicted_events(args.predictions, args.output_dir / "ode_label_audit.csv", args.output_dir), indent=2))
        return
    with np.load(args.dataset, allow_pickle=True) as z:
        names = z["mutant_names"].astype(str)
        patterns = z["pattern_labels"].astype(str)
    real = np.flatnonzero([bool(re.match(r"^\d+(?:\.\d+)?_", n)) for n in names])
    targets = {}
    for start, block in data_chunks(args.dataset):
        for i in real[(real >= start) & (real < start + len(block))]:
            targets[int(i)] = block[i - start].copy()
        if start + len(block) > real.max():
            break
    tasks = [(names[i].split("_", 1)[0], names[i], targets[int(i)], args.rtol, args.timeout) for i in real]
    records = []
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)
    try:
        futures = [pool.submit(worker, task) for task in tasks]
        # Submission order keeps the deadline meaningful even with queued jobs;
        # each running task has its own short per-tolerance SIGALRM deadline.
        for n, future in enumerate(futures, 1):
            records.extend(future.result(timeout=args.future_timeout))
            if n % 20 == 0 or n == len(futures):
                print(f"ODE checks: {n}/{len(futures)} mutants", flush=True)
    except BaseException:
        # Python 3.10 lacks a public terminate_workers API. Only terminate this
        # executor's own diagnostic children, then join them so shutdown cannot
        # wait indefinitely on a solver that did not handle SIGALRM.
        for process in list((getattr(pool, "_processes", None) or {}).values()):
            if process.is_alive():
                process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    table = pd.DataFrame(records)
    original_patterns = dict(zip(names[real], patterns[real]))
    table["stored_pattern"] = table.Mutant.map(original_patterns)
    table["pattern_changed"] = table.pattern != table.stored_pattern
    table = table.merge(pd.read_csv(args.reference)[["Mutant", "MAE", "MSE", "Relative_L2", "Correlation"]],
                        on="Mutant", validate="many_to_one")
    table.to_csv(args.output_dir / "ode_label_audit.csv", index=False)
    summary = {
        "scope": "post-hoc numerical convergence audit; no training dataset is produced",
        "dataset": str(args.dataset), "reference": str(args.reference),
        "tolerances": args.rtol, "mutants": len(real),
        "worker_timeout_seconds": args.timeout, "future_timeout_seconds": args.future_timeout,
        "by_tolerance": {},
    }
    for tol, g in table.groupby("rtol"):
        ok = g[g.status == "ok"]
        summary["by_tolerance"][str(tol)] = {
            "ok": len(ok), "pattern_changes": int(ok.pattern_changed.sum()),
            "truncated_horizon": int((ok.final_integrated_time < 210 - 1e-6).sum()),
            "failed_solver_segments": int(g.failed_solver_segments.sum()),
            "mean_mae_vs_stored": float(ok.stored_mae.mean()),
            "max_mae_vs_stored": float(ok.stored_mae.max()),
            "mean_mae_vs_control": float(ok.control_mae.mean()),
        }
    if args.predictions is not None:
        summary["predicted_events"] = audit_predicted_events(args.predictions, args.output_dir / "ode_label_audit.csv", args.output_dir)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
