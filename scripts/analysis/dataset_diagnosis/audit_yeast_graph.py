#!/usr/bin/env python3
"""Build/audit source-derived potential topology; never simulate or train.

Default: print a summary without writing anything. --output explicitly exports
one JSON record/provenance bundle, never the three historical binary assets.
Legacy comparisons are diagnostic only; no signed/relational artifact is fixed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT.parent))
from budding_yeast_v2.data.yeast_graph import DEFAULT_SOURCE, build_yeast_graph
from budding_yeast_v2.data.yeast_relations import build_yeast_relations


def compare_legacy(graph, directory):
    """Check old *support* with explicit orientation; do not validate edge signs."""
    directory = Path(directory)
    result = {}
    for filename, view in (("yeast_signed_adjacency_matrix.npy", "continuous"),
                           ("yeast_boolean_adjacency_matrix.npy", "hybrid")):
        path = directory / filename
        if not path.exists():
            result[filename] = {"status": "missing"}
            continue
        actual = np.load(path, allow_pickle=False)
        expected = graph.adjacency(view, self_loops="all")
        result[filename] = {"shape": list(actual.shape), "view": view,
                            "support_matches_with_neural_self_loops": bool(
                                actual.shape == expected.shape and np.array_equal(actual != 0, expected)),
                            "signs_checked": False}
    path = directory / "yeast_multirelational_graphs.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            relations = data["rel_graphs"]
            expected_shape = (6, len(graph.variable_names), len(graph.variable_names))
            if relations.shape != expected_shape:
                raise ValueError(f"Unexpected legacy relational shape: {relations.shape}")
            # Historical builder uses [source, target]; no model buffer is changed.
            actual = (relations != 0).any(0).T
            continuous = graph.adjacency("continuous", self_loops="none")
            np.fill_diagonal(actual, False)
            result[path.name] = {
                "stored_orientation": "source_target", "comparison_orientation": "target_source",
                "unique_nonself_edges": int(actual.sum()),
                "matches_continuous": int((actual & continuous).sum()),
                "missing_continuous": int((~actual & continuous).sum()),
                "outside_continuous": int((actual & ~continuous).sum()),
                "other_fields_preserved": [k for k in data.files if k != "rel_graphs"],
                "note": "Legacy relations are not a new authoritative topology or sign proof",
            }
    else:
        result[path.name] = {"status": "missing"}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--check-legacy", action="store_true", help="Read-only support comparison")
    parser.add_argument('--relations', action='store_true', help='Audit ODE term-derived typed/untyped support')
    parser.add_argument('--view', choices=['continuous', 'hybrid'], default='hybrid')
    parser.add_argument("--legacy-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, help="Optional NEW .json record bundle; refuses overwrite")
    args = parser.parse_args(argv)
    graph = build_yeast_graph(args.source)
    summary = graph.summary()
    relations = build_yeast_relations(args.source) if args.relations else None
    if relations is not None:
        summary['relations'] = relations.summary(args.view)
    if args.check_legacy:
        summary["legacy_comparison"] = compare_legacy(graph, args.legacy_dir)
    if args.output is not None:
        if args.output.suffix.lower() != ".json":
            parser.error("--output must be a .json file, not a historical .npy/.npz asset")
        # Exclusive creation protects old evidence and makes export explicitly opt-in.
        with args.output.open("x", encoding="utf-8") as stream:
            payload = graph.to_dict()
            if relations is not None:
                payload['relations'] = {**relations.summary(args.view),
                                        'term_annotations': [asdict(r) for r in relations.records]}
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        summary["exported_records"] = str(args.output.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
