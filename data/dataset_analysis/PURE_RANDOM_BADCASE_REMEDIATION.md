# Pure-random real-mutant bad-case remediation

## Scope and evidence

This note concerns `mutant_aware_pure_random_100000_210m_500s.npz`, not the
older MASS dataset. Reproduce the diagnosis with:

```bash
python budding_yeast_v2/data/dataset_analysis/analyze_pure_random_badcases.py
```

It reads 49 model result files and uses the BioMRefiner+ triple-dynamic run as
the reference. With `MAE > 0.15 OR relL2 > 0.25 OR corr < 0.75`, it identifies
20 reference bad cases and 50 persistent cross-model bad cases (bad for at
least half the models).

The generator samples genes uniformly, samples modes uniformly conditional on
the gene, and fixes each mutation level 1--4 at 25%. This creates sparse
high-order coupled real combinations. It is a distribution problem, not merely
a neural-operator architecture bug.

## Strict blind-protocol audit

Under a strict blind protocol, the current generator cannot be called blind:
it imports the benchmark-containing module, appends `REAL_MUTANTS` at lines
193--198, and calls `is_duplicate_real_mutant` at lines 218--223. The last
operation consults held-out parameter vectors while deciding what synthetic
training samples to retain. The existing pure-random model still does not train
on real trajectories (the DataModule keeps them in `test_real`), but the data
construction protocol has benchmark access.

The admissible replacement is a two-artifact workflow:

1. A **blind training generator** produces synthetic trajectories only. It must
   neither import nor read real-mutant rules, names, parameter vectors,
   trajectories, coverage tables, or per-mutant results. In particular it must
   not perform benchmark-based deduplication/rescue.
2. A separately held immutable **benchmark artifact** contains the 126 real
   mutants and is attached only by the evaluator after checkpoint selection.
   The generator's random seed, graph, priors, quotas and ODE acceptance rules
   are frozen before this attachment.

An accidental equality between a blindly generated discrete synthetic sample
and a hidden benchmark member cannot be proactively filtered without leakage.
It should instead be reported post-hoc by an independent evaluator; it must not
alter the generation or model-selection decision.

## Causal ablation already in the repository

`generate_bams_ablation_datasets.py` constructs a matched 50k control and two
50k datasets with 10k targeted samples. All reported runs use the same
BioMRefiner+, hybrid loss, width, modes, batch size, epochs, and seed.

| Mutant | 50k control MAE / relL2 / corr | hard-targeted | hypo-targeted | Interpretation |
| --- | --- | --- | --- | --- |
| 15 | 0.738 / 7.247 / 0.018 | 0.0026 / 0.0087 / 0.797 | 0.0039 / 0.0126 / 0.789 | Local support solves the arrest branch. |
| 22 | 0.527 / 0.533 / 0.404 | 0.107 / 0.159 / 0.864 | 0.045 / 0.050 / 0.912 | Hypomorphic samples suit this GOF combination. |
| 76 | 0.043 / 0.097 / 0.836 | 0.053 / 0.132 / 0.807 | 0.038 / 0.102 / 0.828 | Already well covered; use little added budget. |
| 124 | 0.699 / 0.191 / 0.720 | 1.024 / 0.106 / 0.860 | 1.805 / 0.113 / 0.835 | Peak amplitude needs its own family. |

The targeted BAMS ablation is evidence that *coverage* matters, but it is not
an admissible remedy for the blind benchmark: selecting regions from real-mutant
parameter rules is test-set leakage. Do not use it to construct the training
dataset or report it as a Sim-to-Real result.

## Recommended next experiment

Keep the 100k synthetic budget and all 126 real mutants completely inaccessible
during generation, training and model selection. The only admissible inputs are
the ODE, its parameter priors, gene/mode catalogue, and a predeclared signed
regulatory graph. Do not use real-mutant names, rules, parameter vectors,
trajectories, per-mutant metrics, or a list derived from them.

Use three *blind, predeclared* sampling families:

1. **Topology/motif combinations**: choose connected or feed-forward motifs in
   the signed regulatory graph, stratified by one through four mutations and by
   mutation mode. This replaces uniform independent gene selection.
2. **Event-boundary discovery**: before any model is trained, run ODE samples
   from the prior and retain samples near automatically detected division/event
   boundaries, with Pattern A/B/C quotas. The labels arise only from synthetic
   ODE trajectories.
3. **Amplitude-tail stratification**: reserve a fixed prior-defined fraction of
   GAL-OE/MC parameter scales for upper-tail values, independently of any real
   benchmark outcome.

Start at 40k / 30k / 30k samples across these families. Fix this allocation,
the graph, random seed and acceptance criteria before looking at real-mutant
metrics. Log attempted, accepted, ODE-failed and Pattern counts.

## Training matrix and gates

Run seeds 42, 43 and 44 with unchanged architecture, schedule and split seed:

| Dataset | normalization | loss | Purpose |
| --- | --- | --- | --- |
| Pure-random | log1p | triple_dynamic | Existing reference. |
| Blind topology/event/tail sampler | log1p | triple_dynamic | Blind distribution intervention. |
| Blind topology/event/tail sampler | asinh_scale | hybrid_mse_rel | Blind scale intervention. |

Call the blind sampler better only when mean real MAE, MSE and relL2 improve,
mean corr does not decrease, and results replicate over all three seeds. Report
Pattern A/B/C means separately, but do not use real-mutant subgroups to alter
the frozen sampler.
