# SpaGCN Baseline

SpaGCN is an external baseline. `dump_all.sh` attempts it after the core dump and before report rendering so the report tables and bar charts can include it.

It is still separate from learned mix and can run through a separate Python environment.

Run one dataset:

```bash
bash scripts/spagcn_baseline.sh DATASET_ID
```

Run with a separate environment:

```bash
XENUM_SPAGCN_PYTHON=/path/to/spagcn-env/bin/python bash scripts/spagcn_baseline.sh DATASET_ID
```

Pipeline controls:

```bash
XENUM_SKIP_SPAGCN=1 bash scripts/dump_all.sh
XENUM_SPAGCN_FORCE=1 bash scripts/dump_all.sh
XENUM_SPAGCN_REQUIRED=1 bash scripts/dump_all.sh
```

Outputs are written to:

```text
outputs/<dataset>/baselines/spagcn/
```

Expected files:

- `domains.csv`: SpaGCN spatial domain assignment per cell.
- `probabilities.csv`: SpaGCN assignment probabilities.
- `edges.csv`: top-k graph artifact derived from SpaGCN adjacency.
- `predictions_spagcn_adjacency_k*.csv`: XY reconstruction from the SpaGCN adjacency graph.
- `bench_xy.csv`: benchmark row aligned with the existing XY reconstruction tables where possible.
- `domains_summary.csv`: per-domain size and spatial compactness.
- `summary.json`: runtime, parameters, image/histology status, environment, and output paths.

The adapter loads SpaGCN through `PYTHONPATH` from `external/SpaGCN/SpaGCN_package` by default, but does not modify that checkout.
