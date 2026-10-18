# MetricDamage

Implementation of **Observability-guided metric damage mapping for monocular road inspection**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data

Geometry contains metric scale and eight homography coefficients. Boundary arrays concatenate centerline points, width-section endpoint pairs and an outline polygon; shape records their counts. joint_covariance covers all geometry and boundary coordinates. Candidate actions provide independent 3x3 measurement information, reachability, distance and time costs. Independent physical dimensions supply truth.

The finite generator writes geometry and independent modeled post-action sensor readings as JSONL; costs and observations are generated quantities. Fix the seed and keep the generated input for paired comparisons.

```bash
python finite_data.py --output data/generated.jsonl --count 200 --seed 17
```

## Execution and evaluation

```bash
python -m metricdamage run --data data/generated.jsonl --output runs/main
```

## Comparisons and ablations

```bash
python -m metricdamage suite --data data/generated.jsonl --output runs/comparison
```

Use `--methods` to select: `metricdamage`, `passive`, `trace_utility`, `no_cross_covariance`, `no_scale_contract`, `random_action`.

## Outputs

`runs/main/metricdamage/records.jsonl` stores per-case results; `metrics.json` stores aggregate measurements. Comparison outputs use one directory per method.

```bash
python -m metricdamage plot --inputs runs/comparison/metricdamage --output figures
```

Plotting writes `figures/summary.csv` and metric PNGs. Statistical intervals can be computed from a retained record file:

```bash
python -m metricdamage.statistics --records runs/main/metricdamage/records.jsonl --output results/intervals.json
```
