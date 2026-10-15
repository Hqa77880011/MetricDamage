# MetricDamage

Implementation of **Observability-guided metric damage mapping for monocular road inspection**.

Applies an explicit scale contract, propagates full joint geometry/boundary covariance including cross terms and selects actions by the dominant tolerance-normalized uncertainty mode.

## Install

Use Python 3.11 or newer.

```bash
git clone https://github.com/Hqa77880011/MetricDamage.git
cd MetricDamage
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install.

```bash
python -m pip install -e .
```

## Data preparation

Geometry contains metric scale and eight homography coefficients. Boundary arrays concatenate centerline points, width-section endpoint pairs and an outline polygon; shape records their counts. joint_covariance covers all geometry and boundary coordinates. Candidate actions provide independent 3x3 measurement information, reachability, distance and time costs. Independent physical dimensions supply truth.

```bash
python -m metricdamage prepare --count 200 --seed 17 --output data/workload.jsonl
```

The generator creates finite workloads from the formal model. Each JSONL row is one scenario. `--count` controls corpus size and `--seed` controls generation. Imported cases use the same schema. Generate calibration and test scenarios independently.

## Execute and compare

```bash
python -m metricdamage run --data data/workload.jsonl --output runs/main
python -m metricdamage suite --data data/workload.jsonl --output runs/comparison
```

Select methods with `--methods`. metricdamage, passive, trace_utility, no_cross_covariance, no_scale_contract and random_action.

## Evaluation and visualization

Width MAE/bias in centimeters, length/area APE, 90% marginal interval coverage/width, invalid-measurement rate, predicted risk reduction per meter/second and extra captures.

Each method writes `records.jsonl` and `metrics.json` with means, sample standard deviations, usable counts and P50/P95/P99. Rates are fractions. Undefined conditional metrics remain unavailable. Method timers exclude reference scoring. RSS includes the Python runtime; serialized sizes measure JSON bytes.

```bash
python -m metricdamage plot --inputs runs/comparison/metricdamage --output figures
```

Pass multiple method directories to compare them. Plotting writes `summary.csv` and metric PNGs.

## Implementation conventions

Finite-difference Jacobians propagate the declared covariance. Scale must be positive, anchored and have relative standard deviation <=0.1. Width averages registered cross-sections and length follows a supplied centerline. The module also supplies covariance intersection for correlated map observations. Predicted action reduction is named separately from measured post-action error improvement.


## Uncertainty and paired comparisons

```bash
python -m metricdamage.statistics --records runs/main/metricdamage/records.jsonl --output results/intervals.json
```

Supply --reference with a matched record file for paired differences. Resampling uses independent case IDs, or explicit cluster IDs when provided. Repeated observations from one source unit must share cluster. Zero-event bounds refer to an event in an independent cluster.
