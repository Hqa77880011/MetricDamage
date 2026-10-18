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

The generator creates synthetic geometry and planning workloads from the formal model. Each JSONL row is one scenario. It does not fabricate a measurement after an action: selected actions remain `pending` until an observation is supplied. `--count` controls corpus size and `--seed` controls generation. Imported cases use the same schema. Generate calibration and test scenarios independently.

## Execute and compare

```bash
python -m metricdamage run --data data/workload.jsonl --output runs/main
python -m metricdamage suite --data data/workload.jsonl --output runs/comparison
```

Select methods with `--methods`. metricdamage, passive, trace_utility, no_cross_covariance, no_scale_contract and random_action.

## Realized action observations

Action ranking uses only the prior covariance, tolerances and candidate information/costs. After selection, `run_case` consumes the selected action's recorded measurement from `case.observations[action_name]`, or invokes the explicitly configured observer. `truth` is read only by scoring. A successful observation updates both dimensions and covariance; predicted information alone never changes the reported measurement.

The following is an illustrative schema, not a recorded robot observation. Dimensions are `[length_m, width_m, area_m2]`; the 3x3 covariance uses the corresponding mixed units. The `case_id` must match the containing row and `action` must match the selected candidate.

```json
{
  "observations": {
    "lateral": {
      "case_id": "route-1:defect-4",
      "action": "lateral",
      "status": "observed",
      "dimensions": [0.50, 0.04, 0.02],
      "covariance": [[0.0001, 0, 0], [0, 0.000004, 0], [0, 0, 0.000001]],
      "independent": false,
      "cost": 0.55,
      "seconds": 2.2
    }
  }
}
```

Provide a finite positive dimension vector and a symmetric positive definite covariance estimated by the measurement frontend. Declare `independent: true` only when the observation error is independent of the prior: fusion then uses a Kalman/Joseph update. Shared or unknown correlation uses covariance intersection (`false`) rather than counting the same image/odometry information twice. Candidate `information` is a prediction and is not substituted for the observed covariance.

A failed attempted action returns only `case_id`, `action`, `status: "failed"`, and measured nonnegative `cost` (meters) / `seconds`; dimensions and covariance stay at the prior. A missing observation returns `pending` with unchanged estimates. Zero measured distance or time makes its per-unit ratio unavailable. No successful capture is counted for pending/failed actions.

For a live measurement frontend:

```bash
python -m metricdamage run --data data/recorded_geometry.jsonl --methods metricdamage --observer road_adapter:observe --output runs/observed
```

The trusted Python callable `observe(case_id, action_name)` executes/looks up only that selected action and returns the observation schema above, a failed receipt, or `None` while unresolved. It receives no ground truth. A robot-specific motion/camera/geometry frontend must implement this callable and supply real costs and covariance; this repository does not infer them from labels. Use separate reset episodes for physical method comparisons, or recorded outcomes for each candidate under the same prior. Running a live observer through `suite` invokes actions separately for each method and requires that reset discipline.

## Evaluation and visualization

Width MAE/bias in centimeters, length/area APE and 90% marginal interval coverage/width use the final realized estimate. Records retain `prior_dimensions`, `prior_covariance`, `predicted_covariance`, `action_status` and the observation. `updated_covariance` is the actual final covariance. Predicted risk reduction is separate from observed dominant-mode risk reduction, measured costs and width-error improvement. Observed gains/error improvements remain unavailable without a successful receipt. Other metrics include invalid-measurement rate, pending/failed actions and successful extra captures.

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
