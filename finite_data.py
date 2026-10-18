"""Finite geometry plus independent lognormal dimension sensors, not robot measurements."""

import argparse
import json
from pathlib import Path
import random

import numpy as np

from metricdamage.core import generate


def sensor(dimensions, relative_sigma, rng):
    """Sample a positive unbiased sensor reading from a physical latent size."""
    value = np.asarray(dimensions) * np.exp(rng.normal(0, relative_sigma, 3) - relative_sigma**2 / 2)
    # Instrument-reported uncertainty uses the reading, never evaluation labels.
    covariance = np.diag(np.square(value) * np.expm1(relative_sigma**2))
    return value.tolist(), covariance.tolist()


def cases(count, seed):
    rng = np.random.default_rng(seed + 1)
    for case in generate(random.Random(seed), count):
        case["id"] = f"finite:{seed}:{case['id']}"
        case["cluster"] = case["id"]
        case["input_source"] = "finite_geometry_and_independent_lognormal_sensors"
        case["observations"] = {}
        for action in case["actions"]:
            record = dict(case_id=case["id"], action=action["name"],
                          status="failed" if rng.random() < 0.08 else "observed",
                          cost=float(action["cost"] * rng.uniform(0.95, 1.15)),
                          seconds=float(action["seconds"] * rng.uniform(0.9, 1.2)),
                          source="finite_sensor", cost_source="declared_finite_motion_model")
            if record["status"] == "observed":
                value, covariance = sensor(case["truth"], 0.015 if action["name"] == "lateral" else 0.025, rng)
                record.update(dimensions=value, covariance=covariance, independent=True)
            case["observations"][action["name"]] = record
        yield case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("Positive case count required")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for case in cases(args.count, args.seed):
            stream.write(json.dumps(case, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
