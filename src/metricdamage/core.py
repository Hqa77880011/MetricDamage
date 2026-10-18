import numpy as np
from scipy.stats import norm


METHODS = [
    "metricdamage",
    "passive",
    "trace_utility",
    "no_cross_covariance",
    "no_scale_contract",
    "random_action",
]


def project(theta, points):
    scale = theta[0]
    H = np.r_[theta[1:], 1.0].reshape(3, 3)
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ H.T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-10):
        raise ValueError("Boundary projects onto the horizon")
    return scale * homogeneous[:, :2] / homogeneous[:, 2:]


def dimensions(theta, boundary, shape):
    points = np.asarray(boundary).reshape(-1, 2)
    center_count, width_count, outline_count = shape
    center = project(theta, points[:center_count])
    sections = project(theta, points[center_count : center_count + 2 * width_count]).reshape(
        width_count, 2, 2
    )
    outline = project(theta, points[-outline_count:])
    length = np.linalg.norm(np.diff(center, axis=0), axis=1).sum()
    width = np.linalg.norm(sections[:, 1] - sections[:, 0], axis=1).mean()
    area = 0.5 * abs(
        np.dot(outline[:, 0], np.roll(outline[:, 1], -1))
        - np.dot(outline[:, 1], np.roll(outline[:, 0], -1))
    )
    return np.array([length, width, area])


def jacobian(function, value):
    value = np.asarray(value, dtype=float)
    columns = []
    for i in range(len(value)):
        step = 1e-5 * max(1, abs(value[i]))
        high, low = value.copy(), value.copy()
        high[i] += step
        low[i] -= step
        columns.append((function(high) - function(low)) / (2 * step))
    return np.column_stack(columns)


def covariance_intersection(mean1, covariance1, mean2, covariance2):
    options = []
    for weight in np.linspace(0, 1, 101):
        precision = weight * np.linalg.inv(covariance1) + (1 - weight) * np.linalg.inv(covariance2)
        covariance = np.linalg.inv(precision)
        mean = covariance @ (
            weight * np.linalg.solve(covariance1, mean1)
            + (1 - weight) * np.linalg.solve(covariance2, mean2)
        )
        options.append((np.linalg.slogdet(covariance)[1], mean, covariance))
    _, mean, covariance = min(options, key=lambda x: x[0])
    return mean, covariance


def fuse_observation(mean, covariance, observation):
    """Fuse a realized dimension observation, preserving unknown correlation."""
    measured = np.asarray(observation["dimensions"], dtype=float)
    noise = np.asarray(observation["covariance"], dtype=float)
    if measured.shape != (3,) or not np.isfinite(measured).all() or np.any(measured <= 0):
        raise ValueError("Observed length/width/area must be three finite positive values")
    if (
        noise.shape != (3, 3)
        or not np.isfinite(noise).all()
        or not np.allclose(noise, noise.T)
        or np.linalg.eigvalsh(noise).min() <= 0
    ):
        raise ValueError("Observed dimension covariance must be positive definite")
    if type(observation.get("independent")) is not bool:
        raise ValueError("Declare whether the observation is independent of the prior")
    if not observation["independent"]:
        return covariance_intersection(mean, covariance, measured, noise)
    # A realized measurement, rather than predicted information, moves the mean.
    gain = np.linalg.solve(covariance + noise, covariance).T
    following_mean = mean + gain @ (measured - mean)
    identity = np.eye(3)
    following_covariance = (
        (identity - gain) @ covariance @ (identity - gain).T + gain @ noise @ gain.T
    )
    return following_mean, (following_covariance + following_covariance.T) / 2


def generate(rng, count):
    for index in range(count):
        length, width = rng.uniform(0.2, 1), rng.uniform(0.01, 0.1)
        center = [[0, 0], [length / 2, 0], [length, 0]]
        sections = [
            [length / 3, -width / 2],
            [length / 3, width / 2],
            [2 * length / 3, -width / 2],
            [2 * length / 3, width / 2],
        ]
        outline = [[0, -width / 2], [length, -width / 2], [length, width / 2], [0, width / 2]]
        boundary = np.array(center + sections + outline)
        observed = boundary + np.array(
            [[rng.gauss(0, 0.001), rng.gauss(0, 0.001)] for _ in boundary]
        )
        theta = [1 + rng.gauss(0, 0.01), 1, 0, 0, 0, 1, 0, 0, 0]
        size = len(theta) + observed.size
        sigma = np.eye(size) * 1e-6
        sigma[0, 0] = 0.0001
        actions = [
            {
                "name": "lateral",
                "information": np.diag([10, 3000, 1000]).tolist(),
                "cost": 0.5,
                "seconds": 2.0,
                "reachable": True,
            },
            {
                "name": "forward",
                "information": np.diag([1000, 10, 500]).tolist(),
                "cost": 1.0,
                "seconds": 1.0,
                "reachable": True,
            },
        ]
        yield {
            "id": str(index),
            "family": f"defect-{index}",
            "theta": theta,
            "boundary": observed.ravel().tolist(),
            "shape": [3, 2, 4],
            "joint_covariance": sigma.tolist(),
            "anchor_valid": rng.random() > 0.1,
            "truth": [length, width, length * width],
            "tolerances": [0.02, 0.005, 0.005],
            "actions": actions,
            "minimum_value": 0.001,
            "minimum_gain": 0.0001,
        }


def run_case(case, method, observer=None):
    theta = np.asarray(case["theta"], dtype=float)
    boundary = np.asarray(case["boundary"], dtype=float)
    covariance = np.asarray(case["joint_covariance"], dtype=float)
    if not np.allclose(covariance, covariance.T) or np.linalg.eigvalsh(covariance).min() < -1e-10:
        raise ValueError("Joint covariance must be positive semidefinite")
    if method == "no_cross_covariance":
        covariance = covariance.copy()
        covariance[: len(theta), len(theta) :] = 0
        covariance[len(theta) :, : len(theta)] = 0
    valid = case["anchor_valid"] and theta[0] > 0 and np.sqrt(covariance[0, 0]) / theta[0] <= 0.1
    if not valid and method != "no_scale_contract":
        return {"valid": False, "dimensions": None, "covariance": None, "action": None}
    joint = np.r_[theta, boundary]
    function = lambda value: dimensions(value[: len(theta)], value[len(theta) :], case["shape"])
    mean = function(joint)
    J = jacobian(function, joint)
    propagated = J @ covariance @ J.T + np.eye(3) * 1e-12
    W = np.diag(1 / np.asarray(case["tolerances"]))

    def risk(sigma):
        matrix = W @ sigma @ W
        return (
            float(np.trace(matrix))
            if method == "trace_utility"
            else float(np.linalg.eigvalsh(matrix)[-1])
        )

    old = risk(propagated)
    options = []
    for index, action in enumerate(case["actions"]):
        if not action["reachable"]:
            continue
        info = np.asarray(action["information"])
        if np.linalg.eigvalsh(info).min() < -1e-10:
            raise ValueError("Candidate measurement information must be positive semidefinite")
        following = np.linalg.inv(np.linalg.inv(propagated) + info)
        reduction = old - risk(following)
        options.append((reduction / (action["cost"] + 1e-8), reduction, index, following))
    selected = max(options, key=lambda option: option[0], default=None)
    if method == "random_action" and options:
        selected = options[int(case["id"].split(":")[-1]) % len(options)]
    if (
        method == "passive"
        or selected is None
        or selected[0] <= case["minimum_value"]
        or selected[1] <= case["minimum_gain"]
    ):
        selected = None
    final_mean, final_covariance = mean, propagated
    observation = None
    action_status = "passive"
    if selected is not None:
        action_index = selected[2]
        name = case["actions"][action_index]["name"]
        observation = (
            observer(case["id"], name)
            if observer is not None
            else case.get("observations", {}).get(name)
        )
        action_status = "pending"
        if observation is not None:
            if observation.get("case_id") != case["id"] or observation.get("action") != name:
                raise ValueError("Action observation does not match the selected case/action")
            action_status = observation.get("status")
            if action_status not in ("observed", "failed"):
                raise ValueError("Action receipt status must be observed or failed")
            costs = [observation.get("cost"), observation.get("seconds")]
            if any(value is None or not np.isfinite(value) or value < 0 for value in costs):
                raise ValueError("Action receipt requires measured nonnegative cost and seconds")
            if action_status == "observed":
                final_mean, final_covariance = fuse_observation(mean, propagated, observation)
    return {
        "valid": True,
        "dimensions": final_mean.tolist(),
        "covariance": final_covariance.tolist(),
        "prior_dimensions": mean.tolist(),
        "prior_covariance": propagated.tolist(),
        "action": selected[2] if selected is not None else None,
        "action_status": action_status,
        "observation": observation,
        "predicted_gain": selected[1] if selected is not None else 0.0,
        "predicted_covariance": selected[3].tolist() if selected is not None else propagated.tolist(),
        "updated_covariance": final_covariance.tolist(),
    }


def score(case, result):
    if not result["valid"]:
        return {
            "invalid_measurement": 1,
            "width_mae_cm": None,
            "length_ape": None,
            "area_ape": None,
            "coverage90": None,
            "interval_width90": None,
            "bias_width_cm": None,
            "legal_effects": 0,
        }
    prediction = np.asarray(result["dimensions"])
    truth = np.asarray(case["truth"])
    errors = prediction - truth
    std = np.sqrt(np.diag(result["covariance"]))
    radius = norm.ppf(0.95) * std
    action = case["actions"][result["action"]] if result["action"] is not None else None
    gain = result["predicted_gain"]
    observed = result["action_status"] == "observed"
    receipt = result["observation"]
    prior = np.asarray(result["prior_dimensions"])
    W = np.diag(1 / np.asarray(case["tolerances"]))
    prior_risk = np.linalg.eigvalsh(W @ np.asarray(result["prior_covariance"]) @ W)[-1]
    final_risk = np.linalg.eigvalsh(W @ np.asarray(result["covariance"]) @ W)[-1]
    actual_gain = float(prior_risk - final_risk) if observed else None
    return {
        "invalid_measurement": 0,
        "width_mae_cm": float(abs(errors[1]) * 100),
        "length_ape": float(abs(errors[0]) / truth[0]),
        "area_ape": float(abs(errors[2]) / truth[2]),
        "coverage90": float(np.mean(np.abs(errors) <= radius)),
        "interval_width90": float(np.mean(2 * radius)),
        "bias_width_cm": float(errors[1] * 100),
        "predicted_risk_reduction_per_meter": gain / action["cost"]
        if action and action["cost"] > 0 else None,
        "predicted_risk_reduction_per_second": gain / action["seconds"]
        if action and action["seconds"] > 0 else None,
        "observed_risk_reduction_per_meter": actual_gain / receipt["cost"]
        if observed and receipt["cost"] > 0 else None,
        "observed_risk_reduction_per_second": actual_gain / receipt["seconds"]
        if observed and receipt["seconds"] > 0 else None,
        "width_error_improvement_cm": float((abs(prior[1] - truth[1]) - abs(errors[1])) * 100)
        if observed else None,
        "action_pending": int(result["action_status"] == "pending"),
        "action_failures": int(result["action_status"] == "failed"),
        "motion_cost": receipt["cost"] if receipt is not None else 0.0,
        "capture_seconds": receipt["seconds"] if receipt is not None else 0.0,
        "extra_captures": int(observed),
        "legal_effects": 1,
    }
