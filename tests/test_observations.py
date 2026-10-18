"""Small analytic/state counterexamples; these are not damage experiments."""

import copy
import unittest

import numpy as np

from metricdamage.core import fuse_observation, run_case, score


def example():
    boundary = [[0, 0], [0.5, 0], [1, 0],
                [0.3, -0.05], [0.3, 0.05], [0.7, -0.05], [0.7, 0.05],
                [0, -0.05], [1, -0.05], [1, 0.05], [0, 0.05]]
    covariance = np.eye(9 + 2 * len(boundary)) * 1e-6
    covariance[0, 0] = 1e-4
    return dict(id="example", theta=[1, 1, 0, 0, 0, 1, 0, 0, 0],
                boundary=np.asarray(boundary).ravel().tolist(), shape=[3, 2, 4],
                joint_covariance=covariance.tolist(), anchor_valid=True,
                tolerances=[0.02, 0.005, 0.005], truth=[1, 0.08, 0.08],
                actions=[dict(name="lateral", information=(np.eye(3) * 1000).tolist(),
                              reachable=True, cost=0.5, seconds=1)],
                minimum_value=0, minimum_gain=0)


def observation():
    return dict(case_id="example", action="lateral", status="observed",
                dimensions=[1, 0.08, 0.08], covariance=(np.eye(3) * 1e-8).tolist(),
                independent=True, cost=0.6, seconds=1.5)


class ObservationTests(unittest.TestCase):
    def test_independent_analytic_posterior(self):
        mean, covariance = fuse_observation(
            np.array([2., 4., 6.]), np.eye(3) * 4,
            dict(dimensions=[4, 6, 8], covariance=(np.eye(3) * 4).tolist(), independent=True),
        )
        np.testing.assert_allclose(mean, [3, 5, 7])
        np.testing.assert_allclose(covariance, np.eye(3) * 2)

    def test_unknown_correlation_does_not_double_count_information(self):
        _, covariance = fuse_observation(
            np.array([2., 4., 6.]), np.eye(3) * 4,
            dict(dimensions=[4, 6, 8], covariance=(np.eye(3) * 4).tolist(), independent=False),
        )
        np.testing.assert_allclose(covariance, np.eye(3) * 4)

    def test_prediction_is_not_an_observation(self):
        result = run_case(example(), "metricdamage")
        self.assertEqual(result["action_status"], "pending")
        np.testing.assert_allclose(result["covariance"], result["prior_covariance"])
        self.assertFalse(np.allclose(result["predicted_covariance"], result["covariance"]))
        self.assertIsNone(score(example(), result)["observed_risk_reduction_per_meter"])
        self.assertEqual(score(example(), result)["extra_captures"], 0)

    def test_realized_mean_covariance_and_scores_change_without_truth_input(self):
        case = example()
        calls = []

        def observe(case_id, action):
            calls.append((case_id, action))
            return observation()

        result = run_case(case, "metricdamage", observe)
        self.assertEqual(calls, [("example", "lateral")])
        self.assertEqual(result["action_status"], "observed")
        prior_width_error = abs(result["prior_dimensions"][1] - case["truth"][1]) * 100
        self.assertLess(score(case, result)["width_mae_cm"], prior_width_error / 10)
        self.assertGreater(score(case, result)["width_error_improvement_cm"], 1.9)
        self.assertEqual(score(case, result)["motion_cost"], 0.6)
        changed_truth = copy.deepcopy(case)
        changed_truth["truth"] = [9, 9, 9]
        self.assertEqual(result, run_case(changed_truth, "metricdamage", observe))

    def test_failed_action_keeps_prior_and_records_measured_cost(self):
        case = example()
        failed = dict(case_id="example", action="lateral", status="failed", cost=0.2, seconds=2)
        case["observations"] = {"lateral": failed}
        result = run_case(case, "metricdamage")
        self.assertEqual(result["dimensions"], result["prior_dimensions"])
        self.assertEqual(result["covariance"], result["prior_covariance"])
        self.assertEqual(score(case, result)["action_failures"], 1)
        self.assertEqual(score(case, result)["motion_cost"], 0.2)

    def test_passive_does_not_invoke_action_and_wrong_receipt_is_rejected(self):
        def unexpected(*args):
            self.fail("Passive processing must not invoke the observer")

        self.assertEqual(run_case(example(), "passive", unexpected)["action_status"], "passive")
        wrong = observation()
        wrong["action"] = "other-action"
        with self.assertRaisesRegex(ValueError, "does not match"):
            run_case(example(), "metricdamage", lambda *_: wrong)


if __name__ == "__main__":
    unittest.main()
