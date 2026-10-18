import copy
import unittest
import numpy as np
from finite_data import cases, sensor
from metricdamage.core import run_case


class FiniteInputTests(unittest.TestCase):
    def test_sensor_covariance_and_truth_isolation(self):
        value, covariance = sensor([1, 0.05, 0.05], 0.02, np.random.default_rng(3))
        self.assertTrue(np.all(np.asarray(value) > 0))
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0))
        case = next(cases(1, 17))
        case["anchor_valid"] = True
        altered = copy.deepcopy(case)
        altered["truth"] = [9, 9, 81]
        self.assertEqual(run_case(case, "metricdamage"), run_case(altered, "metricdamage"))
        for action, record in case["observations"].items():
            self.assertEqual((record["case_id"], record["action"]), (case["id"], action))
            self.assertEqual(record["cost_source"], "declared_finite_motion_model")
