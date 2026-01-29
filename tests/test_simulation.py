import unittest

import numpy as np

from sgp_engine.simulation import ParlayLeg, evaluate_legs, simulate_lognormal_copula


class TestSimulation(unittest.TestCase):
    def test_simulate_lognormal_copula_shape(self) -> None:
        mu = np.array([0.1, 0.2])
        sigma = np.array([0.3, 0.4])
        correlation = np.eye(2)
        rng = np.random.default_rng(0)
        sims = 500
        samples = simulate_lognormal_copula(
            mu=mu,
            sigma=sigma,
            correlation=correlation,
            simulations=sims,
            rng=rng,
        )
        self.assertEqual(samples.shape, (sims, 2))

    def test_evaluate_legs(self) -> None:
        samples = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        legs = [
            ParlayLeg(index=0, line=2.0, direction="over"),
            ParlayLeg(index=1, line=5.0, direction="under"),
        ]
        result = evaluate_legs(samples, legs)
        np.testing.assert_allclose(result.leg_probabilities, np.array([2 / 3, 2 / 3]))
        self.assertAlmostEqual(result.joint_probability, 1 / 3)


if __name__ == "__main__":
    unittest.main()
