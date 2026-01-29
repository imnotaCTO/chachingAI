import unittest

import numpy as np

from sgp_engine.odds import expected_value, prob_to_american
from sgp_engine.pipeline import LegSpec, price_parlay_from_samples


class TestPipeline(unittest.TestCase):
    def test_price_parlay_from_samples(self) -> None:
        samples = np.array(
            [
                [10.0, 5.0, 3.0],
                [12.0, 4.0, 2.0],
                [8.0, 6.0, 4.0],
                [15.0, 7.0, 5.0],
                [9.0, 5.0, 3.0],
            ]
        )
        legs = [
            LegSpec(index=0, line=9.5, direction="over", player="Player A", stat="points"),
            LegSpec(index=2, line=2.5, direction="over", player="Player A", stat="assists"),
        ]
        sportsbook_odds = 200.0
        rng = np.random.default_rng(123)
        result = price_parlay_from_samples(
            samples,
            legs,
            sportsbook_odds=sportsbook_odds,
            game="Test Game",
            simulations=5000,
            rng=rng,
        )
        self.assertEqual(len(result.legs), 2)
        self.assertGreater(result.joint_probability, 0.0)
        self.assertLess(result.joint_probability, 1.0)
        expected_ev = expected_value(result.joint_probability, sportsbook_odds)
        self.assertAlmostEqual(result.expected_value, expected_ev)
        clipped = min(max(result.joint_probability, 1e-6), 1 - 1e-6)
        expected_fair = prob_to_american(clipped)
        self.assertAlmostEqual(result.model_fair_odds, expected_fair)

    def test_price_parlay_requires_samples(self) -> None:
        samples = np.array([[1.0, 2.0]])
        legs = [LegSpec(index=0, line=1.0, direction="over")]
        with self.assertRaises(ValueError):
            price_parlay_from_samples(samples, legs, sportsbook_odds=100.0)


if __name__ == "__main__":
    unittest.main()
