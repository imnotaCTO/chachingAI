import unittest

import numpy as np

from sgp_engine.modeling import (
    fit_lognormal_params,
    lognormal_correlations,
    lognormal_mean,
    make_psd_correlation,
)


class TestModeling(unittest.TestCase):
    def test_fit_lognormal_params_basic(self) -> None:
        samples = np.array([0.0, 1.0, 2.0, 3.0])
        expected_mu = float(np.mean(np.log1p(samples)))
        expected_sigma = float(np.std(np.log1p(samples), ddof=1))
        mu, sigma = fit_lognormal_params(samples)
        self.assertAlmostEqual(mu, expected_mu)
        self.assertAlmostEqual(sigma, expected_sigma)

    def test_fit_lognormal_params_requires_two(self) -> None:
        with self.assertRaises(ValueError):
            fit_lognormal_params(np.array([]))
        with self.assertRaises(ValueError):
            fit_lognormal_params(np.array([1.0]))

    def test_lognormal_mean(self) -> None:
        mu = 0.2
        sigma = 0.5
        expected = float(np.exp(mu + sigma**2 / 2) - 1)
        self.assertAlmostEqual(lognormal_mean(mu, sigma), expected)

    def test_lognormal_correlations_shape(self) -> None:
        samples = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        corr = lognormal_correlations(samples)
        self.assertEqual(corr.shape, (2, 2))
        np.testing.assert_allclose(np.diag(corr), np.array([1.0, 1.0]), rtol=1e-7, atol=1e-7)

    def test_make_psd_correlation(self) -> None:
        matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
        psd = make_psd_correlation(matrix)
        self.assertEqual(psd.shape, (2, 2))
        np.testing.assert_allclose(psd, psd.T, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(np.diag(psd), np.array([1.0, 1.0]), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
