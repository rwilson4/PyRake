"""Test population mean estimators."""

import itertools
import math
import time
import unittest
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import optimize, stats
from scipy.special import expit, logit

from pyrake.estimation.population import (
    AIPWEstimator,
    IPWEstimator,
    PopulationMean,
    RatioEstimator,
    SAIPWEstimator,
    SampleMean,
    SIPWEstimator,
)


class IPWEstimatorTest(unittest.TestCase):
    """Tests for ``IPWEstimator``."""

    @staticmethod
    def get_data(
        binary: bool = True,
    ) -> tuple[int, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        rng = np.random.default_rng(42)
        population_size = 1_000_000
        sample_size = 1_000
        q = sample_size / population_size
        sigma2 = 1e-6 * q * (1 - q)
        s = q * (1 - q) / sigma2 - 1
        alpha = q * s
        beta = (1 - q) * s
        propensities = rng.beta(alpha, beta, size=sample_size)
        if binary:
            logit_mean_outcomes = rng.normal(size=sample_size)
            outcomes = rng.binomial(
                1, expit(logit_mean_outcomes), size=sample_size
            ).astype(float)
        else:
            outcomes = rng.normal(size=sample_size)

        return population_size, propensities, outcomes

    def test_point_estimate(self) -> None:
        """Test point estimate."""
        population_size, propensities, outcomes = IPWEstimatorTest.get_data()
        estimator = IPWEstimator(
            propensities,
            outcomes,
            population_size,
            estimand_class=PopulationMean,
        )
        expected = 0.4561422941383326

        actual = estimator.point_estimate()
        self.assertAlmostEqual(actual, expected)

    def test_variance(self) -> None:
        """Test variance."""
        population_size, propensities, outcomes = IPWEstimatorTest.get_data()
        estimator = IPWEstimator(
            propensities,
            outcomes,
            population_size,
            estimand_class=PopulationMean,
        )
        expected = 0.0002489382388053698

        actual = estimator.variance()
        self.assertAlmostEqual(actual, expected)

    def test_pvalue(self) -> None:
        """Test p-value calculation."""
        population_size, propensities, outcomes = IPWEstimatorTest.get_data()
        estimator = IPWEstimator(
            propensities,
            outcomes,
            population_size,
            estimand_class=PopulationMean,
        )
        expected = 0.005440680301504909

        actual = estimator.pvalue(null_value=0.50)
        self.assertAlmostEqual(actual, expected)

        # Test one-sided p-values
        actual = estimator.pvalue(null_value=0.50, alternative="less")
        self.assertLess(actual, 0.5)
        self.assertAlmostEqual(actual, 0.5 * expected)

        actual = estimator.pvalue(null_value=0.50, alternative="greater")
        self.assertGreater(actual, 0.5)

    def test_confidence_interval(self) -> None:
        """Test confidence interval."""
        population_size, propensities, outcomes = IPWEstimatorTest.get_data()
        estimator = IPWEstimator(
            propensities,
            outcomes,
            population_size,
            estimand_class=PopulationMean,
        )
        lb_expected = 0.43019016084501915
        ub_expected = 0.4820944274316461

        lb_actual, ub_actual = estimator.confidence_interval(alpha=0.1)
        print(lb_actual, ub_actual)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertAlmostEqual(ub_actual, ub_expected)

        # Test one-sided intervals
        lb_actual, ub_actual = estimator.confidence_interval(
            alpha=0.05, alternative="less"
        )
        print(lb_actual, ub_actual)
        self.assertEqual(lb_actual, -np.inf)
        self.assertAlmostEqual(ub_actual, ub_expected)

        lb_actual, ub_actual = estimator.confidence_interval(
            alpha=0.05, alternative="greater"
        )
        print(lb_actual, ub_actual)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertEqual(ub_actual, np.inf)

    def test_sensitivity_analysis_binary_outcomes(self) -> None:
        """Test sensitivity analysis."""
        population_size, propensities, outcomes = IPWEstimatorTest.get_data()
        estimator = IPWEstimator(
            propensities,
            outcomes,
            population_size,
            estimand_class=PopulationMean,
        )
        gamma = 6

        # Note: even when the outcomes are binary, there is nothing preventing the
        # sensitivity interval for IPW to include values greater than 1. If Gamma times
        # the mean weight is materially larger than 1, it will inflate the estimate.
        # This is an advantage of the SIPWEstimator.
        lb_expected = 0.18648915056978735
        ub_expected = 1.1166549034187236

        lb_actual, ub_actual = estimator.sensitivity_analysis(gamma=gamma)
        print(lb_actual, ub_actual)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertAlmostEqual(ub_actual, ub_expected)

    def test_sensitivity_analysis_continuous_outcomes(self) -> None:
        """Test sensitivity analysis."""
        population_size, propensities, outcomes = IPWEstimatorTest.get_data(
            binary=False
        )
        estimator = IPWEstimator(
            propensities,
            outcomes,
            population_size,
            estimand_class=PopulationMean,
        )
        gamma = 6

        lb_expected = -0.8351175791560294
        ub_expected = 0.7661434943047964

        lb_actual, ub_actual = estimator.sensitivity_analysis(gamma=gamma)
        print(lb_actual, ub_actual)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertAlmostEqual(ub_actual, ub_expected)

    def test_expanded_confidence_interval(self) -> None:
        """Test expanded confidence interval."""
        population_size, propensities, outcomes = IPWEstimatorTest.get_data()
        estimator = IPWEstimator(
            propensities,
            outcomes,
            population_size,
            estimand_class=PopulationMean,
        )
        lb_expected = 0.17587891311249543
        ub_expected = 1.180186853269606

        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.1, gamma=6.0, bootstrap=False
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertAlmostEqual(ub_actual, ub_expected)

        # Test one-sided intervals
        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.05, gamma=6.0, alternative="less", bootstrap=False
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        self.assertEqual(lb_actual, -np.inf)
        self.assertAlmostEqual(ub_actual, ub_expected)

        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.05, gamma=6.0, alternative="greater", bootstrap=False
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertEqual(ub_actual, np.inf)

    def test_plot_sensitivity(self) -> None:
        """Test sensitivity analysis."""
        population_size, propensities, outcomes = IPWEstimatorTest.get_data()
        estimator = IPWEstimator(
            propensities,
            outcomes,
            population_size,
            estimand_class=PopulationMean,
        )
        expected_columns = [
            "Gamma",
            "ECI Lower Bound",
            "Sen Lower Bound",
            "Point Estimate",
            "Sen Upper Bound",
            "ECI Upper Bound",
        ]

        df, ax = estimator.plot_sensitivity(
            gamma_upper=6.0,
            num_points=50,
            alpha=0.10,
            alternative="two-sided",
            ylabel="Incidence",
            bootstrap=False,
            axis_label_size=18,
            tick_label_size=14,
        )
        # plt.show()
        self.assertEqual(set(df.columns), set(expected_columns))
        self.assertEqual(len(df), 50)
        plt.close("all")

    def test_min_ess_bound(self) -> None:
        """Conservative worst-case effective sample size bound over the region."""
        population_size, propensities, outcomes = IPWEstimatorTest.get_data()
        estimator = IPWEstimator(
            propensities,
            outcomes,
            population_size,
            estimand_class=PopulationMean,
        )

        def kish_ess(w: npt.NDArray[np.float64]) -> float:
            return float(np.sum(w)) ** 2 / float(np.sum(w**2))

        # At gamma=1 the region collapses to the nominal weights, so both methods
        # return the exact effective sample size.
        nominal_ess = kish_ess(estimator.weights)
        self.assertAlmostEqual(estimator.min_ess_bound(1.0), nominal_ess)
        self.assertAlmostEqual(
            estimator.min_ess_bound(1.0, method="exact"), nominal_ess
        )

        # Both methods are non-increasing in gamma: a wider region can only lower the
        # worst-case effective sample size.
        gammas = np.linspace(1.0, 6.0, 20)
        for method in ("floor", "exact"):
            bounds = [estimator.min_ess_bound(g, method=method) for g in gammas]
            for earlier, later in itertools.pairwise(bounds):
                self.assertLessEqual(later, earlier + 1e-9)

        # Both must be genuine lower bounds: no weight vector in the region has a
        # smaller effective sample size. The exact method also dominates the floor.
        rng = np.random.default_rng(0)
        for gamma in (1.5, 3.0, 6.0):
            wl, wu = estimator.estimand.sensitivity_region(gamma)
            floor = estimator.min_ess_bound(gamma, method="floor")
            exact = estimator.min_ess_bound(gamma, method="exact")
            self.assertGreaterEqual(exact, floor - 1e-9)
            candidates = [wl, wu]
            for _ in range(200):
                u = rng.uniform(size=wl.shape)
                candidates.append(wl + u * (wu - wl))
            for w in candidates:
                self.assertLessEqual(exact, kish_ess(w) + 1e-9)
                self.assertLessEqual(floor, kish_ess(w) + 1e-9)

        with self.assertRaises(ValueError):
            estimator.min_ess_bound(0.5)
        invalid_method: Any = "bogus"
        with self.assertRaises(ValueError):
            estimator.min_ess_bound(2.0, method=invalid_method)

    def test_min_ess_bound_exact_matches_brute_force(self) -> None:
        """The exact method equals the true minimum over all box vertices."""
        propensities = np.array([0.2, 0.5, 0.8, 0.3, 0.9, 0.4, 0.6, 0.7, 0.25, 0.55])
        outcomes = np.arange(propensities.shape[0], dtype=float)
        estimator = IPWEstimator(
            propensities,
            outcomes,
            population_size=100,
            estimand_class=PopulationMean,
        )

        def kish_ess(w: npt.NDArray[np.float64]) -> float:
            return float(np.sum(w)) ** 2 / float(np.sum(w**2))

        for gamma in (2.0, 4.0, 6.0):
            wl, wu = estimator.estimand.sensitivity_region(gamma)
            # Enumerate all 2^n vertices of the box.
            best = math.inf
            for mask in itertools.product((0, 1), repeat=wl.shape[0]):
                w = np.where(np.array(mask, dtype=bool), wu, wl)
                best = min(best, kish_ess(w))
            self.assertAlmostEqual(estimator.min_ess_bound(gamma, method="exact"), best)

    def test_plot_min_ess(self) -> None:
        """Test plotting the worst-case effective sample size."""
        population_size, propensities, outcomes = IPWEstimatorTest.get_data()
        estimator = IPWEstimator(
            propensities,
            outcomes,
            population_size,
            estimand_class=PopulationMean,
        )

        for method in ("floor", "exact"):
            df, ax = estimator.plot_min_ess(
                gamma_upper=6.0,
                num_points=50,
                method=method,
                reference_ess=100.0,
                axis_label_size=18,
                tick_label_size=14,
            )
            # plt.show()
            self.assertEqual(set(df.columns), {"Gamma", "Min ESS Bound"})
            self.assertEqual(len(df), 50)
            # The curve is non-increasing in gamma.
            self.assertTrue(np.all(np.diff(df["Min ESS Bound"].to_numpy()) <= 1e-9))
            plt.close("all")


class AIPWEstimatorTest(unittest.TestCase):
    """Tests for ``AIPWEstimator``."""

    @staticmethod
    def get_data(
        binary: bool = True,
    ) -> tuple[
        int,
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        float,
    ]:
        rng = np.random.default_rng(42)
        population_size = 1_000_000
        sample_size = 1_000
        q = sample_size / population_size
        sigma2 = 1e-6 * q * (1 - q)
        s = q * (1 - q) / sigma2 - 1
        alpha = q * s
        beta = (1 - q) * s
        propensities = rng.beta(alpha, beta, size=sample_size)
        if binary:
            logit_mean_outcomes = rng.normal(size=sample_size)
            outcomes = rng.binomial(
                1, expit(logit_mean_outcomes), size=sample_size
            ).astype(float)

            predicted_outcomes = expit(
                logit(np.where(outcomes == 1, 0.99, 0.01))
                + 0.01 * rng.normal(size=sample_size)
            )
            mean_predicted_outcome = 0.5
        else:
            outcomes = rng.normal(size=sample_size)
            predicted_outcomes = outcomes + 0.01 * rng.normal(size=sample_size)
            mean_predicted_outcome = 0.0

        return (
            population_size,
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        )

    def test_point_estimate(self) -> None:
        """Test point estimate."""
        (
            population_size,
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = AIPWEstimatorTest.get_data()
        estimator = AIPWEstimator(
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
            population_size,
        )
        expected = 0.49912034525845983

        actual = estimator.point_estimate()
        self.assertAlmostEqual(actual, expected)

    def test_variance(self) -> None:
        """Test variance."""
        (
            population_size,
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = AIPWEstimatorTest.get_data()
        estimator = AIPWEstimator(
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
            population_size,
        )
        expected = 9.951767238492971e-08

        actual = estimator.variance()
        # Preserve original ``pytest.approx`` ``rel=1e-6`` semantics: the
        # AIPW variance is O(1e-8), so ``assertAlmostEqual``'s default
        # ``places=7`` (~5e-8 absolute) would be looser than the expected
        # value itself.
        self.assertAlmostEqual(actual, expected, delta=expected * 1e-6)

    def test_pvalue(self) -> None:
        """Test p-value calculation."""
        (
            population_size,
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = AIPWEstimatorTest.get_data()
        estimator = AIPWEstimator(
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
            population_size,
        )
        expected = 0.005296167612649354

        actual = estimator.pvalue(null_value=0.50)
        self.assertAlmostEqual(actual, expected)

        # Test one-sided p-values
        actual = estimator.pvalue(null_value=0.50, alternative="less")
        self.assertLess(actual, 0.5)
        self.assertAlmostEqual(actual, 0.5 * expected)

        actual = estimator.pvalue(null_value=0.50, alternative="greater")
        self.assertGreater(actual, 0.5)

    def test_confidence_interval(self) -> None:
        """Test confidence interval."""
        (
            population_size,
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = AIPWEstimatorTest.get_data()
        estimator = AIPWEstimator(
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
            population_size,
        )
        lb_expected = 0.4986014527964912
        ub_expected = 0.49963923772042845

        lb_actual, ub_actual = estimator.confidence_interval(alpha=0.1)
        print(lb_actual, ub_actual)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertAlmostEqual(ub_actual, ub_expected)

        # Test one-sided intervals
        lb_actual, ub_actual = estimator.confidence_interval(
            alpha=0.05, alternative="less"
        )
        print(lb_actual, ub_actual)
        self.assertEqual(lb_actual, -np.inf)
        self.assertAlmostEqual(ub_actual, ub_expected)

        lb_actual, ub_actual = estimator.confidence_interval(
            alpha=0.05, alternative="greater"
        )
        print(lb_actual, ub_actual)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertEqual(ub_actual, np.inf)

    def test_sensitivity_analysis_binary_outcomes(self) -> None:
        """Test sensitivity analysis."""
        (
            population_size,
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = AIPWEstimatorTest.get_data()
        estimator = AIPWEstimator(
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
            population_size,
        )
        gamma = 6

        lb_expected = 0.4885436500519286
        ub_expected = 0.5089432814518619

        lb_actual, ub_actual = estimator.sensitivity_analysis(gamma=gamma)
        print(lb_actual, ub_actual)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertAlmostEqual(ub_actual, ub_expected)

    def test_sensitivity_analysis_continuous_outcomes(self) -> None:
        """Test sensitivity analysis."""
        (
            population_size,
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = AIPWEstimatorTest.get_data(binary=False)
        estimator = AIPWEstimator(
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
            population_size,
        )
        gamma = 6

        lb_expected = -0.008871225820982506
        ub_expected = 0.007537264300121401

        lb_actual, ub_actual = estimator.sensitivity_analysis(gamma=gamma)
        print(lb_actual, ub_actual)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertAlmostEqual(ub_actual, ub_expected)

    def test_expanded_confidence_interval(self) -> None:
        """Test expanded confidence interval."""
        (
            population_size,
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = AIPWEstimatorTest.get_data()
        estimator = AIPWEstimator(
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
            population_size,
        )
        lb_expected = 0.4878022068837456
        ub_expected = 0.5096846983418535

        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.1, gamma=6.0, bootstrap=False
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertAlmostEqual(ub_actual, ub_expected)

        # Test one-sided intervals
        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.05, gamma=6.0, alternative="less", bootstrap=False
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        self.assertEqual(lb_actual, -np.inf)
        self.assertAlmostEqual(ub_actual, ub_expected)

        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.05, gamma=6.0, alternative="greater", bootstrap=False
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertEqual(ub_actual, np.inf)

    def test_bootstrap_preserves_mean_prediction(self) -> None:
        """Regression: the bootstrap path rebuilds the estimator via
        ``resample``; confirm ``mean_predicted_outcome`` survives, so the
        bootstrap interval and adjusted p-value are on the full (not residual)
        scale. If the offset leaked, the bootstrap interval would sit
        ~``mean_predicted_outcome`` (0.5 here) below the heuristic one."""
        (
            population_size,
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = AIPWEstimatorTest.get_data()
        estimator = AIPWEstimator(
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
            population_size,
        )
        heur_lb, heur_ub = estimator.expanded_confidence_interval(
            alpha=0.1, gamma=6.0, bootstrap=False
        )
        boot_lb, boot_ub = estimator.expanded_confidence_interval(
            alpha=0.1, gamma=6.0, bootstrap=True, B=500, seed=42
        )
        self.assertAlmostEqual(boot_lb, heur_lb, delta=0.1)
        self.assertAlmostEqual(boot_ub, heur_ub, delta=0.1)
        # A two-sided adjusted p-value at the full-scale point estimate is not
        # rejected; it would be near zero on the residual scale.
        pe = estimator.point_estimate()
        boot_p = estimator.adjusted_pvalue(
            null_value=pe, gamma=2.0, bootstrap=True, B=500, seed=42
        )
        self.assertGreater(boot_p, 0.5)

    def test_plot_sensitivity(self) -> None:
        """Test sensitivity analysis."""
        (
            population_size,
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = AIPWEstimatorTest.get_data()
        estimator = AIPWEstimator(
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
            population_size,
        )
        expected_columns = [
            "Gamma",
            "ECI Lower Bound",
            "Sen Lower Bound",
            "Point Estimate",
            "Sen Upper Bound",
            "ECI Upper Bound",
        ]

        df, ax = estimator.plot_sensitivity(
            gamma_upper=6.0,
            num_points=50,
            alpha=0.10,
            alternative="two-sided",
            ylabel="Incidence",
            bootstrap=False,
            axis_label_size=18,
            tick_label_size=14,
            ytick_format=".02%",
        )
        # plt.show()
        self.assertEqual(set(df.columns), set(expected_columns))
        self.assertEqual(len(df), 50)
        plt.close("all")


class SIPWEstimatorTest(unittest.TestCase):
    """Tests for ``SIPWEstimator``."""

    @staticmethod
    def get_data(
        binary: bool = True,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        population_size, propensities, outcomes = IPWEstimatorTest.get_data(
            binary=binary
        )
        return propensities, outcomes

    def test_point_estimate(self) -> None:
        """Test point estimate."""
        propensities, outcomes = SIPWEstimatorTest.get_data()
        estimator = SIPWEstimator(propensities, outcomes)
        expected = 0.4560066389336293

        actual = estimator.point_estimate()
        self.assertAlmostEqual(actual, expected)

        # SIPW is invariant to changing the scale of the weights
        estimator = SIPWEstimator(2.0 * propensities, outcomes)
        actual = estimator.point_estimate()
        self.assertAlmostEqual(actual, expected)

        estimator = SIPWEstimator(propensities, outcomes, estimand_class=SampleMean)
        expected = float(np.mean(outcomes))

        actual = estimator.point_estimate()
        self.assertAlmostEqual(actual, expected)

    def test_variance(self) -> None:
        """Test variance."""
        propensities, outcomes = SIPWEstimatorTest.get_data()
        estimator = SIPWEstimator(propensities, outcomes)
        expected = 0.0002483163797360267

        actual = estimator.variance()
        self.assertAlmostEqual(actual, expected)

        # SIPW is invariant to changing the scale of the weights
        estimator = SIPWEstimator(2.0 * propensities, outcomes)
        actual = estimator.variance()
        self.assertAlmostEqual(actual, expected)

    def test_pvalue(self) -> None:
        """Test p-value calculation."""
        propensities, outcomes = SIPWEstimatorTest.get_data()
        estimator = SIPWEstimator(propensities, outcomes)
        expected = 0.005241558051856385

        actual = estimator.pvalue(null_value=0.50)
        self.assertAlmostEqual(actual, expected)

        # Test one-sided p-values
        actual = estimator.pvalue(null_value=0.50, alternative="less")
        self.assertLess(actual, 0.5)
        self.assertAlmostEqual(actual, 0.5 * expected)

        actual = estimator.pvalue(null_value=0.50, alternative="greater")
        self.assertGreater(actual, 0.5)

        # SIPW is invariant to changing the scale of the weights
        estimator = SIPWEstimator(2.0 * propensities, outcomes)
        actual = estimator.pvalue(null_value=0.50)
        self.assertAlmostEqual(actual, expected)

    def test_confidence_interval(self) -> None:
        """Test confidence interval."""
        propensities, outcomes = SIPWEstimatorTest.get_data()
        estimator = SIPWEstimator(propensities, outcomes)
        lb_expected = 0.43008694071510295
        ub_expected = 0.48192633715215566

        lb_actual, ub_actual = estimator.confidence_interval(alpha=0.1)
        print(lb_actual, ub_actual)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertAlmostEqual(ub_actual, ub_expected)

        # Test one-sided intervals
        lb_actual, ub_actual = estimator.confidence_interval(
            alpha=0.05, alternative="less"
        )
        print(lb_actual, ub_actual)
        self.assertEqual(lb_actual, -np.inf)
        self.assertAlmostEqual(ub_actual, ub_expected)

        lb_actual, ub_actual = estimator.confidence_interval(
            alpha=0.05, alternative="greater"
        )
        print(lb_actual, ub_actual)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertEqual(ub_actual, np.inf)

        # SIPW is invariant to changing the scale of the weights
        estimator = SIPWEstimator(2.0 * propensities, outcomes)
        lb_actual, ub_actual = estimator.confidence_interval(alpha=0.1)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertAlmostEqual(ub_actual, ub_expected)

    def test_sensitivity_analysis_binary_outcomes(self) -> None:
        """Test sensitivity analysis."""
        propensities, outcomes = SIPWEstimatorTest.get_data()
        estimator = SIPWEstimator(propensities, outcomes)
        gamma = 6

        weights = np.ones_like(propensities) / propensities
        wl = weights / math.sqrt(gamma) + (1.0 - 1.0 / math.sqrt(gamma))
        wu = weights * math.sqrt(gamma) - (math.sqrt(gamma) - 1.0)
        G = np.vstack([-np.eye(len(propensities)), np.eye(len(propensities))])
        h = np.concatenate([-wl, wu])

        lb_res = optimize.linprog(
            c=np.append(outcomes, 0),
            A_ub=np.hstack([G, -h.reshape(-1, 1)]),
            b_ub=np.zeros_like(h),
            A_eq=np.append(np.ones_like(outcomes), 0).reshape(1, -1),
            b_eq=np.array([1]),
        )

        ub_res = optimize.linprog(
            c=np.append(-outcomes, 0),
            A_ub=np.hstack([G, -h.reshape(-1, 1)]),
            b_ub=np.zeros_like(h),
            A_eq=np.append(np.ones_like(outcomes), 0).reshape(1, -1),
            b_eq=np.array([1]),
        )

        lb_expected = lb_res.fun
        ub_expected = -ub_res.fun

        lb_actual, ub_actual = estimator.sensitivity_analysis(gamma=gamma)
        print(lb_actual, ub_actual)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertAlmostEqual(ub_actual, ub_expected)

    def test_sensitivity_analysis_continuous_outcomes(self) -> None:
        """Test sensitivity analysis."""
        propensities, outcomes = SIPWEstimatorTest.get_data(binary=False)
        estimator = SIPWEstimator(propensities, outcomes)
        gamma = 6

        weights = np.ones_like(propensities) / propensities
        wl = weights / math.sqrt(gamma) + (1.0 - 1.0 / math.sqrt(gamma))
        wu = weights * math.sqrt(gamma) - (math.sqrt(gamma) - 1.0)
        G = np.vstack([-np.eye(len(propensities)), np.eye(len(propensities))])
        h = np.concatenate([-wl, wu])

        lb_res = optimize.linprog(
            c=np.append(outcomes, 0),
            A_ub=np.hstack([G, -h.reshape(-1, 1)]),
            b_ub=np.zeros_like(h),
            A_eq=np.append(np.ones_like(outcomes), 0).reshape(1, -1),
            b_eq=np.array([1]),
        )

        ub_res = optimize.linprog(
            c=np.append(-outcomes, 0),
            A_ub=np.hstack([G, -h.reshape(-1, 1)]),
            b_ub=np.zeros_like(h),
            A_eq=np.append(np.ones_like(outcomes), 0).reshape(1, -1),
            b_eq=np.array([1]),
        )

        lb_expected = lb_res.fun
        ub_expected = -ub_res.fun

        lb_actual, ub_actual = estimator.sensitivity_analysis(gamma=gamma)
        print(lb_actual, ub_actual)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertAlmostEqual(ub_actual, ub_expected)

    def test_adjusted_pvalue(self) -> None:
        """Test adjusted p-value calculation."""
        propensities, outcomes = SIPWEstimatorTest.get_data()
        estimator = SIPWEstimator(propensities, outcomes)
        print("Point Estimate:", estimator.point_estimate())

        print("P-values")
        # gamma=1 reduces to the standard p-value
        std_pvalue = estimator.pvalue(null_value=0.50)
        actual = estimator.adjusted_pvalue(null_value=0.50, gamma=1.0)
        self.assertAlmostEqual(actual, std_pvalue)
        print(f"Unadjusted: {actual:.05f}")

        # For gamma > 1, null values inside the sensitivity interval inflate the
        # p-value toward 1 (adjusted >= standard).
        st = time.time()
        bootstrap = estimator.adjusted_pvalue(null_value=0.50, gamma=1.1, seed=42)
        print(
            f"Bootstrap:  {bootstrap:.05f} (calculated in {1000 * (time.time() - st):.03f} ms)"
        )
        self.assertGreaterEqual(bootstrap, std_pvalue)

        st = time.time()
        heuristic = estimator.adjusted_pvalue(
            null_value=0.50, gamma=1.1, bootstrap=False
        )
        print(
            f"Heuristic:  {heuristic:.05f} (calculated in {1000 * (time.time() - st):.03f} ms)"
        )
        self.assertGreaterEqual(heuristic, std_pvalue)
        self.assertAlmostEqual(heuristic, bootstrap, delta=0.02)

        # One-sided tests with null values near the bootstrap sensitivity-bound
        # medians return adjusted p-values near 0.5.
        actual = estimator.adjusted_pvalue(
            null_value=0.295, gamma=2.0, alternative="greater", seed=42
        )
        self.assertAlmostEqual(actual, 0.491)

        actual = estimator.adjusted_pvalue(
            null_value=0.626, gamma=2.0, alternative="less", seed=42
        )
        self.assertAlmostEqual(actual, 0.505)

        # Two-sided combines both one-sided adjusted p-values.
        actual = estimator.adjusted_pvalue(null_value=0.295, gamma=2.0, seed=42)
        self.assertAlmostEqual(actual, 0.982)

    def test_gamma_star(self) -> None:
        """Test gamma_star method."""
        propensities, outcomes = SIPWEstimatorTest.get_data()
        estimator = SIPWEstimator(propensities, outcomes)
        null_value = 0.50

        # Standard p-value is significant, so gamma_star > 1.
        self.assertLess(estimator.pvalue(null_value=null_value), 0.05)

        # Non-bootstrap path is deterministic.
        gs = estimator.gamma_star(null_value=null_value, bootstrap=False)
        self.assertGreater(gs, 1.0)

        # Adjusted p-value at gamma_star is not significant (>= alpha).
        self.assertGreaterEqual(
            estimator.adjusted_pvalue(null_value=null_value, gamma=gs, bootstrap=False),
            0.05,
        )

        # Adjusted p-value just below gamma_star is still significant (< alpha).
        if gs > 1.0 + 2e-3:
            self.assertLess(
                estimator.adjusted_pvalue(
                    null_value=null_value, gamma=gs - 2e-3, bootstrap=False
                ),
                0.05,
            )

        # When already not significant at gamma=1, gamma_star returns 1.0.
        gs_not_sig = estimator.gamma_star(
            null_value=estimator.point_estimate(), bootstrap=False
        )
        self.assertEqual(gs_not_sig, 1.0)

        # A tiny gamma_upper triggers the exponential search phase but still
        # finds the correct crossing (gamma_star is the same regardless of
        # the initial gamma_upper).
        gs_small_upper = estimator.gamma_star(
            null_value=null_value, bootstrap=False, gamma_upper=1.001
        )
        self.assertAlmostEqual(gs_small_upper, gs, delta=1e-2)

        # One-sided tests.
        # H1: theta < 0.5 is significant (point estimate < null), so gamma_star > 1.
        gs_less = estimator.gamma_star(
            null_value=null_value, alternative="less", bootstrap=False
        )
        self.assertGreater(gs_less, 1.0)

        # H1: theta > 0.5 is not significant (point estimate < null), so gamma_star = 1.
        gs_greater = estimator.gamma_star(
            null_value=null_value, alternative="greater", bootstrap=False
        )
        self.assertEqual(gs_greater, 1.0)

        # Bootstrap path: gamma_star > 1 and adjusted p-value at gamma_star >= alpha.
        gs_boot = estimator.gamma_star(null_value=null_value, B=500, seed=42, tol=0.05)
        self.assertGreater(gs_boot, 1.0)
        self.assertGreaterEqual(
            estimator.adjusted_pvalue(
                null_value=null_value, gamma=gs_boot, B=500, seed=42
            ),
            0.05,
        )

        # A more lenient alpha is harder to reach, so it requires a larger gamma_star.
        # (adjusted_pvalue=0.01 is crossed at a smaller gamma than adjusted_pvalue=0.10)
        gs_strict = estimator.gamma_star(
            null_value=null_value, alpha=0.01, bootstrap=False
        )
        gs_lenient = estimator.gamma_star(
            null_value=null_value, alpha=0.10, bootstrap=False
        )
        self.assertGreaterEqual(gs_lenient, gs_strict)

    def test_expanded_confidence_interval(self) -> None:
        """Test expanded confidence interval."""
        propensities, outcomes = SIPWEstimatorTest.get_data()
        estimator = SIPWEstimator(propensities, outcomes)
        lb_expected = 0.11176031608287877
        ub_expected = 0.8474751777705286

        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.1, gamma=6.0, seed=42
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertAlmostEqual(ub_actual, ub_expected)

        # Test heuristic
        start_time = time.time()
        lb_heuristic, ub_heuristic = estimator.expanded_confidence_interval(
            alpha=0.1, gamma=6.0, bootstrap=False
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_heuristic, ub_heuristic)
        self.assertAlmostEqual(lb_heuristic, lb_actual, delta=0.001)
        self.assertAlmostEqual(ub_heuristic, ub_actual, delta=0.001)

        # Test one-sided intervals
        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.05, gamma=6.0, alternative="less", seed=42
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        self.assertEqual(lb_actual, -np.inf)
        self.assertAlmostEqual(ub_actual, ub_expected)

        # Test heuristic
        start_time = time.time()
        lb_heuristic, ub_heuristic = estimator.expanded_confidence_interval(
            alpha=0.05, gamma=6.0, bootstrap=False, alternative="less"
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_heuristic, ub_heuristic)
        self.assertEqual(lb_heuristic, -np.inf)
        self.assertAlmostEqual(ub_heuristic, ub_actual, delta=0.001)

        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.05, gamma=6.0, alternative="greater", seed=42
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertEqual(ub_actual, np.inf)

        # Test heuristic
        start_time = time.time()
        lb_heuristic, ub_heuristic = estimator.expanded_confidence_interval(
            alpha=0.05, gamma=6.0, bootstrap=False, alternative="greater"
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_heuristic, ub_heuristic)
        self.assertAlmostEqual(lb_heuristic, lb_actual, delta=0.001)
        self.assertEqual(ub_heuristic, np.inf)

    def test_plot_sensitivity(self) -> None:
        """Test sensitivity analysis."""
        propensities, outcomes = SIPWEstimatorTest.get_data()
        estimator = SIPWEstimator(propensities, outcomes)
        expected_columns = [
            "Gamma",
            "ECI Lower Bound",
            "Sen Lower Bound",
            "Point Estimate",
            "Sen Upper Bound",
            "ECI Upper Bound",
        ]

        st = time.time()
        df, ax = estimator.plot_sensitivity(
            gamma_upper=6.0,
            num_points=50,
            alpha=0.10,
            alternative="two-sided",
            ylabel="Incidence",
            bootstrap=False,
            axis_label_size=18,
            tick_label_size=14,
        )
        print(f"Figure generated in {time.time() - st:.03f} sec")
        # plt.show()
        self.assertEqual(set(df.columns), set(expected_columns))
        self.assertEqual(len(df), 50)
        plt.close("all")


class SAIPWEstimatorTest(unittest.TestCase):
    """Tests for ``SAIPWEstimator``."""

    @staticmethod
    def get_data(
        binary: bool = True,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        float,
    ]:
        (
            population_size,
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = AIPWEstimatorTest.get_data(binary=binary)
        return (
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        )

    def test_point_estimate(self) -> None:
        """Test point estimate."""
        (
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = SAIPWEstimatorTest.get_data()
        estimator = SAIPWEstimator(
            propensities, outcomes, predicted_outcomes, mean_predicted_outcome
        )
        expected = 0.4991206068648611

        actual = estimator.point_estimate()
        self.assertAlmostEqual(actual, expected)

        # SAIPW is invariant to changing the scale of the weights
        estimator = SAIPWEstimator(
            2.0 * propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        )
        actual = estimator.point_estimate()
        self.assertAlmostEqual(actual, expected)

        estimator = SAIPWEstimator(
            propensities,
            outcomes,
            predicted_outcomes,
            float(np.mean(predicted_outcomes)),
            estimand_class=SampleMean,
        )
        expected = float(np.mean(outcomes))

        actual = estimator.point_estimate()
        self.assertAlmostEqual(actual, expected)

    def test_variance(self) -> None:
        """Test variance."""
        (
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = SAIPWEstimatorTest.get_data()
        estimator = SAIPWEstimator(
            propensities, outcomes, predicted_outcomes, mean_predicted_outcome
        )
        expected = 9.935866236133295e-08

        actual = estimator.variance()
        # Preserve original ``pytest.approx`` ``rel=1e-6`` semantics: the
        # SAIPW variance is O(1e-8), so ``assertAlmostEqual``'s default
        # ``places=7`` (~5e-8 absolute) would be looser than the expected
        # value itself.
        self.assertAlmostEqual(actual, expected, delta=expected * 1e-6)

        # SAIPW is invariant to changing the scale of the weights
        estimator = SAIPWEstimator(
            2.0 * propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        )
        actual = estimator.variance()
        self.assertAlmostEqual(actual, expected, delta=expected * 1e-6)

    def test_pvalue(self) -> None:
        """Test p-value calculation."""
        (
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = SAIPWEstimatorTest.get_data()
        estimator = SAIPWEstimator(
            propensities, outcomes, predicted_outcomes, mean_predicted_outcome
        )
        expected = 0.005273314664915336

        actual = estimator.pvalue(null_value=0.50)
        self.assertAlmostEqual(actual, expected)

        # Test one-sided p-values
        actual = estimator.pvalue(null_value=0.50, alternative="less")
        self.assertLess(actual, 0.5)
        self.assertAlmostEqual(actual, 0.5 * expected)

        actual = estimator.pvalue(null_value=0.50, alternative="greater")
        self.assertGreater(actual, 0.5)

        # SAIPW is invariant to changing the scale of the weights
        estimator = SAIPWEstimator(
            2.0 * propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        )
        actual = estimator.pvalue(null_value=0.50)
        self.assertAlmostEqual(actual, expected)

    def test_confidence_interval(self) -> None:
        """Test confidence interval."""
        (
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = SAIPWEstimatorTest.get_data()
        estimator = SAIPWEstimator(
            propensities, outcomes, predicted_outcomes, mean_predicted_outcome
        )
        lb_expected = 0.49860212911359364
        ub_expected = 0.49963908461612855

        lb_actual, ub_actual = estimator.confidence_interval(alpha=0.1)
        print(lb_actual, ub_actual)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertAlmostEqual(ub_actual, ub_expected)

        # Test one-sided intervals
        lb_actual, ub_actual = estimator.confidence_interval(
            alpha=0.05, alternative="less"
        )
        print(lb_actual, ub_actual)
        self.assertEqual(lb_actual, -np.inf)
        self.assertAlmostEqual(ub_actual, ub_expected)

        lb_actual, ub_actual = estimator.confidence_interval(
            alpha=0.05, alternative="greater"
        )
        print(lb_actual, ub_actual)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertEqual(ub_actual, np.inf)

        # SAIPW is invariant to changing the scale of the weights
        estimator = SAIPWEstimator(
            2.0 * propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        )
        lb_actual, ub_actual = estimator.confidence_interval(alpha=0.1)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertAlmostEqual(ub_actual, ub_expected)

    def test_sensitivity_analysis_binary_outcomes(self) -> None:
        """Test sensitivity analysis."""
        (
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = SAIPWEstimatorTest.get_data()
        estimator = SAIPWEstimator(
            propensities, outcomes, predicted_outcomes, mean_predicted_outcome
        )
        gamma = 6

        weights = np.ones_like(propensities) / propensities
        wl = weights / math.sqrt(gamma) + (1.0 - 1.0 / math.sqrt(gamma))
        wu = weights * math.sqrt(gamma) - (math.sqrt(gamma) - 1.0)
        G = np.vstack([-np.eye(len(propensities)), np.eye(len(propensities))])
        h = np.concatenate([-wl, wu])

        lb_res = optimize.linprog(
            c=np.append(outcomes - predicted_outcomes, 0),
            A_ub=np.hstack([G, -h.reshape(-1, 1)]),
            b_ub=np.zeros_like(h),
            A_eq=np.append(np.ones_like(outcomes), 0).reshape(1, -1),
            b_eq=np.array([1]),
        )

        ub_res = optimize.linprog(
            c=np.append(predicted_outcomes - outcomes, 0),
            A_ub=np.hstack([G, -h.reshape(-1, 1)]),
            b_ub=np.zeros_like(h),
            A_eq=np.append(np.ones_like(outcomes), 0).reshape(1, -1),
            b_eq=np.array([1]),
        )

        lb_expected = mean_predicted_outcome + lb_res.fun
        ub_expected = mean_predicted_outcome - ub_res.fun

        lb_actual, ub_actual = estimator.sensitivity_analysis(gamma=gamma)
        print(lb_actual, ub_actual)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertAlmostEqual(ub_actual, ub_expected)

    def test_sensitivity_analysis_continuous_outcomes(self) -> None:
        """Test sensitivity analysis."""
        (
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = SAIPWEstimatorTest.get_data()
        estimator = SAIPWEstimator(
            propensities, outcomes, predicted_outcomes, mean_predicted_outcome
        )
        gamma = 6

        weights = np.ones_like(propensities) / propensities
        wl = weights / math.sqrt(gamma) + (1.0 - 1.0 / math.sqrt(gamma))
        wu = weights * math.sqrt(gamma) - (math.sqrt(gamma) - 1.0)
        G = np.vstack([-np.eye(len(propensities)), np.eye(len(propensities))])
        h = np.concatenate([-wl, wu])

        lb_res = optimize.linprog(
            c=np.append(outcomes - predicted_outcomes, 0),
            A_ub=np.hstack([G, -h.reshape(-1, 1)]),
            b_ub=np.zeros_like(h),
            A_eq=np.append(np.ones_like(outcomes), 0).reshape(1, -1),
            b_eq=np.array([1]),
        )

        ub_res = optimize.linprog(
            c=np.append(predicted_outcomes - outcomes, 0),
            A_ub=np.hstack([G, -h.reshape(-1, 1)]),
            b_ub=np.zeros_like(h),
            A_eq=np.append(np.ones_like(outcomes), 0).reshape(1, -1),
            b_eq=np.array([1]),
        )

        lb_expected = mean_predicted_outcome + lb_res.fun
        ub_expected = mean_predicted_outcome - ub_res.fun

        lb_actual, ub_actual = estimator.sensitivity_analysis(gamma=gamma)
        print(lb_actual, ub_actual)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertAlmostEqual(ub_actual, ub_expected)

    def test_expanded_confidence_interval(self) -> None:
        """Test expanded confidence interval."""
        (
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = SAIPWEstimatorTest.get_data()
        estimator = SAIPWEstimator(
            propensities, outcomes, predicted_outcomes, mean_predicted_outcome
        )
        lb_expected = 0.49223076029818014
        ub_expected = 0.5069680562955309

        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.1, gamma=6.0, bootstrap=False
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertAlmostEqual(ub_actual, ub_expected)

        # Test one-sided intervals
        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.05, gamma=6.0, alternative="less", bootstrap=False
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        self.assertEqual(lb_actual, -np.inf)
        self.assertAlmostEqual(ub_actual, ub_expected)

        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.05, gamma=6.0, alternative="greater", bootstrap=False
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        self.assertAlmostEqual(lb_actual, lb_expected)
        self.assertEqual(ub_actual, np.inf)

    def test_bootstrap_preserves_mean_prediction(self) -> None:
        """Regression: confirm the bootstrap path preserves
        ``mean_predicted_outcome`` after rebuilding via ``resample``, so the
        interval and adjusted p-value stay on the full (not residual) scale."""
        (
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = SAIPWEstimatorTest.get_data()
        estimator = SAIPWEstimator(
            propensities, outcomes, predicted_outcomes, mean_predicted_outcome
        )
        heur_lb, heur_ub = estimator.expanded_confidence_interval(
            alpha=0.1, gamma=6.0, bootstrap=False
        )
        boot_lb, boot_ub = estimator.expanded_confidence_interval(
            alpha=0.1, gamma=6.0, bootstrap=True, B=500, seed=42
        )
        self.assertAlmostEqual(boot_lb, heur_lb, delta=0.1)
        self.assertAlmostEqual(boot_ub, heur_ub, delta=0.1)
        pe = estimator.point_estimate()
        boot_p = estimator.adjusted_pvalue(
            null_value=pe, gamma=2.0, bootstrap=True, B=500, seed=42
        )
        self.assertGreater(boot_p, 0.5)

    def test_plot_sensitivity(self) -> None:
        """Test sensitivity analysis."""
        (
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = SAIPWEstimatorTest.get_data()
        estimator = SAIPWEstimator(
            propensities, outcomes, predicted_outcomes, mean_predicted_outcome
        )
        expected_columns = [
            "Gamma",
            "ECI Lower Bound",
            "Sen Lower Bound",
            "Point Estimate",
            "Sen Upper Bound",
            "ECI Upper Bound",
        ]

        df, ax = estimator.plot_sensitivity(
            gamma_upper=6.0,
            num_points=50,
            alpha=0.10,
            alternative="two-sided",
            ylabel="Incidence",
            bootstrap=False,
            axis_label_size=18,
            tick_label_size=14,
            ytick_format=".02%",
        )
        # plt.show()
        self.assertEqual(set(df.columns), set(expected_columns))
        self.assertEqual(len(df), 50)
        plt.close("all")


class RatioEstimatorTest(unittest.TestCase):
    """Tests for ``RatioEstimator``."""

    @staticmethod
    def generate_data_binary(
        n: int = 100, seed: int = 42
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Generate synthetic data for binary numerator/denominator outcomes."""
        rng = np.random.default_rng(seed)
        propensity_scores = rng.uniform(0.2, 0.8, size=n)
        numerator_outcomes = rng.binomial(1, 0.4, size=n)
        denominator_outcomes = rng.binomial(1, 0.8, size=n)
        return propensity_scores, numerator_outcomes, denominator_outcomes

    @staticmethod
    def generate_data_continuous(n: int = 100, seed: int = 24) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Generate synthetic data for continuous numerator/denominator outcomes."""
        rng = np.random.default_rng(seed)
        propensity_scores = rng.uniform(0.2, 0.9, size=n)
        numerator_outcomes = rng.normal(2.0, 1.0, size=n)
        denominator_outcomes = rng.normal(4.0, 2.0, size=n)
        return propensity_scores, numerator_outcomes, denominator_outcomes

    def test_point_estimate_binary(self) -> None:
        """Test that the point estimate is computed as expected for binary outcomes."""
        ps, num, den = self.generate_data_binary()
        estimator = RatioEstimator(
            propensity_scores=ps,
            numerator_outcomes=num,
            denominator_outcomes=den,
        )
        # Compute stabilized means directly
        w = 1 / ps
        x_hat = np.sum(w * num) / np.sum(w)
        y_hat = np.sum(w * den) / np.sum(w)
        expected = x_hat / y_hat
        self.assertAlmostEqual(estimator.point_estimate(), expected, places=12)

    def test_point_estimate_continuous(self) -> None:
        """Test point estimate for continuous outcomes."""
        ps, num, den = self.generate_data_continuous()
        estimator = RatioEstimator(
            propensity_scores=ps,
            numerator_outcomes=num,
            denominator_outcomes=den,
        )
        w = 1 / ps
        x_hat = np.sum(w * num) / np.sum(w)
        y_hat = np.sum(w * den) / np.sum(w)
        expected = x_hat / y_hat
        self.assertAlmostEqual(estimator.point_estimate(), expected, places=12)

    def test_variance(self) -> None:
        """Test variance calculation with delta method."""
        ps, num, den = self.generate_data_binary()
        estimator = RatioEstimator(ps, num, den)
        w = 1 / ps
        x_hat = np.sum(w * num) / np.sum(w)
        y_hat = np.sum(w * den) / np.sum(w)
        # Variance of stabilized mean
        var_x = np.sum(w**2 * (num - x_hat) ** 2) / (np.sum(w) ** 2)
        var_y = np.sum(w**2 * (den - y_hat) ** 2) / (np.sum(w) ** 2)
        cov_xy = np.sum(w**2 * (num - x_hat) * (den - y_hat)) / (np.sum(w) ** 2)
        expected_var = (
            (1 / y_hat) ** 2 * var_x
            + (x_hat / (y_hat**2)) ** 2 * var_y
            - 2 * x_hat / (y_hat**3) * cov_xy
        )
        self.assertAlmostEqual(estimator.variance(), expected_var, places=12)

    def test_pvalue(self) -> None:
        """Test p-value calculation for two-sided and one-sided alternatives."""
        ps, num, den = self.generate_data_binary()
        estimator = RatioEstimator(ps, num, den)
        pe = estimator.point_estimate()
        se = np.sqrt(estimator.variance())
        null = pe - 0.05
        # two-sided
        t = (pe - null) / se
        expected_p = min(1.0, 2 * min(stats.norm.sf(t), stats.norm.cdf(t)))
        self.assertAlmostEqual(
            estimator.pvalue(null_value=null, alternative="two-sided"),
            expected_p,
            places=8,
        )
        # greater
        expected_pg = stats.norm.cdf(t)
        self.assertAlmostEqual(
            estimator.pvalue(null_value=null, alternative="less"),
            expected_pg,
            places=8,
        )
        # less
        expected_pl = stats.norm.sf(t)
        self.assertAlmostEqual(
            estimator.pvalue(null_value=null, alternative="greater"),
            expected_pl,
            places=8,
        )

    def test_confidence_interval(self) -> None:
        """Test confidence interval computation for two-sided and one-sided."""
        ps, num, den = self.generate_data_binary()
        estimator = RatioEstimator(ps, num, den)
        pe = estimator.point_estimate()
        se = np.sqrt(estimator.variance())

        # Two-sided 90%
        lower, upper = estimator.confidence_interval(
            alpha=0.10, alternative="two-sided"
        )
        z = stats.norm.isf(0.10 / 2)
        expected_lb, expected_ub = pe - z * se, pe + z * se
        self.assertAlmostEqual(lower, expected_lb, places=12)
        self.assertAlmostEqual(upper, expected_ub, places=12)

        # Greater
        lower, upper = estimator.confidence_interval(alpha=0.05, alternative="greater")
        z = stats.norm.isf(0.05)
        expected_ub = pe + z * se
        self.assertAlmostEqual(lower, pe - z * se, places=12)
        self.assertEqual(upper, np.inf)

        # Less
        lower, upper = estimator.confidence_interval(alpha=0.05, alternative="less")
        self.assertAlmostEqual(upper, pe + z * se, places=12)
        self.assertEqual(lower, -np.inf)

    def test_sensitivity_analysis_binary(self) -> None:
        """Test sensitivity analysis for binary outcomes returns bounds in sensible order."""
        ps, num, den = self.generate_data_binary()
        estimator = RatioEstimator(ps, num, den)
        lb, ub = estimator.sensitivity_analysis(gamma=2.0)
        self.assertLessEqual(lb, ub)
        # Should be no wider than the expanded CI with gamma=6.0
        lb6, ub6 = estimator.sensitivity_analysis(gamma=6.0)
        self.assertLessEqual(lb6, lb)
        self.assertLessEqual(lb, ub)
        self.assertLessEqual(ub, ub6)

    def test_sensitivity_analysis_continuous(self) -> None:
        """Test sensitivity analysis for continuous outcomes."""
        ps, num, den = self.generate_data_continuous()
        estimator = RatioEstimator(ps, num, den)
        lb, ub = estimator.sensitivity_analysis(gamma=3.0)
        self.assertIsInstance(lb, float)
        self.assertIsInstance(ub, float)
        self.assertLessEqual(lb, ub)

    def test_expanded_confidence_interval(self) -> None:
        """Test expanded confidence interval bootstrapping produces interval with proper order."""
        ps, num, den = self.generate_data_binary(n=200)
        estimator = RatioEstimator(ps, num, den)
        lb_expected = 0.2943791251935982
        ub_expected = 0.6917976484180762
        lb, ub = estimator.expanded_confidence_interval(
            alpha=0.10, gamma=2.0, bootstrap=False
        )
        self.assertLessEqual(lb, ub)
        self.assertAlmostEqual(lb, lb_expected, places=5)
        self.assertAlmostEqual(ub, ub_expected, places=5)
        # Check that gamma=1 gives back the regular CI
        lb1, ub1 = estimator.expanded_confidence_interval(alpha=0.10, gamma=1.0)
        ci_lb, ci_ub = estimator.confidence_interval(alpha=0.10)
        self.assertAlmostEqual(lb1, ci_lb, places=5)
        self.assertAlmostEqual(ub1, ci_ub, places=5)

    def test_plot_sensitivity(self) -> None:
        """Sanity-check test: plot sensitivity does not error out and returns DataFrame and Axes."""
        ps, num, den = self.generate_data_binary(n=50)
        estimator = RatioEstimator(ps, num, den)
        expected_columns = [
            "Gamma",
            "ECI Lower Bound",
            "Sen Lower Bound",
            "Point Estimate",
            "Sen Upper Bound",
            "ECI Upper Bound",
        ]

        df, ax = estimator.plot_sensitivity(
            gamma_lower=1.0, gamma_upper=2.0, bootstrap=False, title="Test Plot"
        )
        # plt.show()
        self.assertEqual(set(df.columns), set(expected_columns))
        self.assertEqual(len(df), 50)
        plt.close("all")


class OutcomeProxySensitivityTest(unittest.TestCase):
    """Tests for outcome-proxy-constrained sensitivity analysis."""

    @staticmethod
    def _make_proxy(
        outcomes: npt.NDArray[np.float64],
        seed: int = 7,
        noise_scale: float = 0.05,
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Return (proxy, proxy_mean) where proxy = outcome + small noise."""
        rng = np.random.default_rng(seed)
        proxies = outcomes + noise_scale * rng.standard_normal(len(outcomes))
        proxy_mean = float(np.mean(proxies))
        return proxies, proxy_mean

    def test_ipw_proxy_tightens_interval(self) -> None:
        """Proxy-constrained IPW interval is contained in the unconstrained one."""
        population_size, propensities, outcomes = IPWEstimatorTest.get_data(
            binary=False
        )
        estimator = IPWEstimator(
            propensities, outcomes, population_size, estimand_class=PopulationMean
        )
        gamma = 4.0
        proxies, proxy_mean = OutcomeProxySensitivityTest._make_proxy(outcomes)

        lb0, ub0 = estimator.sensitivity_analysis(gamma=gamma)
        lb1, ub1 = estimator.sensitivity_analysis(
            gamma=gamma, outcome_proxies=proxies, proxy_mean=proxy_mean
        )

        self.assertGreaterEqual(lb1, lb0 - 1e-10)
        self.assertLessEqual(ub1, ub0 + 1e-10)
        self.assertLess(ub1 - lb1, ub0 - lb0)

    def test_ipw_proxy_gamma_one(self) -> None:
        """gamma=1 returns the point estimate regardless of proxy."""
        population_size, propensities, outcomes = IPWEstimatorTest.get_data(
            binary=False
        )
        estimator = IPWEstimator(
            propensities, outcomes, population_size, estimand_class=PopulationMean
        )
        proxies, proxy_mean = OutcomeProxySensitivityTest._make_proxy(outcomes)
        lb, ub = estimator.sensitivity_analysis(
            gamma=1.0, outcome_proxies=proxies, proxy_mean=proxy_mean
        )
        pe = estimator.point_estimate()
        self.assertAlmostEqual(lb, pe)
        self.assertAlmostEqual(ub, pe)

    def test_ipw_proxy_missing_proxy_mean_raises(self) -> None:
        """Providing outcome_proxies without proxy_mean raises ValueError."""
        population_size, propensities, outcomes = IPWEstimatorTest.get_data(
            binary=False
        )
        estimator = IPWEstimator(
            propensities, outcomes, population_size, estimand_class=PopulationMean
        )
        proxies, _ = OutcomeProxySensitivityTest._make_proxy(outcomes)
        with self.assertRaisesRegex(ValueError, "proxy_mean"):
            estimator.sensitivity_analysis(gamma=4.0, outcome_proxies=proxies)

    def test_aipw_proxy_tightens_interval(self) -> None:
        """Proxy-constrained AIPW interval is contained in the unconstrained one."""
        (
            population_size,
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = AIPWEstimatorTest.get_data(binary=False)
        estimator = AIPWEstimator(
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
            population_size,
        )
        gamma = 4.0
        proxies, proxy_mean = OutcomeProxySensitivityTest._make_proxy(outcomes)

        lb0, ub0 = estimator.sensitivity_analysis(gamma=gamma)
        lb1, ub1 = estimator.sensitivity_analysis(
            gamma=gamma, outcome_proxies=proxies, proxy_mean=proxy_mean
        )

        self.assertGreaterEqual(lb1, lb0 - 1e-10)
        self.assertLessEqual(ub1, ub0 + 1e-10)
        self.assertLess(ub1 - lb1, ub0 - lb0)

    def test_sipw_proxy_tightens_interval_binary(self) -> None:
        """Proxy-constrained SIPW interval (binary)."""
        propensities, outcomes = SIPWEstimatorTest.get_data(binary=True)
        estimator = SIPWEstimator(propensities, outcomes)
        gamma = 4.0
        proxies, proxy_mean = OutcomeProxySensitivityTest._make_proxy(
            outcomes, noise_scale=0.02
        )

        lb0, ub0 = estimator.sensitivity_analysis(gamma=gamma)
        lb1, ub1 = estimator.sensitivity_analysis(
            gamma=gamma, outcome_proxies=proxies, proxy_mean=proxy_mean
        )

        self.assertGreaterEqual(lb1, lb0 - 1e-10)
        self.assertLessEqual(ub1, ub0 + 1e-10)
        self.assertLess(ub1 - lb1, ub0 - lb0)

    def test_sipw_proxy_tightens_interval_continuous(self) -> None:
        """Proxy-constrained SIPW interval (continuous)."""
        propensities, outcomes = SIPWEstimatorTest.get_data(binary=False)
        estimator = SIPWEstimator(propensities, outcomes)
        gamma = 4.0
        proxies, proxy_mean = OutcomeProxySensitivityTest._make_proxy(outcomes)

        lb0, ub0 = estimator.sensitivity_analysis(gamma=gamma)
        lb1, ub1 = estimator.sensitivity_analysis(
            gamma=gamma, outcome_proxies=proxies, proxy_mean=proxy_mean
        )

        self.assertGreaterEqual(lb1, lb0 - 1e-10)
        self.assertLessEqual(ub1, ub0 + 1e-10)
        self.assertLess(ub1 - lb1, ub0 - lb0)

    def test_sipw_proxy_gamma_one(self) -> None:
        """gamma=1 returns the point estimate regardless of proxy."""
        propensities, outcomes = SIPWEstimatorTest.get_data(binary=False)
        estimator = SIPWEstimator(propensities, outcomes)
        proxies, proxy_mean = OutcomeProxySensitivityTest._make_proxy(outcomes)
        lb, ub = estimator.sensitivity_analysis(
            gamma=1.0, outcome_proxies=proxies, proxy_mean=proxy_mean
        )
        pe = estimator.point_estimate()
        self.assertAlmostEqual(lb, pe)
        self.assertAlmostEqual(ub, pe)

    def test_sipw_proxy_missing_proxy_mean_raises(self) -> None:
        """Providing outcome_proxies without proxy_mean raises ValueError."""
        propensities, outcomes = SIPWEstimatorTest.get_data(binary=False)
        estimator = SIPWEstimator(propensities, outcomes)
        proxies, _ = OutcomeProxySensitivityTest._make_proxy(outcomes)
        with self.assertRaisesRegex(ValueError, "proxy_mean"):
            estimator.sensitivity_analysis(gamma=4.0, outcome_proxies=proxies)

    def test_saipw_proxy_tightens_interval(self) -> None:
        """Proxy-constrained SAIPW interval is contained in the unconstrained one."""
        (
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = SAIPWEstimatorTest.get_data(binary=False)
        estimator = SAIPWEstimator(
            propensities, outcomes, predicted_outcomes, mean_predicted_outcome
        )
        gamma = 4.0
        proxies, proxy_mean = OutcomeProxySensitivityTest._make_proxy(outcomes)

        lb0, ub0 = estimator.sensitivity_analysis(gamma=gamma)
        lb1, ub1 = estimator.sensitivity_analysis(
            gamma=gamma, outcome_proxies=proxies, proxy_mean=proxy_mean
        )

        self.assertGreaterEqual(lb1, lb0 - 1e-10)
        self.assertLessEqual(ub1, ub0 + 1e-10)
        self.assertLess(ub1 - lb1, ub0 - lb0)
