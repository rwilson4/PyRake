"""Test estimators."""

import math
import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest
from scipy import optimize, stats
from scipy.special import expit, logit

from pyrake.estimators import (
    AIPWEstimator,
    ATCEstimator,
    ATEEstimator,
    ATTEstimator,
    IPWEstimator,
    PopulationMean,
    RatioEstimator,
    SAIPWEstimator,
    SampleMean,
    SimpleDifferenceEstimator,
    SIPWEstimator,
    TreatmentEffectEstimator,
    TreatmentEffectRatioEstimator,
)


class TestIPWEstimator:
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

    @staticmethod
    def test_point_estimate() -> None:
        """Test point estimate."""
        population_size, propensities, outcomes = TestIPWEstimator.get_data()
        estimator = IPWEstimator(
            propensities,
            outcomes,
            population_size,
            estimand_class=PopulationMean,
        )
        expected = 0.4561422941383326

        actual = estimator.point_estimate()
        assert actual == pytest.approx(expected)

        # estimator = IPWEstimator(
        #     propensities,
        #     outcomes,
        #     population_size - len(propensities),
        #     estimand_class=NonRespondentMean,
        # )
        # expected = 0.4561424365749074

        # actual = estimator.point_estimate()
        # assert actual == pytest.approx(expected)

        # estimator = IPWEstimator(
        #     propensities, outcomes, len(propensities), estimand_class=SampleMean
        # )
        # expected = float(np.mean(outcomes))

        # actual = estimator.point_estimate()
        # assert actual == pytest.approx(expected)

    @staticmethod
    def test_variance() -> None:
        """Test variance."""
        population_size, propensities, outcomes = TestIPWEstimator.get_data()
        estimator = IPWEstimator(
            propensities,
            outcomes,
            population_size,
            estimand_class=PopulationMean,
        )
        expected = 0.0002489382388053698

        actual = estimator.variance()
        assert actual == pytest.approx(expected)

        # estimator = IPWEstimator(
        #     propensities,
        #     outcomes,
        #     population_size - len(propensities),
        #     estimand_class=NonRespondentMean,
        # )
        # expected = 0.00024893933725724123

        # actual = estimator.variance()
        # assert actual == pytest.approx(expected)

        # estimator = IPWEstimator(
        #     propensities, outcomes, len(propensities), estimand_class=SampleMean
        # )
        # expected = float(
        #     np.mean(outcomes)
        #     * (1.0 - float(np.mean(outcomes)))
        #     / (len(propensities) - 1)
        # )

        # actual = estimator.variance()
        # assert actual == pytest.approx(expected)

    @staticmethod
    def test_pvalue() -> None:
        """Test p-value calculation."""
        population_size, propensities, outcomes = TestIPWEstimator.get_data()
        estimator = IPWEstimator(
            propensities,
            outcomes,
            population_size,
            estimand_class=PopulationMean,
        )
        expected = 0.005440680301504909

        actual = estimator.pvalue(null_value=0.50)
        assert actual == pytest.approx(expected)

        # Test one-sided p-values
        actual = estimator.pvalue(null_value=0.50, alternative="less")
        assert actual < 0.5
        assert actual == pytest.approx(0.5 * expected)

        actual = estimator.pvalue(null_value=0.50, alternative="greater")
        assert actual > 0.5

    @staticmethod
    def test_confidence_interval() -> None:
        """Test confidence interval."""
        population_size, propensities, outcomes = TestIPWEstimator.get_data()
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
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

        # Test one-sided intervals
        lb_actual, ub_actual = estimator.confidence_interval(
            alpha=0.05, alternative="less"
        )
        print(lb_actual, ub_actual)
        assert lb_actual == -np.inf
        assert ub_actual == pytest.approx(ub_expected)

        lb_actual, ub_actual = estimator.confidence_interval(
            alpha=0.05, alternative="greater"
        )
        print(lb_actual, ub_actual)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == np.inf

    @staticmethod
    def test_sensitivity_analysis_binary_outcomes() -> None:
        """Test sensitivity analysis."""
        population_size, propensities, outcomes = TestIPWEstimator.get_data()
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
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

        # estimator = IPWEstimator(
        #     propensities,
        #     outcomes,
        #     population_size - len(propensities),
        #     estimand_class=NonRespondentMean,
        # )
        # lb_expected = 0.18621936993972707
        # ub_expected = 1.1173162196383621

        # lb_actual, ub_actual = estimator.sensitivity_analysis(gamma=gamma)
        # print(lb_actual, ub_actual)
        # assert lb_actual == pytest.approx(lb_expected)
        # assert ub_actual == pytest.approx(ub_expected)

        # estimator = IPWEstimator(
        #     propensities, outcomes, len(propensities), estimand_class=SampleMean
        # )
        # lb_expected = float(np.mean(outcomes))
        # ub_expected = float(np.mean(outcomes))

        # lb_actual, ub_actual = estimator.sensitivity_analysis(gamma=gamma)
        # print(lb_actual, ub_actual)
        # assert lb_actual == pytest.approx(lb_expected)
        # assert ub_actual == pytest.approx(ub_expected)

    @staticmethod
    def test_sensitivity_analysis_continuous_outcomes() -> None:
        """Test sensitivity analysis."""
        population_size, propensities, outcomes = TestIPWEstimator.get_data(
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
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

    @staticmethod
    def test_expanded_confidence_interval() -> None:
        """Test expanded confidence interval."""
        population_size, propensities, outcomes = TestIPWEstimator.get_data()
        estimator = IPWEstimator(
            propensities,
            outcomes,
            population_size,
            estimand_class=PopulationMean,
        )
        lb_expected = 0.17558574796577736
        ub_expected = 1.1792322609834998

        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.1, gamma=6.0, seed=42
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

        # Test one-sided intervals
        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.05, gamma=6.0, alternative="less", seed=42
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        assert lb_actual == -np.inf
        assert ub_actual == pytest.approx(ub_expected)

        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.05, gamma=6.0, alternative="greater", seed=42
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == np.inf

    @staticmethod
    def test_plot_sensitivity() -> None:
        """Test sensitivity analysis."""
        population_size, propensities, outcomes = TestIPWEstimator.get_data()
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
            B=1_000,
            axis_label_size=18,
            tick_label_size=14,
        )
        # plt.show()
        assert set(df.columns) == set(expected_columns)
        assert len(df) == 50
        plt.close("all")


class TestAIPWEstimator:
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

    @staticmethod
    def test_point_estimate() -> None:
        """Test point estimate."""
        (
            population_size,
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = TestAIPWEstimator.get_data()
        estimator = AIPWEstimator(
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
            population_size,
        )
        expected = 0.49912034525845983

        actual = estimator.point_estimate()
        assert actual == pytest.approx(expected)

        # estimator = AIPWEstimator(
        #     propensities,
        #     outcomes,
        #     predicted_outcomes,
        #     float(np.mean(predicted_outcomes)),
        #     len(propensities),
        #     estimand_class=SampleMean,
        # )
        # expected = float(np.mean(outcomes))

        # actual = estimator.point_estimate()
        # assert actual == pytest.approx(expected)

    @staticmethod
    def test_variance() -> None:
        """Test variance."""
        (
            population_size,
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = TestAIPWEstimator.get_data()
        estimator = AIPWEstimator(
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
            population_size,
        )
        expected = 9.951767238492971e-08

        actual = estimator.variance()
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_pvalue() -> None:
        """Test p-value calculation."""
        (
            population_size,
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = TestAIPWEstimator.get_data()
        estimator = AIPWEstimator(
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
            population_size,
        )
        expected = 0.005296167612649354

        actual = estimator.pvalue(null_value=0.50)
        assert actual == pytest.approx(expected)

        # Test one-sided p-values
        actual = estimator.pvalue(null_value=0.50, alternative="less")
        assert actual < 0.5
        assert actual == pytest.approx(0.5 * expected)

        actual = estimator.pvalue(null_value=0.50, alternative="greater")
        assert actual > 0.5

    @staticmethod
    def test_confidence_interval() -> None:
        """Test confidence interval."""
        (
            population_size,
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = TestAIPWEstimator.get_data()
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
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

        # Test one-sided intervals
        lb_actual, ub_actual = estimator.confidence_interval(
            alpha=0.05, alternative="less"
        )
        print(lb_actual, ub_actual)
        assert lb_actual == -np.inf
        assert ub_actual == pytest.approx(ub_expected)

        lb_actual, ub_actual = estimator.confidence_interval(
            alpha=0.05, alternative="greater"
        )
        print(lb_actual, ub_actual)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == np.inf

    @staticmethod
    def test_sensitivity_analysis_binary_outcomes() -> None:
        """Test sensitivity analysis."""
        (
            population_size,
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = TestAIPWEstimator.get_data()
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
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

    @staticmethod
    def test_sensitivity_analysis_continuous_outcomes() -> None:
        """Test sensitivity analysis."""
        (
            population_size,
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = TestAIPWEstimator.get_data(binary=False)
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
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

    @staticmethod
    def test_expanded_confidence_interval() -> None:
        """Test expanded confidence interval."""
        (
            population_size,
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = TestAIPWEstimator.get_data()
        estimator = AIPWEstimator(
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
            population_size,
        )
        lb_expected = 0.48780560815933593
        ub_expected = 0.5096700826129239

        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.1, gamma=6.0, seed=42
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

        # Test one-sided intervals
        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.05, gamma=6.0, alternative="less", seed=42
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        assert lb_actual == -np.inf
        assert ub_actual == pytest.approx(ub_expected)

        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.05, gamma=6.0, alternative="greater", seed=42
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == np.inf

    @staticmethod
    def test_plot_sensitivity() -> None:
        """Test sensitivity analysis."""
        (
            population_size,
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = TestAIPWEstimator.get_data()
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
            B=1_000,
            axis_label_size=18,
            tick_label_size=14,
            ytick_format=".02%",
        )
        # plt.show()
        assert set(df.columns) == set(expected_columns)
        assert len(df) == 50
        plt.close("all")


class TestSIPWEstimator:
    @staticmethod
    def get_data(
        binary: bool = True,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        population_size, propensities, outcomes = TestIPWEstimator.get_data(
            binary=binary
        )
        return propensities, outcomes

    @staticmethod
    def test_point_estimate() -> None:
        """Test point estimate."""
        propensities, outcomes = TestSIPWEstimator.get_data()
        estimator = SIPWEstimator(propensities, outcomes)
        expected = 0.4560066389336293

        actual = estimator.point_estimate()
        assert actual == pytest.approx(expected)

        # SIPW is invariant to changing the scale of the weights
        estimator = SIPWEstimator(2.0 * propensities, outcomes)
        actual = estimator.point_estimate()
        assert actual == pytest.approx(expected)

        estimator = SIPWEstimator(propensities, outcomes, estimand_class=SampleMean)
        expected = float(np.mean(outcomes))

        actual = estimator.point_estimate()
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_variance() -> None:
        """Test variance."""
        propensities, outcomes = TestSIPWEstimator.get_data()
        estimator = SIPWEstimator(propensities, outcomes)
        expected = 0.0002483163797360267

        actual = estimator.variance()
        assert actual == pytest.approx(expected)

        # SIPW is invariant to changing the scale of the weights
        estimator = SIPWEstimator(2.0 * propensities, outcomes)
        actual = estimator.variance()
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_pvalue() -> None:
        """Test p-value calculation."""
        propensities, outcomes = TestSIPWEstimator.get_data()
        estimator = SIPWEstimator(propensities, outcomes)
        expected = 0.005241558051856385

        actual = estimator.pvalue(null_value=0.50)
        assert actual == pytest.approx(expected)

        # Test one-sided p-values
        actual = estimator.pvalue(null_value=0.50, alternative="less")
        assert actual < 0.5
        assert actual == pytest.approx(0.5 * expected)

        actual = estimator.pvalue(null_value=0.50, alternative="greater")
        assert actual > 0.5

        # SIPW is invariant to changing the scale of the weights
        estimator = SIPWEstimator(2.0 * propensities, outcomes)
        actual = estimator.pvalue(null_value=0.50)
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_confidence_interval() -> None:
        """Test confidence interval."""
        propensities, outcomes = TestSIPWEstimator.get_data()
        estimator = SIPWEstimator(propensities, outcomes)
        lb_expected = 0.43008694071510295
        ub_expected = 0.48192633715215566

        lb_actual, ub_actual = estimator.confidence_interval(alpha=0.1)
        print(lb_actual, ub_actual)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

        # Test one-sided intervals
        lb_actual, ub_actual = estimator.confidence_interval(
            alpha=0.05, alternative="less"
        )
        print(lb_actual, ub_actual)
        assert lb_actual == -np.inf
        assert ub_actual == pytest.approx(ub_expected)

        lb_actual, ub_actual = estimator.confidence_interval(
            alpha=0.05, alternative="greater"
        )
        print(lb_actual, ub_actual)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == np.inf

        # SIPW is invariant to changing the scale of the weights
        estimator = SIPWEstimator(2.0 * propensities, outcomes)
        lb_actual, ub_actual = estimator.confidence_interval(alpha=0.1)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

    @staticmethod
    def test_sensitivity_analysis_binary_outcomes() -> None:
        """Test sensitivity analysis."""
        propensities, outcomes = TestSIPWEstimator.get_data()
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
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

    @staticmethod
    def test_sensitivity_analysis_continuous_outcomes() -> None:
        """Test sensitivity analysis."""
        propensities, outcomes = TestSIPWEstimator.get_data(binary=False)
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
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

    @staticmethod
    def test_expanded_confidence_interval() -> None:
        """Test expanded confidence interval."""
        propensities, outcomes = TestSIPWEstimator.get_data()
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
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

        # Test one-sided intervals
        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.05, gamma=6.0, alternative="less", seed=42
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        assert lb_actual == -np.inf
        assert ub_actual == pytest.approx(ub_expected)

        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.05, gamma=6.0, alternative="greater", seed=42
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == np.inf

    @staticmethod
    def test_plot_sensitivity() -> None:
        """Test sensitivity analysis."""
        propensities, outcomes = TestSIPWEstimator.get_data()
        estimator = SIPWEstimator(propensities, outcomes)
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
            B=1_000,
            axis_label_size=18,
            tick_label_size=14,
        )
        # plt.show()
        assert set(df.columns) == set(expected_columns)
        assert len(df) == 50
        plt.close("all")


class TestSAIPWEstimator:
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
        ) = TestAIPWEstimator.get_data(binary=binary)
        return (
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        )

    @staticmethod
    def test_point_estimate() -> None:
        """Test point estimate."""
        (
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = TestSAIPWEstimator.get_data()
        estimator = SAIPWEstimator(
            propensities, outcomes, predicted_outcomes, mean_predicted_outcome
        )
        expected = 0.4991206068648611

        actual = estimator.point_estimate()
        assert actual == pytest.approx(expected)

        # SAIPW is invariant to changing the scale of the weights
        estimator = SAIPWEstimator(
            2.0 * propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        )
        actual = estimator.point_estimate()
        assert actual == pytest.approx(expected)

        estimator = SAIPWEstimator(
            propensities,
            outcomes,
            predicted_outcomes,
            float(np.mean(predicted_outcomes)),
            estimand_class=SampleMean,
        )
        expected = float(np.mean(outcomes))

        actual = estimator.point_estimate()
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_variance() -> None:
        """Test variance."""
        (
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = TestSAIPWEstimator.get_data()
        estimator = SAIPWEstimator(
            propensities, outcomes, predicted_outcomes, mean_predicted_outcome
        )
        expected = 9.935866236133295e-08

        actual = estimator.variance()
        assert actual == pytest.approx(expected)

        # SAIPW is invariant to changing the scale of the weights
        estimator = SAIPWEstimator(
            2.0 * propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        )
        actual = estimator.variance()
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_pvalue() -> None:
        """Test p-value calculation."""
        (
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = TestSAIPWEstimator.get_data()
        estimator = SAIPWEstimator(
            propensities, outcomes, predicted_outcomes, mean_predicted_outcome
        )
        expected = 0.005273314664915336

        actual = estimator.pvalue(null_value=0.50)
        assert actual == pytest.approx(expected)

        # Test one-sided p-values
        actual = estimator.pvalue(null_value=0.50, alternative="less")
        assert actual < 0.5
        assert actual == pytest.approx(0.5 * expected)

        actual = estimator.pvalue(null_value=0.50, alternative="greater")
        assert actual > 0.5

        # SAIPW is invariant to changing the scale of the weights
        estimator = SAIPWEstimator(
            2.0 * propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        )
        actual = estimator.pvalue(null_value=0.50)
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_confidence_interval() -> None:
        """Test confidence interval."""
        (
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = TestSAIPWEstimator.get_data()
        estimator = SAIPWEstimator(
            propensities, outcomes, predicted_outcomes, mean_predicted_outcome
        )
        lb_expected = 0.49860212911359364
        ub_expected = 0.49963908461612855

        lb_actual, ub_actual = estimator.confidence_interval(alpha=0.1)
        print(lb_actual, ub_actual)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

        # Test one-sided intervals
        lb_actual, ub_actual = estimator.confidence_interval(
            alpha=0.05, alternative="less"
        )
        print(lb_actual, ub_actual)
        assert lb_actual == -np.inf
        assert ub_actual == pytest.approx(ub_expected)

        lb_actual, ub_actual = estimator.confidence_interval(
            alpha=0.05, alternative="greater"
        )
        print(lb_actual, ub_actual)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == np.inf

        # SAIPW is invariant to changing the scale of the weights
        estimator = SAIPWEstimator(
            2.0 * propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        )
        lb_actual, ub_actual = estimator.confidence_interval(alpha=0.1)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

    @staticmethod
    def test_sensitivity_analysis_binary_outcomes() -> None:
        """Test sensitivity analysis."""
        (
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = TestSAIPWEstimator.get_data()
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
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

    @staticmethod
    def test_sensitivity_analysis_continuous_outcomes() -> None:
        """Test sensitivity analysis."""
        (
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = TestSAIPWEstimator.get_data()
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
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

    @staticmethod
    def test_expanded_confidence_interval() -> None:
        """Test expanded confidence interval."""
        (
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = TestSAIPWEstimator.get_data()
        estimator = SAIPWEstimator(
            propensities, outcomes, predicted_outcomes, mean_predicted_outcome
        )
        lb_expected = 0.4922371081338087
        ub_expected = 0.5069528390649397

        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.1, gamma=6.0, seed=42
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

        # Test one-sided intervals
        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.05, gamma=6.0, alternative="less", seed=42
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        assert lb_actual == -np.inf
        assert ub_actual == pytest.approx(ub_expected)

        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.05, gamma=6.0, alternative="greater", seed=42
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == np.inf

    @staticmethod
    def test_plot_sensitivity() -> None:
        """Test sensitivity analysis."""
        (
            propensities,
            outcomes,
            predicted_outcomes,
            mean_predicted_outcome,
        ) = TestSAIPWEstimator.get_data()
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
            B=100,
            axis_label_size=18,
            tick_label_size=14,
            ytick_format=".02%",
        )
        # plt.show()
        assert set(df.columns) == set(expected_columns)
        assert len(df) == 50
        plt.close("all")


class TestRatioEstimator:
    """Pytest test cases for RatioEstimator class."""

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
        assert estimator.point_estimate() == pytest.approx(expected, rel=1e-12)

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
        assert estimator.point_estimate() == pytest.approx(expected, rel=1e-12)

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
        assert estimator.variance() == pytest.approx(expected_var, rel=1e-12)

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
        assert estimator.pvalue(
            null_value=null, alternative="two-sided"
        ) == pytest.approx(expected_p, rel=1e-8)
        # greater
        expected_pg = stats.norm.cdf(t)
        assert estimator.pvalue(null_value=null, alternative="less") == pytest.approx(
            expected_pg, rel=1e-8
        )
        # less
        expected_pl = stats.norm.sf(t)
        assert estimator.pvalue(
            null_value=null, alternative="greater"
        ) == pytest.approx(expected_pl, rel=1e-8)

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
        assert lower == pytest.approx(expected_lb, rel=1e-12)
        assert upper == pytest.approx(expected_ub, rel=1e-12)

        # Greater
        lower, upper = estimator.confidence_interval(alpha=0.05, alternative="greater")
        z = stats.norm.isf(0.05)
        expected_ub = pe + z * se
        assert lower == pytest.approx(pe - z * se, rel=1e-12)
        assert upper == np.inf

        # Less
        lower, upper = estimator.confidence_interval(alpha=0.05, alternative="less")
        assert upper == pytest.approx(pe + z * se, rel=1e-12)
        assert lower == -np.inf

    def test_sensitivity_analysis_binary(self) -> None:
        """Test sensitivity analysis for binary outcomes returns bounds in sensible order."""
        ps, num, den = self.generate_data_binary()
        estimator = RatioEstimator(ps, num, den)
        lb, ub = estimator.sensitivity_analysis(gamma=2.0)
        assert lb <= ub
        # Should be no wider than the expanded CI with gamma=6.0
        lb6, ub6 = estimator.sensitivity_analysis(gamma=6.0)
        assert lb6 <= lb <= ub <= ub6

    def test_sensitivity_analysis_continuous(self) -> None:
        """Test sensitivity analysis for continuous outcomes."""
        ps, num, den = self.generate_data_continuous()
        estimator = RatioEstimator(ps, num, den)
        lb, ub = estimator.sensitivity_analysis(gamma=3.0)
        assert isinstance(lb, float)
        assert isinstance(ub, float)
        assert lb <= ub

    def test_expanded_confidence_interval(self) -> None:
        """Test expanded confidence interval bootstrapping produces interval with proper order."""
        ps, num, den = self.generate_data_binary(n=200)
        estimator = RatioEstimator(ps, num, den)
        lb, ub = estimator.expanded_confidence_interval(
            alpha=0.10, gamma=2.0, B=250, seed=10
        )
        assert lb <= ub
        # Check that gamma=1 gives back the regular CI
        lb1, ub1 = estimator.expanded_confidence_interval(
            alpha=0.10, gamma=1.0, B=100, seed=11
        )
        ci_lb, ci_ub = estimator.confidence_interval(alpha=0.10)
        assert lb1 == pytest.approx(ci_lb, rel=1e-5)
        assert ub1 == pytest.approx(ci_ub, rel=1e-5)

    def test_plot_sensitivity(self) -> None:
        """Sanity-check test: plot sensitivity does not error out and returns DataFrame and Axes."""
        # import matplotlib
        # matplotlib.use('Agg')  # Use a non-interactive backend
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
            gamma_lower=1.0, gamma_upper=2.0, B=30, title="Test Plot"
        )
        # plt.show()
        assert set(df.columns) == set(expected_columns)
        assert len(df) == 50
        plt.close("all")


class TestSimpleDifferenceEstimator:
    @staticmethod
    def get_data(
        binary: bool = True,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        population_size, propensities, outcomes = TestIPWEstimator.get_data(
            binary=binary
        )
        sample_size = len(propensities)
        rng = np.random.default_rng(64)
        assignments = rng.binomial(1, 0.5, size=sample_size)

        control_mask = assignments == 0
        treated_mask = assignments == 1

        return (
            propensities[control_mask],
            outcomes[control_mask],
            propensities[treated_mask],
            outcomes[treated_mask],
        )

    @staticmethod
    def test_point_estimate() -> None:
        """Test point estimate."""

        (
            control_propensities,
            control_outcomes,
            treated_propensities,
            treated_outcomes,
        ) = TestSimpleDifferenceEstimator.get_data()

        estimator = SimpleDifferenceEstimator(
            control_outcomes=control_outcomes,
            treated_outcomes=treated_outcomes,
        )

        expected = float(np.mean(treated_outcomes) - np.mean(control_outcomes))
        actual = estimator.point_estimate()
        assert actual == pytest.approx(expected)

        estimator = SimpleDifferenceEstimator(
            control_outcomes=control_outcomes,
            treated_outcomes=treated_outcomes,
            control_sampling_propensity_scores=control_propensities,
            treated_sampling_propensity_scores=treated_propensities,
        )

        pate_estimator = TreatmentEffectEstimator(
            control_estimator=SIPWEstimator(
                propensity_scores=control_propensities,
                outcomes=control_outcomes,
            ),
            treated_estimator=SIPWEstimator(
                propensity_scores=treated_propensities,
                outcomes=treated_outcomes,
            ),
        )

        expected = pate_estimator.point_estimate()
        actual = estimator.point_estimate()
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_variance() -> None:
        """Test variance."""

        (
            control_propensities,
            control_outcomes,
            treated_propensities,
            treated_outcomes,
        ) = TestSimpleDifferenceEstimator.get_data()

        estimator = SimpleDifferenceEstimator(
            control_outcomes=control_outcomes,
            treated_outcomes=treated_outcomes,
        )

        mu_t = float(np.mean(treated_outcomes))
        mu_c = float(np.mean(control_outcomes))

        expected = float(
            mu_t * (1 - mu_t) / len(treated_outcomes)
            + mu_c * (1 - mu_c) / len(control_outcomes)
        )
        actual = estimator.variance()
        assert actual == pytest.approx(expected)

        estimator = SimpleDifferenceEstimator(
            control_outcomes=control_outcomes,
            treated_outcomes=treated_outcomes,
            control_sampling_propensity_scores=control_propensities,
            treated_sampling_propensity_scores=treated_propensities,
        )

        pate_estimator = TreatmentEffectEstimator(
            control_estimator=SIPWEstimator(
                propensity_scores=control_propensities,
                outcomes=control_outcomes,
            ),
            treated_estimator=SIPWEstimator(
                propensity_scores=treated_propensities,
                outcomes=treated_outcomes,
            ),
        )
        expected = pate_estimator.variance()
        actual = estimator.variance()
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_pvalue() -> None:
        """Test pvalue."""

        (
            control_propensities,
            control_outcomes,
            treated_propensities,
            treated_outcomes,
        ) = TestSimpleDifferenceEstimator.get_data()

        estimator = SimpleDifferenceEstimator(
            control_outcomes=control_outcomes,
            treated_outcomes=treated_outcomes,
        )

        mu_t = float(np.mean(treated_outcomes))
        mu_c = float(np.mean(control_outcomes))

        var_tc = float(
            mu_t * (1 - mu_t) / len(treated_outcomes)
            + mu_c * (1 - mu_c) / len(control_outcomes)
        )
        z = (mu_t - mu_c) / math.sqrt(var_tc)
        expected = 2.0 * stats.norm.sf(abs(z))

        actual = estimator.pvalue()
        assert actual == pytest.approx(expected)

        estimator = SimpleDifferenceEstimator(
            control_outcomes=control_outcomes,
            treated_outcomes=treated_outcomes,
            control_sampling_propensity_scores=control_propensities,
            treated_sampling_propensity_scores=treated_propensities,
        )

        pate_estimator = TreatmentEffectEstimator(
            control_estimator=SIPWEstimator(
                propensity_scores=control_propensities,
                outcomes=control_outcomes,
            ),
            treated_estimator=SIPWEstimator(
                propensity_scores=treated_propensities,
                outcomes=treated_outcomes,
            ),
        )
        expected = pate_estimator.pvalue()
        actual = estimator.pvalue()
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_confidence_interval() -> None:
        """Test confidence interval."""

        (
            control_propensities,
            control_outcomes,
            treated_propensities,
            treated_outcomes,
        ) = TestSimpleDifferenceEstimator.get_data()

        estimator = SimpleDifferenceEstimator(
            control_outcomes=control_outcomes,
            treated_outcomes=treated_outcomes,
        )

        mu_t = float(np.mean(treated_outcomes))
        mu_c = float(np.mean(control_outcomes))

        var_tc = float(
            mu_t * (1 - mu_t) / len(treated_outcomes)
            + mu_c * (1 - mu_c) / len(control_outcomes)
        )
        zcrit = stats.norm.isf(0.05)
        expected_lb = mu_t - mu_c - zcrit * math.sqrt(var_tc)
        expected_ub = mu_t - mu_c + zcrit * math.sqrt(var_tc)

        actual_lb, actual_ub = estimator.confidence_interval(alpha=0.1)
        assert actual_lb == pytest.approx(expected_lb)
        assert actual_ub == pytest.approx(expected_ub)

        estimator = SimpleDifferenceEstimator(
            control_outcomes=control_outcomes,
            treated_outcomes=treated_outcomes,
            control_sampling_propensity_scores=control_propensities,
            treated_sampling_propensity_scores=treated_propensities,
        )

        pate_estimator = TreatmentEffectEstimator(
            control_estimator=SIPWEstimator(
                propensity_scores=control_propensities,
                outcomes=control_outcomes,
            ),
            treated_estimator=SIPWEstimator(
                propensity_scores=treated_propensities,
                outcomes=treated_outcomes,
            ),
        )
        expected_lb, expected_ub = pate_estimator.confidence_interval(alpha=0.1)
        actual_lb, actual_ub = estimator.confidence_interval(alpha=0.1)
        assert actual_lb == pytest.approx(expected_lb)
        assert actual_ub == pytest.approx(expected_ub)

    @staticmethod
    def test_sensitivity_analysis() -> None:
        """Test sensitivity analysis."""

        (
            control_propensities,
            control_outcomes,
            treated_propensities,
            treated_outcomes,
        ) = TestSimpleDifferenceEstimator.get_data()

        estimator = SimpleDifferenceEstimator(
            control_outcomes=control_outcomes,
            treated_outcomes=treated_outcomes,
        )

        mu_t = np.mean(treated_outcomes)
        mu_c = np.mean(control_outcomes)

        expected_lb = float(mu_t - mu_c)
        expected_ub = float(mu_t - mu_c)

        actual_lb, actual_ub = estimator.sensitivity_analysis()
        assert actual_lb == pytest.approx(expected_lb)
        assert actual_ub == pytest.approx(expected_ub)

        estimator = SimpleDifferenceEstimator(
            control_outcomes=control_outcomes,
            treated_outcomes=treated_outcomes,
            control_sampling_propensity_scores=control_propensities,
            treated_sampling_propensity_scores=treated_propensities,
        )

        pate_estimator = TreatmentEffectEstimator(
            control_estimator=SIPWEstimator(
                propensity_scores=control_propensities,
                outcomes=control_outcomes,
            ),
            treated_estimator=SIPWEstimator(
                propensity_scores=treated_propensities,
                outcomes=treated_outcomes,
            ),
        )
        expected_lb, expected_ub = pate_estimator.sensitivity_analysis()
        actual_lb, actual_ub = estimator.sensitivity_analysis()
        assert actual_lb == pytest.approx(expected_lb)

    @staticmethod
    def test_expanded_confidence_interval() -> None:
        """Test confidence interval."""

        (
            control_propensities,
            control_outcomes,
            treated_propensities,
            treated_outcomes,
        ) = TestSimpleDifferenceEstimator.get_data()

        estimator = SimpleDifferenceEstimator(
            control_outcomes=control_outcomes,
            treated_outcomes=treated_outcomes,
        )

        mu_t = float(np.mean(treated_outcomes))
        mu_c = float(np.mean(control_outcomes))

        var_tc = float(
            mu_t * (1 - mu_t) / len(treated_outcomes)
            + mu_c * (1 - mu_c) / len(control_outcomes)
        )
        zcrit = stats.norm.isf(0.05)
        expected_lb = mu_t - mu_c - zcrit * math.sqrt(var_tc)
        expected_ub = mu_t - mu_c + zcrit * math.sqrt(var_tc)

        actual_lb, actual_ub = estimator.expanded_confidence_interval(
            alpha=0.1, seed=1234
        )
        # Some differences are expected here, b/c the expanded confidence interval is using a bootstrap.
        assert actual_lb == pytest.approx(expected_lb, abs=0.001)
        assert actual_ub == pytest.approx(expected_ub, abs=0.001)

        estimator = SimpleDifferenceEstimator(
            control_outcomes=control_outcomes,
            treated_outcomes=treated_outcomes,
            control_sampling_propensity_scores=control_propensities,
            treated_sampling_propensity_scores=treated_propensities,
        )

        pate_estimator = TreatmentEffectEstimator(
            control_estimator=SIPWEstimator(
                propensity_scores=control_propensities,
                outcomes=control_outcomes,
            ),
            treated_estimator=SIPWEstimator(
                propensity_scores=treated_propensities,
                outcomes=treated_outcomes,
            ),
        )
        expected_lb, expected_ub = pate_estimator.expanded_confidence_interval(
            alpha=0.1, seed=1234
        )
        actual_lb, actual_ub = estimator.expanded_confidence_interval(
            alpha=0.1, seed=1234
        )
        assert actual_lb == pytest.approx(expected_lb)
        assert actual_ub == pytest.approx(expected_ub)
        assert actual_ub == pytest.approx(expected_ub)


class TestATEEstimator:
    @staticmethod
    def get_data() -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        float,
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        float,
        npt.NDArray[np.float64],
    ]:
        rng = np.random.default_rng(42)
        population_size = 2_000_000
        sample_size = 2_000
        q = 0.5
        sigma2 = 0.01 * q * (1 - q)
        s = q * (1 - q) / sigma2 - 1
        alpha = q * s
        beta = (1 - q) * s
        propensities = rng.beta(alpha, beta, size=sample_size)
        assignments = rng.binomial(1, propensities, size=sample_size)
        y0 = rng.normal(0.0, 1.0, size=sample_size)
        y1 = rng.normal(0.05, 1.0, size=sample_size)
        outcomes = np.where(assignments == 0, y0, y1)
        predicted_outcomes = outcomes + 0.01 * rng.normal(size=sample_size)

        q = sample_size / population_size
        sigma2 = 1e-6 * q * (1 - q)
        s = q * (1 - q) / sigma2 - 1
        alpha = q * s
        beta = (1 - q) * s
        sampling_propensities = rng.beta(alpha, beta, size=sample_size)

        control_mask = assignments == 0
        treated_mask = assignments == 1

        control_propensities = propensities[control_mask]
        control_outcomes = outcomes[control_mask]
        control_predicted_outcomes = predicted_outcomes[control_mask]
        control_mean_predicted_outcome = 0.0
        control_sampling_propensities = sampling_propensities[control_mask]
        treated_propensities = propensities[treated_mask]
        treated_outcomes = outcomes[treated_mask]
        treated_predicted_outcomes = predicted_outcomes[treated_mask]
        treated_mean_predicted_outcome = 0.05
        treated_sampling_propensities = sampling_propensities[treated_mask]

        return (
            control_propensities,
            control_outcomes,
            control_predicted_outcomes,
            control_mean_predicted_outcome,
            control_sampling_propensities,
            treated_propensities,
            treated_outcomes,
            treated_predicted_outcomes,
            treated_mean_predicted_outcome,
            treated_sampling_propensities,
        )

    @staticmethod
    def test_point_estimate() -> None:
        """Test point estimate."""
        (
            control_propensities,
            control_outcomes,
            control_predicted_outcomes,
            control_mean_predicted_outcome,
            control_sampling_propensities,
            treated_propensities,
            treated_outcomes,
            treated_predicted_outcomes,
            treated_mean_predicted_outcome,
            treated_sampling_propensities,
        ) = TestATEEstimator.get_data()

        estimator = ATEEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
        )
        # Data simulates an ATE of 0.05
        expected = 0.10634476337015086

        actual = estimator.point_estimate()
        assert actual == pytest.approx(expected)

        # Test specifying the sampling propensities, but requesting SampleMean
        estimator = ATEEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
            control_sampling_propensity_scores=control_sampling_propensities,
            treated_sampling_propensity_scores=treated_sampling_propensities,
            sampling_estimand_class="SampleMean",
        )

        actual = estimator.point_estimate()
        assert actual == pytest.approx(expected)

        # Test specifying the sampling propensities, with PopulationMean
        estimator = ATEEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
            control_sampling_propensity_scores=control_sampling_propensities,
            treated_sampling_propensity_scores=treated_sampling_propensities,
            sampling_estimand_class="PopulationMean",
        )
        expected = 0.10576562237906262

        actual = estimator.point_estimate()
        assert actual == pytest.approx(expected)

        # Test with SAIPW estimators
        estimator = ATEEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
            estimator_class="SAIPW",
            control_kwargs={
                "predicted_outcomes": control_predicted_outcomes,
                "mean_predicted_outcome": control_mean_predicted_outcome,
            },
            treated_kwargs={
                "predicted_outcomes": treated_predicted_outcomes,
                "mean_predicted_outcome": treated_mean_predicted_outcome,
            },
        )

        # Data simulates an ATE of 0.05
        expected = 0.05015618839030087

        actual = estimator.point_estimate()
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_variance() -> None:
        """Test variance."""
        (
            control_propensities,
            control_outcomes,
            control_predicted_outcomes,
            control_mean_predicted_outcome,
            control_sampling_propensities,
            treated_propensities,
            treated_outcomes,
            treated_predicted_outcomes,
            treated_mean_predicted_outcome,
            treated_sampling_propensities,
        ) = TestATEEstimator.get_data()

        estimator = ATEEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
            estimator_class="SIPW",
        )
        # Data simulates variance of 1, sample size in each experiment group of ~1000,
        # so variance is ~ 1/1000 + 1/1000 = 2/1000.
        expected = 0.0020067844455491065

        actual = estimator.variance()
        assert actual == pytest.approx(expected)

        # Test specifying the sampling propensities, but requesting SampleMean
        estimator = ATEEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
            control_sampling_propensity_scores=control_sampling_propensities,
            treated_sampling_propensity_scores=treated_sampling_propensities,
            sampling_estimand_class="SampleMean",
        )

        actual = estimator.variance()
        assert actual == pytest.approx(expected)

        # Test with SAIPW estimators
        estimator = ATEEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
            estimator_class="SAIPW",
            control_kwargs={
                "predicted_outcomes": control_predicted_outcomes,
                "mean_predicted_outcome": control_mean_predicted_outcome,
            },
            treated_kwargs={
                "predicted_outcomes": treated_predicted_outcomes,
                "mean_predicted_outcome": treated_mean_predicted_outcome,
            },
        )

        expected = 1.995112125278196e-07

        actual = estimator.variance()
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_pvalue() -> None:
        """Test p-value calculation."""
        (
            control_propensities,
            control_outcomes,
            control_predicted_outcomes,
            control_mean_predicted_outcome,
            control_sampling_propensities,
            treated_propensities,
            treated_outcomes,
            treated_predicted_outcomes,
            treated_mean_predicted_outcome,
            treated_sampling_propensities,
        ) = TestATEEstimator.get_data()

        estimator = ATEEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
        )

        expected = 0.017600450212022885

        actual = estimator.pvalue()
        assert actual == pytest.approx(expected)

        # Test one-sided p-values
        actual = estimator.pvalue(alternative="greater")
        # The point estimate is positive (and fairly large relative to the variance),
        # which provides evidence in favor of a positive treatment effect. Thus, when
        # testing for evidence in favor of a treatment effect greater than 0, we should
        # get a p-value less than 0.5, and half the two-sided interval.
        assert actual < 0.5
        assert actual == pytest.approx(0.5 * expected)

        actual = estimator.pvalue(alternative="less")
        # If for some reason, we wanted to prove the treatment were harmful, this is
        # what we would do. But in this case, the evidence points in favor of a
        # beneficial treatment, so the evidence in favor of a harmful treatment is quite
        # weak, as represented by a p-value > 0.5.
        assert actual > 0.5

        # Test specifying the sampling propensities, but requesting SampleMean
        estimator = ATEEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
            control_sampling_propensity_scores=control_sampling_propensities,
            treated_sampling_propensity_scores=treated_sampling_propensities,
            sampling_estimand_class="SampleMean",
        )

        actual = estimator.pvalue()
        assert actual == pytest.approx(expected)

        # Test with SAIPW estimators
        estimator = ATEEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
            estimator_class="SAIPW",
            control_kwargs={
                "predicted_outcomes": control_predicted_outcomes,
                "mean_predicted_outcome": control_mean_predicted_outcome,
            },
            treated_kwargs={
                "predicted_outcomes": treated_predicted_outcomes,
                "mean_predicted_outcome": treated_mean_predicted_outcome,
            },
        )

        # Because our outcome model is artificially amazing, the variance is negligible
        # and we get a p-value of nearly exactly 0.
        expected = 0.0

        actual = estimator.pvalue()
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_confidence_interval() -> None:
        """Test confidence interval."""
        (
            control_propensities,
            control_outcomes,
            control_predicted_outcomes,
            control_mean_predicted_outcome,
            control_sampling_propensities,
            treated_propensities,
            treated_outcomes,
            treated_predicted_outcomes,
            treated_mean_predicted_outcome,
            treated_sampling_propensities,
        ) = TestATEEstimator.get_data()

        estimator = ATEEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
        )
        # Data simulates an ATE of 0.05
        lb_expected = 0.03266001243458834
        ub_expected = 0.1800295143057134

        lb_actual, ub_actual = estimator.confidence_interval(alpha=0.1)
        print(lb_actual, ub_actual)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

        # Test one-sided intervals
        lb_actual, ub_actual = estimator.confidence_interval(
            alpha=0.05, alternative="less"
        )
        print(lb_actual, ub_actual)
        assert lb_actual == -np.inf
        assert ub_actual == pytest.approx(ub_expected)

        lb_actual, ub_actual = estimator.confidence_interval(
            alpha=0.05, alternative="greater"
        )
        print(lb_actual, ub_actual)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == np.inf

        # Test specifying the sampling propensities, but requesting SampleMean
        estimator = ATEEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
            control_sampling_propensity_scores=control_sampling_propensities,
            treated_sampling_propensity_scores=treated_sampling_propensities,
            sampling_estimand_class="SampleMean",
        )

        lb_actual, ub_actual = estimator.confidence_interval(alpha=0.1)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

        # Test with SAIPW estimators
        estimator = ATEEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
            estimator_class="SAIPW",
            control_kwargs={
                "predicted_outcomes": control_predicted_outcomes,
                "mean_predicted_outcome": control_mean_predicted_outcome,
            },
            treated_kwargs={
                "predicted_outcomes": treated_predicted_outcomes,
                "mean_predicted_outcome": treated_mean_predicted_outcome,
            },
        )
        # Data simulates an ATE of 0.05
        lb_expected = 0.049421486916861976
        ub_expected = 0.05089088986373977

        lb_actual, ub_actual = estimator.confidence_interval(alpha=0.1)
        print(lb_actual, ub_actual)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

    @staticmethod
    def test_sensitivity_analysis() -> None:
        """Test sensitivity analysis."""
        (
            control_propensities,
            control_outcomes,
            control_predicted_outcomes,
            control_mean_predicted_outcome,
            control_sampling_propensities,
            treated_propensities,
            treated_outcomes,
            treated_predicted_outcomes,
            treated_mean_predicted_outcome,
            treated_sampling_propensities,
        ) = TestATEEstimator.get_data()

        estimator = ATEEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
        )
        gamma = 6

        lb_expected = -0.604901076552984
        ub_expected = 0.8190572443345536

        lb_actual, ub_actual = estimator.sensitivity_analysis(gamma=gamma)
        print(lb_actual, ub_actual)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

        # Test specifying the sampling propensities, but requesting SampleMean
        estimator = ATEEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
            control_sampling_propensity_scores=control_sampling_propensities,
            treated_sampling_propensity_scores=treated_sampling_propensities,
            sampling_estimand_class="SampleMean",
        )

        lb_actual, ub_actual = estimator.sensitivity_analysis(gamma=gamma)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

        # Test with SAIPW estimators
        estimator = ATEEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
            estimator_class="SAIPW",
            control_kwargs={
                "predicted_outcomes": control_predicted_outcomes,
                "mean_predicted_outcome": control_mean_predicted_outcome,
            },
            treated_kwargs={
                "predicted_outcomes": treated_predicted_outcomes,
                "mean_predicted_outcome": treated_mean_predicted_outcome,
            },
        )

        lb_expected = 0.043027033013628514
        ub_expected = 0.05733341412893974

        lb_actual, ub_actual = estimator.sensitivity_analysis(gamma=gamma)
        print(lb_actual, ub_actual)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

    @staticmethod
    def test_expanded_confidence_interval() -> None:
        """Test expanded confidence interval."""
        (
            control_propensities,
            control_outcomes,
            control_predicted_outcomes,
            control_mean_predicted_outcome,
            control_sampling_propensities,
            treated_propensities,
            treated_outcomes,
            treated_predicted_outcomes,
            treated_mean_predicted_outcome,
            treated_sampling_propensities,
        ) = TestATEEstimator.get_data()

        estimator = ATEEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
        )
        lb_expected = -0.6784724957263869
        ub_expected = 0.8917726898653108

        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.1, gamma=6.0, seed=42
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

        # Test one-sided intervals
        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.05, gamma=6.0, alternative="less", seed=42
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        assert lb_actual == -np.inf
        assert ub_actual == pytest.approx(ub_expected)

        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.05, gamma=6.0, alternative="greater", seed=42
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == np.inf

        # Test specifying the sampling propensities, but requesting SampleMean
        estimator = ATEEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
            control_sampling_propensity_scores=control_sampling_propensities,
            treated_sampling_propensity_scores=treated_sampling_propensities,
            sampling_estimand_class="SampleMean",
        )

        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.1, gamma=6.0, seed=42
        )
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

        # Test specifying the sampling propensities, but requesting PopulationMean
        estimator = ATEEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
            control_sampling_propensity_scores=control_sampling_propensities,
            treated_sampling_propensity_scores=treated_sampling_propensities,
            sampling_estimand_class="PopulationMean",
        )
        lb_expected = -2.0721958774613065
        ub_expected = 2.2940124004956144

        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.1, gamma=6.0, seed=42
        )
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

        # Test with SAIPW estimators
        estimator = ATEEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
            estimator_class="SAIPW",
            control_kwargs={
                "predicted_outcomes": control_predicted_outcomes,
                "mean_predicted_outcome": control_mean_predicted_outcome,
            },
            treated_kwargs={
                "predicted_outcomes": treated_predicted_outcomes,
                "mean_predicted_outcome": treated_mean_predicted_outcome,
            },
        )
        lb_expected = 0.04230439882983481
        ub_expected = 0.05807741535242331

        start_time = time.time()
        lb_actual, ub_actual = estimator.expanded_confidence_interval(
            alpha=0.1, gamma=6.0, seed=42
        )
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.03f} seconds")
        print(lb_actual, ub_actual)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

    @staticmethod
    def test_plot_sensitivity() -> None:
        """Test sensitivity analysis."""
        (
            control_propensities,
            control_outcomes,
            control_predicted_outcomes,
            control_mean_predicted_outcome,
            control_sampling_propensities,
            treated_propensities,
            treated_outcomes,
            treated_predicted_outcomes,
            treated_mean_predicted_outcome,
            treated_sampling_propensities,
        ) = TestATEEstimator.get_data()

        estimator = ATEEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
            estimator_class="SAIPW",
            control_kwargs={
                "predicted_outcomes": control_predicted_outcomes,
                "mean_predicted_outcome": control_mean_predicted_outcome,
            },
            treated_kwargs={
                "predicted_outcomes": treated_predicted_outcomes,
                "mean_predicted_outcome": treated_mean_predicted_outcome,
            },
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
            ylabel="ATE",
            B=100,
            axis_label_size=18,
            tick_label_size=14,
            ytick_format=".02%",
        )
        # plt.show()
        assert set(df.columns) == set(expected_columns)
        assert len(df) == 50
        plt.close("all")


class TestATTEstimator:
    @staticmethod
    def test_point_estimate() -> None:
        """Test point estimate."""
        (
            control_propensities,
            control_outcomes,
            control_predicted_outcomes,
            control_mean_predicted_outcome,
            control_sampling_propensities,
            treated_propensities,
            treated_outcomes,
            treated_predicted_outcomes,
            treated_mean_predicted_outcome,
            treated_sampling_propensities,
        ) = TestATEEstimator.get_data()

        estimator: TreatmentEffectEstimator = ATTEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_outcomes=treated_outcomes,
        )
        # Data simulates an ATE of 0.05
        expected = 0.10130356555998879

        actual = estimator.point_estimate()
        assert actual == pytest.approx(expected)

        # Test that ATC == -ATT when we interchange treated/control units
        estimator = ATCEstimator(
            treated_propensity_scores=np.ones_like(control_propensities)
            - control_propensities,
            treated_outcomes=control_outcomes,
            control_outcomes=treated_outcomes,
        )
        actual = -estimator.point_estimate()
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_variance() -> None:
        """Test variance."""
        (
            control_propensities,
            control_outcomes,
            control_predicted_outcomes,
            control_mean_predicted_outcome,
            control_sampling_propensities,
            treated_propensities,
            treated_outcomes,
            treated_predicted_outcomes,
            treated_mean_predicted_outcome,
            treated_sampling_propensities,
        ) = TestATEEstimator.get_data()

        estimator = ATTEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_outcomes=treated_outcomes,
            estimator_class="SIPW",
        )
        # Data simulates variance of 1, sample size in each experiment group of ~1000,
        # so variance is ~ 1/1000 + 1/1000 = 2/1000.
        expected = 0.0020332556235972184

        actual = estimator.variance()
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_sensitivity_analysis() -> None:
        """Test sensitivity analysis."""
        (
            control_propensities,
            control_outcomes,
            control_predicted_outcomes,
            control_mean_predicted_outcome,
            control_sampling_propensities,
            treated_propensities,
            treated_outcomes,
            treated_predicted_outcomes,
            treated_mean_predicted_outcome,
            treated_sampling_propensities,
        ) = TestATEEstimator.get_data()

        estimator: TreatmentEffectEstimator = ATTEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_outcomes=treated_outcomes,
        )
        gamma = 6

        lb_expected = -0.628248664522913
        ub_expected = 0.7954444033051249

        lb_actual, ub_actual = estimator.sensitivity_analysis(gamma=gamma)
        print(lb_actual, ub_actual)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

        # Test that ATC == -ATT when we interchange treated/control units
        estimator = ATCEstimator(
            treated_propensity_scores=np.ones_like(control_propensities)
            - control_propensities,
            treated_outcomes=control_outcomes,
            control_outcomes=treated_outcomes,
        )
        lb_actual, ub_actual = estimator.sensitivity_analysis(gamma=gamma)
        print(lb_actual, ub_actual)
        assert lb_actual == pytest.approx(-ub_expected)
        assert ub_actual == pytest.approx(-lb_expected)

    @staticmethod
    def test_plot_sensitivity() -> None:
        """Test sensitivity analysis."""
        (
            control_propensities,
            control_outcomes,
            control_predicted_outcomes,
            control_mean_predicted_outcome,
            control_sampling_propensities,
            treated_propensities,
            treated_outcomes,
            treated_predicted_outcomes,
            treated_mean_predicted_outcome,
            treated_sampling_propensities,
        ) = TestATEEstimator.get_data()

        estimator = ATTEstimator(
            control_propensity_scores=control_propensities,
            control_outcomes=control_outcomes,
            treated_outcomes=treated_outcomes,
            estimator_class="SAIPW",
            control_kwargs={
                "predicted_outcomes": control_predicted_outcomes,
                "mean_predicted_outcome": control_mean_predicted_outcome,
            },
            treated_kwargs={
                "predicted_outcomes": treated_predicted_outcomes,
                "mean_predicted_outcome": np.mean(treated_predicted_outcomes),
            },
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
            ylabel="ATT",
            B=100,
            axis_label_size=18,
            tick_label_size=14,
            ytick_format=".02%",
        )
        # plt.show()
        assert set(df.columns) == set(expected_columns)
        assert len(df) == 50
        plt.close("all")


class TestATCEstimator:
    @staticmethod
    def test_point_estimate() -> None:
        """Test point estimate."""
        (
            control_propensities,
            control_outcomes,
            control_predicted_outcomes,
            control_mean_predicted_outcome,
            control_sampling_propensities,
            treated_propensities,
            treated_outcomes,
            treated_predicted_outcomes,
            treated_mean_predicted_outcome,
            treated_sampling_propensities,
        ) = TestATEEstimator.get_data()

        estimator = ATCEstimator(
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
        )
        # Data simulates an ATE of 0.05
        expected = 0.11139202188516385

        actual = estimator.point_estimate()
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_variance() -> None:
        """Test variance."""
        (
            control_propensities,
            control_outcomes,
            control_predicted_outcomes,
            control_mean_predicted_outcome,
            control_sampling_propensities,
            treated_propensities,
            treated_outcomes,
            treated_predicted_outcomes,
            treated_mean_predicted_outcome,
            treated_sampling_propensities,
        ) = TestATEEstimator.get_data()

        estimator = ATCEstimator(
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
            estimator_class="SIPW",
        )
        # Data simulates variance of 1, sample size in each experiment group of ~1000,
        # so variance is ~ 1/1000 + 1/1000 = 2/1000.
        expected = 0.0020194493137473046

        actual = estimator.variance()
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_sensitivity_analysis() -> None:
        """Test sensitivity analysis."""
        (
            control_propensities,
            control_outcomes,
            control_predicted_outcomes,
            control_mean_predicted_outcome,
            control_sampling_propensities,
            treated_propensities,
            treated_outcomes,
            treated_predicted_outcomes,
            treated_mean_predicted_outcome,
            treated_sampling_propensities,
        ) = TestATEEstimator.get_data()

        estimator = ATCEstimator(
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
        )
        gamma = 6

        lb_expected = -0.5708995002562959
        ub_expected = 0.8316749638939132

        lb_actual, ub_actual = estimator.sensitivity_analysis(gamma=gamma)
        print(lb_actual, ub_actual)
        assert lb_actual == pytest.approx(lb_expected)
        assert ub_actual == pytest.approx(ub_expected)

    @staticmethod
    def test_plot_sensitivity() -> None:
        """Test sensitivity analysis."""
        (
            control_propensities,
            control_outcomes,
            control_predicted_outcomes,
            control_mean_predicted_outcome,
            control_sampling_propensities,
            treated_propensities,
            treated_outcomes,
            treated_predicted_outcomes,
            treated_mean_predicted_outcome,
            treated_sampling_propensities,
        ) = TestATEEstimator.get_data()

        estimator = ATCEstimator(
            control_outcomes=control_outcomes,
            treated_propensity_scores=treated_propensities,
            treated_outcomes=treated_outcomes,
            estimator_class="SAIPW",
            control_kwargs={
                "predicted_outcomes": control_predicted_outcomes,
                "mean_predicted_outcome": np.mean(control_predicted_outcomes),
            },
            treated_kwargs={
                "predicted_outcomes": treated_predicted_outcomes,
                "mean_predicted_outcome": treated_mean_predicted_outcome,
            },
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
            ylabel="ATC",
            B=100,
            axis_label_size=18,
            tick_label_size=14,
            ytick_format=".02%",
        )
        # plt.show()
        assert set(df.columns) == set(expected_columns)
        assert len(df) == 50
        plt.close("all")


class TestTreatmentEffectRatioEstimator:
    @staticmethod
    def get_data() -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        rng = np.random.default_rng(42)
        sample_size = 2000
        q = 0.5
        sigma2 = 0.01 * q * (1 - q)
        s = q * (1 - q) / sigma2 - 1
        alpha = q * s
        beta = (1 - q) * s
        propensities = rng.beta(alpha, beta, size=sample_size)
        assignments = rng.binomial(1, propensities, size=sample_size)

        # Numerator outcomes: simulate treatment effect ~0.05
        y0_num = rng.normal(0.0, 1.0, size=sample_size)
        y1_num = rng.normal(0.05, 1.0, size=sample_size)
        numerator_outcomes = np.where(assignments == 0, y0_num, y1_num)

        # Denominator outcomes: simulate treatment effect ~0.1
        y0_den = rng.normal(1.0, 1.0, size=sample_size)
        y1_den = rng.normal(1.1, 1.0, size=sample_size)
        denominator_outcomes = np.where(assignments == 0, y0_den, y1_den)

        control_mask = assignments == 0
        treated_mask = assignments == 1

        return (
            propensities[control_mask],
            propensities[treated_mask],
            numerator_outcomes[control_mask],
            numerator_outcomes[treated_mask],
            denominator_outcomes[control_mask],
            denominator_outcomes[treated_mask],
        )

    @staticmethod
    def test_point_estimate() -> None:
        (
            control_propensities,
            treated_propensities,
            numerator_control_outcomes,
            numerator_treated_outcomes,
            denominator_control_outcomes,
            denominator_treated_outcomes,
        ) = TestTreatmentEffectRatioEstimator.get_data()

        estimator = TreatmentEffectRatioEstimator(
            control_propensity_scores=control_propensities,
            treated_propensity_scores=treated_propensities,
            numerator_control_outcomes=numerator_control_outcomes,
            numerator_treated_outcomes=numerator_treated_outcomes,
            denominator_control_outcomes=denominator_control_outcomes,
            denominator_treated_outcomes=denominator_treated_outcomes,
            mean_estimator_class="SIPW",
            treatment_effect_estimator_class="ATE",
        )

        # Calculate expected ratio of treatment effects manually
        numerator_estimator = ATEEstimator(
            control_propensity_scores=control_propensities,
            treated_propensity_scores=treated_propensities,
            control_outcomes=numerator_control_outcomes,
            treated_outcomes=numerator_treated_outcomes,
            estimator_class="SIPW",
        )

        denominator_estimator = ATEEstimator(
            control_propensity_scores=control_propensities,
            treated_propensity_scores=treated_propensities,
            control_outcomes=denominator_control_outcomes,
            treated_outcomes=denominator_treated_outcomes,
            estimator_class="SIPW",
        )

        expected = (
            numerator_estimator.point_estimate()
            / denominator_estimator.point_estimate()
        )
        actual = estimator.point_estimate()
        assert actual == pytest.approx(expected, rel=1e-1)

    @staticmethod
    def test_variance() -> None:
        (
            control_propensities,
            treated_propensities,
            numerator_control_outcomes,
            numerator_treated_outcomes,
            denominator_control_outcomes,
            denominator_treated_outcomes,
        ) = TestTreatmentEffectRatioEstimator.get_data()

        estimator = TreatmentEffectRatioEstimator(
            control_propensity_scores=control_propensities,
            treated_propensity_scores=treated_propensities,
            numerator_control_outcomes=numerator_control_outcomes,
            numerator_treated_outcomes=numerator_treated_outcomes,
            denominator_control_outcomes=denominator_control_outcomes,
            denominator_treated_outcomes=denominator_treated_outcomes,
            mean_estimator_class="SIPW",
            treatment_effect_estimator_class="ATE",
        )

        expected = 0.2815034627377356

        actual = estimator.variance()

        # Variance should be positive and finite
        assert actual > 0
        assert np.isfinite(actual)
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_pvalue() -> None:
        (
            control_propensities,
            treated_propensities,
            numerator_control_outcomes,
            numerator_treated_outcomes,
            denominator_control_outcomes,
            denominator_treated_outcomes,
        ) = TestTreatmentEffectRatioEstimator.get_data()

        estimator = TreatmentEffectRatioEstimator(
            control_propensity_scores=control_propensities,
            treated_propensity_scores=treated_propensities,
            numerator_control_outcomes=numerator_control_outcomes,
            numerator_treated_outcomes=numerator_treated_outcomes,
            denominator_control_outcomes=denominator_control_outcomes,
            denominator_treated_outcomes=denominator_treated_outcomes,
            mean_estimator_class="SIPW",
            treatment_effect_estimator_class="ATE",
        )

        pe = estimator.point_estimate()
        se = np.sqrt(estimator.variance())
        null = pe - 0.05
        t = (pe - null) / se

        # two-sided
        expected_p = min(1.0, 2 * min(stats.norm.sf(t), stats.norm.cdf(t)))
        assert estimator.pvalue(
            null_value=null, alternative="two-sided"
        ) == pytest.approx(expected_p, rel=1e-8)

        # greater
        expected_pg = stats.norm.sf(t)
        assert estimator.pvalue(
            null_value=null, alternative="greater"
        ) == pytest.approx(expected_pg, rel=1e-8)

        # less
        expected_pl = stats.norm.cdf(t)
        assert estimator.pvalue(null_value=null, alternative="less") == pytest.approx(
            expected_pl, rel=1e-8
        )

    @staticmethod
    def test_confidence_interval() -> None:
        (
            control_propensities,
            treated_propensities,
            numerator_control_outcomes,
            numerator_treated_outcomes,
            denominator_control_outcomes,
            denominator_treated_outcomes,
        ) = TestTreatmentEffectRatioEstimator.get_data()

        estimator = TreatmentEffectRatioEstimator(
            control_propensity_scores=control_propensities,
            treated_propensity_scores=treated_propensities,
            numerator_control_outcomes=numerator_control_outcomes,
            numerator_treated_outcomes=numerator_treated_outcomes,
            denominator_control_outcomes=denominator_control_outcomes,
            denominator_treated_outcomes=denominator_treated_outcomes,
            mean_estimator_class="SIPW",
            treatment_effect_estimator_class="ATE",
        )

        pe = estimator.point_estimate()
        se = np.sqrt(estimator.variance())

        # Two-sided 90%
        lower, upper = estimator.confidence_interval(
            alpha=0.10, alternative="two-sided"
        )
        z = stats.norm.isf(0.10 / 2)
        expected_lb, expected_ub = pe - z * se, pe + z * se
        assert lower == pytest.approx(expected_lb, rel=1e-12)
        assert upper == pytest.approx(expected_ub, rel=1e-12)

        # Greater
        lower, upper = estimator.confidence_interval(alpha=0.05, alternative="greater")
        z = stats.norm.isf(0.05)
        expected_ub = pe + z * se
        assert lower == pytest.approx(pe - z * se, rel=1e-12)
        assert upper == np.inf

        # Less
        lower, upper = estimator.confidence_interval(alpha=0.05, alternative="less")
        assert upper == pytest.approx(pe + z * se, rel=1e-12)
        assert lower == -np.inf
