"""Test LinearCombinationEstimator."""

import numpy as np
import numpy.typing as npt
import pytest

from pyrake.estimation.linear_combination import LinearCombinationEstimator
from pyrake.estimation.population import SIPWEstimator
from pyrake.estimation.treatment_effects import TreatmentEffectEstimator


def make_estimator(
    n: int = 500,
    seed: int = 0,
) -> tuple[SIPWEstimator, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return a SIPWEstimator together with the raw propensities and outcomes."""
    rng = np.random.default_rng(seed)
    propensities = rng.beta(2, 5, size=n)
    outcomes = rng.normal(0.5, 1.0, size=n)
    return SIPWEstimator(propensities, outcomes), propensities, outcomes


class TestLinearCombinationEstimator:
    @staticmethod
    def test_point_estimate_basic() -> None:
        """Affine + weighted sum equals manual computation."""
        est1, _, _ = make_estimator(n=400, seed=1)
        est2, _, _ = make_estimator(n=300, seed=2)
        c1, c2, affine = 2.0, -0.5, 0.1

        lce = LinearCombinationEstimator(terms=[(c1, est1), (c2, est2)], affine=affine)

        expected = affine + c1 * est1.point_estimate() + c2 * est2.point_estimate()
        assert lce.point_estimate() == pytest.approx(expected)

    @staticmethod
    def test_point_estimate_affine_only() -> None:
        """A LinearCombinationEstimator with no terms returns the affine value."""
        lce = LinearCombinationEstimator(terms=[], affine=0.42)
        assert lce.point_estimate() == pytest.approx(0.42)

    @staticmethod
    def test_variance_independence() -> None:
        """Variance equals sum of c_i^2 * var_i (independence assumption)."""
        est1, _, _ = make_estimator(n=400, seed=1)
        est2, _, _ = make_estimator(n=300, seed=2)
        c1, c2 = 3.0, -1.5

        lce = LinearCombinationEstimator(terms=[(c1, est1), (c2, est2)])

        expected = c1**2 * est1.variance() + c2**2 * est2.variance()
        assert lce.variance() == pytest.approx(expected)

    @staticmethod
    def test_variance_affine_only() -> None:
        """Affine-only estimator has zero variance."""
        lce = LinearCombinationEstimator(terms=[], affine=1.0)
        assert lce.variance() == pytest.approx(0.0)

    @staticmethod
    def test_sensitivity_analysis_positive_coefficients() -> None:
        """Positive coefficient: lb uses component lb, ub uses component ub."""
        est, _, _ = make_estimator(n=500, seed=3)
        c = 2.0
        lce = LinearCombinationEstimator(terms=[(c, est)])

        e_lb, e_ub = est.sensitivity_analysis(gamma=2.0)
        lb, ub = lce.sensitivity_analysis(gamma=2.0)

        assert lb == pytest.approx(c * e_lb)
        assert ub == pytest.approx(c * e_ub)
        assert lb <= ub

    @staticmethod
    def test_sensitivity_analysis_negative_coefficient() -> None:
        """Negative coefficient flips which component bound feeds which output bound."""
        est, _, _ = make_estimator(n=500, seed=3)
        c = -2.0
        lce = LinearCombinationEstimator(terms=[(c, est)])

        e_lb, e_ub = est.sensitivity_analysis(gamma=2.0)
        lb, ub = lce.sensitivity_analysis(gamma=2.0)

        # c < 0: lb = c * e_ub, ub = c * e_lb
        assert lb == pytest.approx(c * e_ub)
        assert ub == pytest.approx(c * e_lb)
        assert lb <= ub

    @staticmethod
    def test_sensitivity_analysis_affine_shifts_both_bounds() -> None:
        """Affine term shifts both bounds by the same constant."""
        est, _, _ = make_estimator(n=500, seed=4)
        offset = 0.25

        lce_base = LinearCombinationEstimator(terms=[(1.0, est)])
        lce_shifted = LinearCombinationEstimator(terms=[(1.0, est)], affine=offset)

        lb_base, ub_base = lce_base.sensitivity_analysis(gamma=3.0)
        lb_shifted, ub_shifted = lce_shifted.sensitivity_analysis(gamma=3.0)

        assert lb_shifted == pytest.approx(lb_base + offset)
        assert ub_shifted == pytest.approx(ub_base + offset)

    @staticmethod
    def test_sensitivity_analysis_mixed_signs() -> None:
        """Mixed-sign coefficients combine correctly."""
        est1, _, _ = make_estimator(n=400, seed=5)
        est2, _, _ = make_estimator(n=300, seed=6)
        c1, c2 = 1.5, -0.8

        lce = LinearCombinationEstimator(terms=[(c1, est1), (c2, est2)], affine=0.05)

        lb1, ub1 = est1.sensitivity_analysis(gamma=2.0)
        lb2, ub2 = est2.sensitivity_analysis(gamma=2.0)
        lb, ub = lce.sensitivity_analysis(gamma=2.0)

        # c1 >= 0: uses lb1 / ub1 normally
        # c2 < 0: flips lb2 / ub2
        assert lb == pytest.approx(0.05 + c1 * lb1 + c2 * ub2)
        assert ub == pytest.approx(0.05 + c1 * ub1 + c2 * lb2)
        assert lb <= ub

    @staticmethod
    def test_reduces_to_treatment_effect_estimator() -> None:
        """LCE with coefficients (1, -1) is equivalent to TreatmentEffectEstimator."""
        est_control, _, _ = make_estimator(n=400, seed=7)
        est_treated, _, _ = make_estimator(n=300, seed=8)

        te = TreatmentEffectEstimator(
            control_estimator=est_control,
            treated_estimator=est_treated,
        )
        lce = LinearCombinationEstimator(
            terms=[(1.0, est_treated), (-1.0, est_control)]
        )

        assert lce.point_estimate() == pytest.approx(te.point_estimate())
        assert lce.variance() == pytest.approx(te.variance())

        te_lb, te_ub = te.sensitivity_analysis(gamma=3.0)
        lce_lb, lce_ub = lce.sensitivity_analysis(gamma=3.0)
        assert lce_lb == pytest.approx(te_lb)
        assert lce_ub == pytest.approx(te_ub)

    @staticmethod
    def test_nested_linear_combination() -> None:
        """LCE whose components include another LCE."""
        est1, _, _ = make_estimator(n=300, seed=9)
        est2, _, _ = make_estimator(n=300, seed=10)
        est3, _, _ = make_estimator(n=300, seed=11)

        inner = LinearCombinationEstimator(
            terms=[(1.0, est1), (-1.0, est2)], affine=0.1
        )
        outer = LinearCombinationEstimator(terms=[(0.5, inner), (1.0, est3)])

        expected_pe = (
            0.5 * (0.1 + est1.point_estimate() - est2.point_estimate())
            + est3.point_estimate()
        )
        assert outer.point_estimate() == pytest.approx(expected_pe)

        # Sensitivity analysis should propagate through the nesting
        inner_lb, inner_ub = inner.sensitivity_analysis(gamma=2.0)
        est3_lb, est3_ub = est3.sensitivity_analysis(gamma=2.0)
        outer_lb, outer_ub = outer.sensitivity_analysis(gamma=2.0)

        assert outer_lb == pytest.approx(0.5 * inner_lb + est3_lb)
        assert outer_ub == pytest.approx(0.5 * inner_ub + est3_ub)

    @staticmethod
    def test_with_treatment_effect_component() -> None:
        """LCE can take a TreatmentEffectEstimator as a component."""
        est_control, _, _ = make_estimator(n=400, seed=12)
        est_treated, _, _ = make_estimator(n=400, seed=13)
        est_baseline, _, _ = make_estimator(n=400, seed=14)

        te = TreatmentEffectEstimator(
            control_estimator=est_control,
            treated_estimator=est_treated,
        )
        lce = LinearCombinationEstimator(
            terms=[(1.0, te), (-0.1, est_baseline)],
            affine=0.0,
        )

        expected_pe = te.point_estimate() - 0.1 * est_baseline.point_estimate()
        assert lce.point_estimate() == pytest.approx(expected_pe)

        te_lb, te_ub = te.sensitivity_analysis(gamma=2.0)
        bl_lb, bl_ub = est_baseline.sensitivity_analysis(gamma=2.0)
        lce_lb, lce_ub = lce.sensitivity_analysis(gamma=2.0)

        # coeff on te is +1 (positive), coeff on est_baseline is -0.1 (negative)
        assert lce_lb == pytest.approx(te_lb + (-0.1) * bl_ub)
        assert lce_ub == pytest.approx(te_ub + (-0.1) * bl_lb)

    @staticmethod
    def test_expanded_confidence_interval_two_sided() -> None:
        """ECI is wider than the standard CI and has lb <= pe <= ub (smoke test)."""
        est1, _, _ = make_estimator(n=400, seed=15)
        est2, _, _ = make_estimator(n=400, seed=16)
        lce = LinearCombinationEstimator(terms=[(1.0, est1), (-1.0, est2)])

        ci_lb, ci_ub = lce.confidence_interval(alpha=0.10)
        eci_lb, eci_ub = lce.expanded_confidence_interval(
            alpha=0.10, gamma=2.0, B=200, seed=42
        )

        assert eci_lb <= ci_lb + 1e-6
        assert eci_ub >= ci_ub - 1e-6
        assert eci_lb < eci_ub

    @staticmethod
    def test_expanded_confidence_interval_gamma_1() -> None:
        """gamma=1 ECI falls back to the standard CI."""
        est, _, _ = make_estimator(n=400, seed=17)
        lce = LinearCombinationEstimator(terms=[(1.0, est)])

        ci = lce.confidence_interval(alpha=0.10)
        eci = lce.expanded_confidence_interval(alpha=0.10, gamma=1.0, B=200, seed=0)

        assert eci[0] == pytest.approx(ci[0])
        assert eci[1] == pytest.approx(ci[1])

    @staticmethod
    def test_expanded_confidence_interval_no_terms() -> None:
        """ECI for affine-only LCE returns a degenerate interval (no uncertainty)."""
        lce = LinearCombinationEstimator(terms=[], affine=0.5)
        eci_lb, eci_ub = lce.expanded_confidence_interval(
            alpha=0.10, gamma=3.0, B=100, seed=0
        )
        # Variance is zero, so CI and ECI both collapse to (0.5, 0.5)
        assert eci_lb == pytest.approx(0.5)
        assert eci_ub == pytest.approx(0.5)

    @staticmethod
    def test_expanded_confidence_interval_with_te_component() -> None:
        """ECI works when a component is a TreatmentEffectEstimator (smoke test)."""
        est_control, _, _ = make_estimator(n=300, seed=18)
        est_treated, _, _ = make_estimator(n=300, seed=19)
        te = TreatmentEffectEstimator(
            control_estimator=est_control,
            treated_estimator=est_treated,
        )
        lce = LinearCombinationEstimator(terms=[(1.0, te)])
        lb, ub = lce.expanded_confidence_interval(alpha=0.10, gamma=2.0, B=100, seed=42)
        assert lb < ub

    @staticmethod
    def test_expanded_confidence_interval_nested() -> None:
        """ECI works for a nested LinearCombinationEstimator (smoke test)."""
        est1, _, _ = make_estimator(n=300, seed=20)
        est2, _, _ = make_estimator(n=300, seed=21)
        est3, _, _ = make_estimator(n=300, seed=22)

        inner = LinearCombinationEstimator(terms=[(1.0, est1), (-1.0, est2)])
        outer = LinearCombinationEstimator(
            terms=[(0.5, inner), (1.0, est3)], affine=0.1
        )

        lb, ub = outer.expanded_confidence_interval(
            alpha=0.10, gamma=2.0, B=100, seed=42
        )
        assert lb < ub

    @staticmethod
    def test_adjusted_pvalue_gamma_1_fallback() -> None:
        """gamma=1 adjusted p-value equals the standard p-value."""
        est1, _, _ = make_estimator(n=400, seed=15)
        est2, _, _ = make_estimator(n=400, seed=16)
        lce = LinearCombinationEstimator(terms=[(1.0, est1), (-1.0, est2)])

        null_value = lce.point_estimate() + 0.1
        std_p = lce.pvalue(null_value=null_value)
        adj_p = lce.adjusted_pvalue(null_value=null_value, gamma=1.0)
        assert adj_p == pytest.approx(std_p)

    @staticmethod
    def test_adjusted_pvalue_ge_standard() -> None:
        """Adjusted p-value with gamma > 1 is >= the standard p-value."""
        est1, _, _ = make_estimator(n=400, seed=15)
        est2, _, _ = make_estimator(n=400, seed=16)
        lce = LinearCombinationEstimator(terms=[(1.0, est1), (-1.0, est2)])

        null_value = lce.point_estimate() + 0.1
        std_p = lce.pvalue(null_value=null_value)
        adj_p = lce.adjusted_pvalue(null_value=null_value, gamma=2.0, B=200, seed=42)
        assert adj_p >= std_p

    @staticmethod
    def test_adjusted_pvalue_affine_only() -> None:
        """Affine-only LCE: adjusted p-value is determined entirely by the point estimate."""
        lce = LinearCombinationEstimator(terms=[], affine=0.42)

        # null above the point estimate: bootstrap lb=pe <= null, so p=1 for "greater"
        p_greater = lce.adjusted_pvalue(
            null_value=0.50, gamma=3.0, alternative="greater", B=50, seed=42
        )
        assert p_greater == pytest.approx(1.0)

        # null below the point estimate: bootstrap lb=pe > null, so p=0 for "greater"
        p_greater_below = lce.adjusted_pvalue(
            null_value=0.30, gamma=3.0, alternative="greater", B=50, seed=42
        )
        assert p_greater_below == pytest.approx(0.0)

    @staticmethod
    def test_expanded_confidence_interval_no_bootstrap_gamma_1() -> None:
        """bootstrap=False, gamma=1 falls back to the standard CI."""
        est, _, _ = make_estimator(n=400, seed=17)
        lce = LinearCombinationEstimator(terms=[(1.0, est)])

        ci = lce.confidence_interval(alpha=0.10)
        eci = lce.expanded_confidence_interval(alpha=0.10, gamma=1.0, bootstrap=False)

        assert eci[0] == pytest.approx(ci[0])
        assert eci[1] == pytest.approx(ci[1])

    @staticmethod
    def test_expanded_confidence_interval_no_bootstrap_wider_than_ci() -> None:
        """bootstrap=False ECI with gamma > 1 is wider than the standard CI."""
        est1, _, _ = make_estimator(n=400, seed=15)
        est2, _, _ = make_estimator(n=400, seed=16)
        lce = LinearCombinationEstimator(terms=[(1.0, est1), (-1.0, est2)])

        ci_lb, ci_ub = lce.confidence_interval(alpha=0.10)
        eci_lb, eci_ub = lce.expanded_confidence_interval(
            alpha=0.10, gamma=2.0, bootstrap=False
        )

        assert eci_lb <= ci_lb + 1e-6
        assert eci_ub >= ci_ub - 1e-6
        assert eci_lb < eci_ub

    @staticmethod
    def test_expanded_confidence_interval_no_bootstrap_formula() -> None:
        """bootstrap=False ECI equals sensitivity bounds +/- z * se_at_worst_case_weights."""
        import math

        from scipy import stats

        est1, _, _ = make_estimator(n=400, seed=15)
        est2, _, _ = make_estimator(n=400, seed=16)
        lce = LinearCombinationEstimator(terms=[(1.0, est1), (-1.0, est2)])

        alpha, gamma = 0.10, 2.0
        lb_var, ub_var = lce._sensitivity_variance(gamma)
        zcrit = stats.norm.isf(alpha / 2)
        sen_lb, sen_ub = lce.sensitivity_analysis(gamma=gamma)

        eci_lb, eci_ub = lce.expanded_confidence_interval(
            alpha=alpha, gamma=gamma, bootstrap=False
        )

        assert eci_lb == pytest.approx(sen_lb - zcrit * math.sqrt(lb_var))
        assert eci_ub == pytest.approx(sen_ub + zcrit * math.sqrt(ub_var))

    @staticmethod
    def test_expanded_confidence_interval_no_bootstrap_one_sided() -> None:
        """bootstrap=False one-sided ECI has exactly one finite bound."""
        est, _, _ = make_estimator(n=400, seed=23)
        lce = LinearCombinationEstimator(terms=[(1.0, est)])

        lb, ub = lce.expanded_confidence_interval(
            alpha=0.10, gamma=2.0, alternative="greater", bootstrap=False
        )
        assert np.isfinite(lb)
        assert ub == np.inf

        lb, ub = lce.expanded_confidence_interval(
            alpha=0.10, gamma=2.0, alternative="less", bootstrap=False
        )
        assert lb == -np.inf
        assert np.isfinite(ub)

    @staticmethod
    def test_expanded_confidence_interval_no_bootstrap_no_terms() -> None:
        """Affine-only LCE bootstrap=False ECI collapses to the point estimate."""
        lce = LinearCombinationEstimator(terms=[], affine=0.5)
        eci_lb, eci_ub = lce.expanded_confidence_interval(
            alpha=0.10, gamma=3.0, bootstrap=False
        )
        assert eci_lb == pytest.approx(0.5)
        assert eci_ub == pytest.approx(0.5)
