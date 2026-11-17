"""Model means share balance-chart coordinates without synthetic weights."""

import unittest
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from pyrake.calibration.visualizations import (
    plot_balance,
    plot_balance_2_sample,
)


class ModelMeanBalanceTest(unittest.TestCase):
    """Verify model means share balance-chart coordinates."""

    def tearDown(self) -> None:
        plt.close("all")

    def test_one_sample_means_use_nominal_count_and_population_sd(self) -> None:
        """Verify one-sample means use nominal count and population SD."""
        X = np.array([[0.0, 1.0], [1.0, 1.0], [0.0, 2.0], [1.0, 3.0]])
        df, _ = plot_balance(
            X,
            np.array([0.5, 2.0]),
            ["a", "b"],
            {"raw": np.ones(4)},
            sigma=np.array([0.5, 2.0]),
            means={"model": np.array([0.75, 3.0])},
            quantify_imbalance=False,
        )
        np.testing.assert_allclose(
            df.loc[df["Method"].eq("model"), "delta"], [1.0, 1.0]
        )
        np.testing.assert_allclose(
            df.loc[df["Method"].eq("raw"), "delta"], [0.0, -0.25]
        )
        self.assertEqual(len(df), 4)

    def test_two_sample_means_use_common_fixed_scale(self) -> None:
        """Verify two-sample means use a common fixed scale."""
        df, _ = plot_balance_2_sample(
            np.ones((4, 1)),
            np.ones((9, 1)),
            ["a"],
            {},
            {},
            sigma1=np.array([2.0]),
            sigma2=np.array([2.0]),
            means1={"model": np.array([1.0])},
            means2={"model": np.array([3.0])},
            quantify_imbalance=False,
        )
        self.assertAlmostEqual(df.iloc[0]["delta"], 2 / np.sqrt(4 / 4 + 4 / 9))

    def test_model_means_have_same_imbalance_annotation(self) -> None:
        """Verify model means carry the same imbalance annotation."""
        X = np.ones((4, 1))
        df, ax = plot_balance(
            X,
            np.array([0.0]),
            ["a"],
            {"raw": np.ones(4)},
            sigma=np.ones(1),
            means={"model": np.ones(1)},
        )
        legend = ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.texts]
        self.assertIn("model (imbalance: 3)", labels)
        np.testing.assert_allclose(df["delta"], [2.0, 2.0])

    def test_conflicting_nonfinite_and_rank_based_means_raise(self) -> None:
        """Verify conflicting nonfinite and rank-based means raise."""
        cases: tuple[
            tuple[dict[str, npt.NDArray[np.float64]], Literal["z", "wilcoxon"] | None],
            ...,
        ] = (
            ({"raw": np.ones(1)}, None),
            ({"model": np.array([np.nan])}, None),
            ({"model": np.ones(1)}, "wilcoxon"),
        )
        for means, test in cases:
            with self.subTest(means=means, test=test), self.assertRaises(ValueError):
                plot_balance(
                    np.ones((4, 1)),
                    np.ones(1),
                    ["a"],
                    {"raw": np.ones(4)},
                    means=means,
                    test=test,
                )

    def test_two_sample_method_mismatch_raises(self) -> None:
        """Verify a two-sample method mismatch raises."""
        with self.assertRaisesRegex(ValueError, "consistent keys"):
            plot_balance_2_sample(
                np.ones((2, 1)),
                np.ones((3, 1)),
                ["a"],
                {},
                {},
                means1={"m": np.ones(1)},
            )

    def test_named_subset_omits_unmodeled_coordinates(self) -> None:
        """Verify a named subset omits unmodeled coordinates."""
        df, _ = plot_balance(
            np.ones((4, 2)),
            np.zeros(2),
            ["a", "b"],
            {"raw": np.ones(4)},
            sigma=np.ones(2),
            means={"model": pd.Series({"a": 1.0})},
            quantify_imbalance=False,
        )
        selected = df.loc[df["Method"].eq("model")]
        self.assertEqual(selected["covariate"].tolist(), ["a"])
        self.assertEqual(selected["delta"].tolist(), [2.0])

    def test_two_sample_zero_sd_preserves_weighted_rows(self) -> None:
        """Verify zero SD preserves weighted rows in two-sample plots."""
        df, _ = plot_balance_2_sample(
            np.ones((4, 1)),
            np.ones((4, 1)),
            ["a"],
            {"raw": np.ones(4)},
            {"raw": np.ones(4)},
            sigma1=np.zeros(1),
            sigma2=np.zeros(1),
        )
        self.assertEqual(len(df), 1)
        self.assertTrue(np.isnan(df.iloc[0]["delta"]))

    def test_disjoint_model_subsets_are_omitted_without_losing_weighted_rows(
        self,
    ) -> None:
        """Verify disjoint subsets omit cleanly without losing rows."""
        df, _ = plot_balance_2_sample(
            np.ones((4, 2)),
            np.ones((4, 2)),
            ["a", "b"],
            {"raw": np.ones(4)},
            {"raw": np.ones(4)},
            means1={"model": pd.Series({"a": 1.0})},
            means2={"model": pd.Series({"b": 1.0})},
        )
        self.assertEqual(len(df), 2)
        self.assertTrue(df["Method"].str.startswith("raw").all())
