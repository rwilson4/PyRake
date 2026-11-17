"""Calibration utilities."""

from pyrake.calibration.distance_metrics import (
    Huber,
    KLDivergence,
    SquaredL2,
)
from pyrake.calibration.frontier import (
    KNEE_GRID_POINTS,
    EfficientFrontier,
    EfficientFrontierResults,
    ExternalValidityVarianceFrontier,
    ExternalValidityVarianceFrontierResults,
    external_band,
    knee_external_scale,
    relabel_for_the_chosen_band,
    solve_with_knee_fallback,
)
from pyrake.calibration.joint_calibrator import (
    FallbackOptimizationResult,
    JointCalibrator,
)
from pyrake.calibration.rake import Rake
from pyrake.calibration.visualizations import (
    plot_balance,
    plot_balance_2_sample,
)

__all__ = [
    "KNEE_GRID_POINTS",
    "EfficientFrontier",
    "EfficientFrontierResults",
    "ExternalValidityVarianceFrontier",
    "ExternalValidityVarianceFrontierResults",
    "FallbackOptimizationResult",
    "Huber",
    "JointCalibrator",
    "KLDivergence",
    "Rake",
    "SquaredL2",
    "external_band",
    "knee_external_scale",
    "plot_balance",
    "plot_balance_2_sample",
    "relabel_for_the_chosen_band",
    "solve_with_knee_fallback",
]
