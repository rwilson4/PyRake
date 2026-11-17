"""Balancing Weights and Raking."""

from __future__ import annotations

import logging
import os
import pathlib
from importlib.resources import files as _resource_files

import cvxium

from pyrake.calibration import (
    KNEE_GRID_POINTS,
    EfficientFrontier,
    EfficientFrontierResults,
    ExternalValidityVarianceFrontier,
    ExternalValidityVarianceFrontierResults,
    FallbackOptimizationResult,
    Huber,
    JointCalibrator,
    KLDivergence,
    Rake,
    SquaredL2,
    external_band,
    knee_external_scale,
    plot_balance,
    plot_balance_2_sample,
    relabel_for_the_chosen_band,
    solve_with_knee_fallback,
)
from pyrake.estimation import (
    AIPWEstimator,
    ATCEstimator,
    ATEEstimator,
    ATTEstimator,
    DoubleSamplingEstimand,
    Estimand,
    IPWEstimator,
    LinearCombinationEstimator,
    MeanEstimator,
    NonRespondentMean,
    PopulationMean,
    RatioEstimator,
    SAIPWEstimator,
    SampleMean,
    SimpleDifferenceEstimator,
    SIPWEstimator,
    TreatmentEffectEstimator,
    TreatmentEffectRatioEstimator,
    WeightingEstimator,
    meta_analysis,
)

__all__ = [
    "KNEE_GRID_POINTS",
    "AIPWEstimator",
    "ATCEstimator",
    "ATEEstimator",
    "ATTEstimator",
    "DoubleSamplingEstimand",
    "EfficientFrontier",
    "EfficientFrontierResults",
    "Estimand",
    "ExternalValidityVarianceFrontier",
    "ExternalValidityVarianceFrontierResults",
    "FallbackOptimizationResult",
    "Huber",
    "IPWEstimator",
    "JointCalibrator",
    "KLDivergence",
    "LinearCombinationEstimator",
    "MeanEstimator",
    "NonRespondentMean",
    "PopulationMean",
    "Rake",
    "RatioEstimator",
    "SAIPWEstimator",
    "SIPWEstimator",
    "SampleMean",
    "SimpleDifferenceEstimator",
    "SquaredL2",
    "TreatmentEffectEstimator",
    "TreatmentEffectRatioEstimator",
    "WeightingEstimator",
    "enable_logging",
    "external_band",
    "knee_external_scale",
    "meta_analysis",
    "plot_balance",
    "plot_balance_2_sample",
    "relabel_for_the_chosen_band",
    "solve_with_knee_fallback",
    "usage",
]


def usage() -> str:
    """Return the contents of USAGE.md."""
    try:
        return (
            _resource_files("pyrake").joinpath("USAGE.md").read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        # A source checkout keeps USAGE.md at the repository root; the wheel
        # force-includes it inside the package.
        return (pathlib.Path(__file__).parent.parent / "USAGE.md").read_text(
            encoding="utf-8"
        )


def __dir__() -> list[str]:
    return __all__


# A library must not configure logging for the application that imports it, so
# the root of PyRake's namespace gets a NullHandler and a quiet level. Callers
# opt in with ``enable_logging``. Mirrors what Cvxium does for ``cvxium``.
logging.getLogger("pyrake").addHandler(logging.NullHandler())
# Only when nobody has chosen a level yet: `enable_logging` needs the logger's
# NAME, not this module, so a caller may configure the namespace before
# importing PyRake. An unconditional setLevel would undo that on import and
# the log would come back empty with nothing to explain why.
if logging.getLogger("pyrake").level == logging.NOTSET:
    logging.getLogger("pyrake").setLevel(logging.WARNING)


def enable_logging(
    path: str | os.PathLike[str] | None = None,
    level: int = logging.INFO,
    solver_level: int = logging.WARNING,
    mode: str = "w",
) -> logging.Handler:
    """Log PyRake's calibration narration, and Cvxium's solves beneath it.

    Wires ``pyrake`` and its dependency ``cvxium`` to one shared handler, so
    the inner solver's output lands underneath the line naming the stage that
    invoked it rather than in a separate file with separate timestamps.

    The two namespaces get separate thresholds because they are wanted at
    different volumes. The default is the narration alone -- one line per
    calibration, one per fallback stage, each with a timing -- which is a
    couple of dozen lines and answers "which stage, and where did the time
    go". Cvxium's own INFO is per-centering-step and runs to thousands of
    lines on a real problem, so it is off until asked for, and its DEBUG is
    per-Newton-iteration.

    Parameters
    ----------
     path : str or PathLike, optional
        Log file. ``~`` is expanded. None logs to stderr.
     level : int, optional
        Threshold for ``pyrake``: the calibration narration.
     solver_level : int, optional
        Threshold for ``cvxium``: the solves underneath. Defaults to
        ``logging.WARNING``, i.e. silent unless something goes wrong.
     mode : str, optional
        File mode; ``"w"`` truncates per run.

    Returns
    -------
     logging.Handler
        The attached handler.

    Examples
    --------
    Narration only::

        pyrake.enable_logging("~/calibration.log")

    A slow stage, and you want to see the solver work through it::

        pyrake.enable_logging("~/calibration.log", solver_level=logging.INFO)

    """
    # The handler is the floor for both, so it has to sit at the more
    # permissive of the two or it would filter out what the logger admitted.
    handler = cvxium.enable_logging(
        path,
        level=min(level, solver_level),
        namespaces=("cvxium", "pyrake"),
        mode=mode,
    )
    logging.getLogger("pyrake").setLevel(level)
    logging.getLogger("cvxium").setLevel(solver_level)
    return handler
