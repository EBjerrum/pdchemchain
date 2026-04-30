"""Score transformation links: map raw numeric scores to [0, 1].

Direction is inferred from good_value vs bad_value — no explicit objective/reverse parameters needed.
All transforms share the pattern: good_value → 1.0, bad_value → 0.0.
"""
import io
import math
from dataclasses import dataclass

import pandas as pd

from pdchemchain.base import RowLink
from pdchemchain.typing import InColumnName

_SIGMOID_EPSILON = 0.02  # Score target at bad_value for sigmoid auto-tuning
_EXP_CLAMP = 500.0       # Clamp exp() argument to avoid OverflowError


def _safe_sigmoid(z: float) -> float:
    """Compute 1/(1+exp(-z)) without overflow for extreme z."""
    if z < -_EXP_CLAMP:
        return 0.0
    if z > _EXP_CLAMP:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def _auto_k(half_distance: float, epsilon: float = _SIGMOID_EPSILON) -> float:
    """Compute sigmoid k so score ≈ 1-epsilon at good and ≈ epsilon at bad."""
    if half_distance <= 0:
        return 1.0
    return math.log(1.0 / epsilon - 1.0) / half_distance


@dataclass
class _ScoreTransform(RowLink):
    """Internal base class for all score transform links.

    Provides shared _row_apply, plot(), and _repr_png_() for all subclasses.
    Subclasses must implement _compute(value) and _plot_range().
    """

    in_column: InColumnName = "score"
    out_column: str = "score_t"

    def _compute(self, value: float) -> float:
        raise NotImplementedError

    def _row_apply(self, row: pd.Series) -> pd.Series:
        value = float(row[self.in_column])
        row[self.out_column] = float("nan") if math.isnan(value) else self._compute(value)
        return row

    def _plot_range(self) -> tuple[float, float]:
        """Default range for scalar good/bad transforms. Override for Gaussian/Plateau."""
        lo, hi = min(self.good_value, self.bad_value), max(self.good_value, self.bad_value)
        pad = max((hi - lo) * 0.15, abs(hi) * 0.05, 0.1)
        return (lo - pad, hi + pad)

    def _build_figure(self, x_range: tuple[float, float] | None = None, n: int = 300):
        """Build and return a matplotlib Figure for this transform."""
        import matplotlib.pyplot as plt
        import numpy as np

        if x_range is None:
            x_range = self._plot_range()

        xs = np.linspace(x_range[0], x_range[1], n)
        ys = [self._compute(float(x)) for x in xs]

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(xs, ys, lw=2)
        ax.set_xlabel(self.in_column)
        ax.set_ylabel(self.out_column)
        ax.set_title(type(self).__name__)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(x_range[0], x_range[1])
        self._annotate(ax)
        fig.tight_layout()
        return fig

    def _annotate(self, ax):
        """Add good_value / bad_value markers. Override in PlateauTransform."""
        ax.axvline(self.good_value, color="green", linestyle="--", alpha=0.7, label=f"good ({self.good_value})")
        ax.axvline(self.bad_value, color="red", linestyle="--", alpha=0.7, label=f"bad ({self.bad_value})")
        ax.legend(fontsize=8)

    def plot(self, x_range: tuple[float, float] | None = None, n: int = 300):
        """Plot the transform curve and return the Figure.

        Parameters
        ----------
        x_range
            (min, max) for x-axis. Defaults to good/bad range ± 15%.
        n
            Number of sample points (default 300).

        Returns
        -------
        matplotlib.figure.Figure
        """
        return self._build_figure(x_range=x_range, n=n)

    def _repr_png_(self):
        """Return PNG bytes for Jupyter auto-display."""
        try:
            import matplotlib.pyplot as plt
            fig = self._build_figure()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            return buf.getvalue()
        except ImportError:
            return None


@dataclass
class StepTransform(_ScoreTransform):
    """Hard threshold: 1.0 on the good side of bad_value, 0.0 on the bad side.

    Direction is inferred: good_value > bad_value means higher is better (ascending).
    Threshold is strictly at bad_value — values equal to bad_value score 0.0
    regardless of direction.

    Parameters
    ----------
    in_column
        Column with the raw score to transform.
    out_column
        Column to store the transformed score in [0, 1].
    good_value
        Raw value representing a good outcome (used to infer direction).
    bad_value
        Threshold; score drops to 0.0 on the bad side of this value.

    Examples
    --------
    >>> t = StepTransform(in_column="QED", out_column="QED_score", good_value=0.7, bad_value=0.4)
    >>> # QED > 0.4 → 1.0,  QED <= 0.4 → 0.0
    """

    good_value: float = 1.0
    bad_value: float = 0.0

    def _compute(self, value: float) -> float:
        if self.good_value > self.bad_value:
            return 1.0 if value > self.bad_value else 0.0
        else:
            return 1.0 if value < self.bad_value else 0.0


@dataclass
class LinearTransform(_ScoreTransform):
    """Linear ramp from bad_value (0.0) to good_value (1.0), clipped outside.

    Direction is inferred from good_value vs bad_value.

    Parameters
    ----------
    in_column
        Column with the raw score to transform.
    out_column
        Column to store the transformed score in [0, 1].
    good_value
        Raw value that maps to 1.0.
    bad_value
        Raw value that maps to 0.0.

    Examples
    --------
    >>> t = LinearTransform(in_column="MolWt", out_column="MolWt_score", good_value=400, bad_value=600)
    >>> # 400 → 1.0, 600 → 0.0, linear between, clipped outside
    """

    good_value: float = 1.0
    bad_value: float = 0.0

    def _compute(self, value: float) -> float:
        span = self.good_value - self.bad_value
        if span == 0.0:
            return 0.0
        return min(1.0, max(0.0, (value - self.bad_value) / span))


@dataclass
class SigmoidTransform(_ScoreTransform):
    """Smooth S-curve centered between good_value and bad_value.

    Direction is inferred: good_value > bad_value means higher is better.
    When k is None (default), steepness is auto-tuned so score ≈ 0.98 at
    good_value and ≈ 0.02 at bad_value.

    Parameters
    ----------
    in_column
        Column with the raw score to transform.
    out_column
        Column to store the transformed score in [0, 1].
    good_value
        Raw value where score approaches 1.0 (≈ 0.98 with auto k).
    bad_value
        Raw value where score approaches 0.0 (≈ 0.02 with auto k).
    k
        Sigmoid steepness. None = auto-tune (default). Higher = sharper transition.

    Examples
    --------
    >>> t = SigmoidTransform(in_column="docking_score", out_column="dock_score",
    ...                      good_value=-10, bad_value=-4)
    """

    good_value: float = 1.0
    bad_value: float = 0.0
    k: float | None = None

    def _effective_k(self) -> float:
        if self.k is not None:
            return self.k
        half = abs(self.good_value - self.bad_value) / 2.0
        return _auto_k(half)

    def _compute(self, value: float) -> float:
        midpoint = (self.good_value + self.bad_value) / 2.0
        sign = 1.0 if self.good_value > self.bad_value else -1.0
        return _safe_sigmoid(sign * self._effective_k() * (value - midpoint))


@dataclass
class GaussianTransform(_ScoreTransform):
    """Bell-curve transform: 1.0 at good_value, falls to ~0.01 at bad_value.

    sigma is derived via the 3-sigma rule: sigma = abs(good_value - bad_value) / 3.
    Unlike directional transforms, only distance from the peak matters.

    Parameters
    ----------
    in_column
        Column with the raw score to transform.
    out_column
        Column to store the transformed score in [0, 1].
    good_value
        Peak of the Gaussian (maps to 1.0).
    bad_value
        Point where score drops to ~0.01 (defines sigma = |good - bad| / 3).

    Examples
    --------
    >>> t = GaussianTransform(in_column="logP", out_column="logP_score",
    ...                       good_value=2.5, bad_value=5.5)
    >>> # Peak at 2.5, decays to ~0 at 5.5
    """

    good_value: float = 2.5
    bad_value: float = 5.5

    def _effective_sigma(self) -> float:
        return abs(self.good_value - self.bad_value) / 3.0

    def _compute(self, value: float) -> float:
        sigma = self._effective_sigma()
        if sigma == 0.0:
            return 1.0 if value == self.good_value else 0.0
        return math.exp(-0.5 * ((value - self.good_value) / sigma) ** 2)

    def _plot_range(self) -> tuple[float, float]:
        delta = abs(self.good_value - self.bad_value)
        pad = max(delta * 0.15, 0.1)
        return (self.good_value - delta - pad, self.good_value + delta + pad)


@dataclass
class PlateauTransform(_ScoreTransform):
    """Flat-topped transform: 1.0 within good_value range, sigmoid falloff to 0.0 at bad_value bounds.

    bad_value bounds must enclose good_value bounds on both sides.
    Asymmetric slopes are natural since left/right distances can differ.
    k is auto-tuned per side when None (default).

    Parameters
    ----------
    in_column
        Column with the raw score to transform.
    out_column
        Column to store the transformed score in [0, 1].
    good_value
        (low, high) plateau range where score is 1.0. Order does not matter.
    bad_value
        (low, high) outer range where score reaches ~0.0. Order does not matter.
        Must enclose good_value on both sides.
    k
        Sigmoid steepness for both shoulders. None = auto-tune per side (default).

    Examples
    --------
    >>> t = PlateauTransform(in_column="logP", out_column="logP_score",
    ...                      good_value=(1.0, 3.0), bad_value=(-1.0, 6.0))
    >>> # logP 1–3 → 1.0, smooth falloff toward 0 at -1 and 6
    """

    good_value: tuple = (1.0, 3.0)
    bad_value: tuple = (-1.0, 6.0)
    k: float | None = None

    def __post_init__(self):
        # Coerce list → tuple (YAML round-trip returns lists)
        self.good_value = tuple(self.good_value)
        self.bad_value = tuple(self.bad_value)
        super().__post_init__()
        # Pre-compute sorted bounds (used in _compute, _plot_range, _annotate)
        self._g_lo, self._g_hi = sorted(self.good_value)
        self._b_lo, self._b_hi = sorted(self.bad_value)
        if not (self._b_lo <= self._g_lo and self._b_hi >= self._g_hi):
            raise ValueError(
                f"bad_value bounds {self.bad_value} must enclose good_value bounds {self.good_value}. "
                f"Expected: min(bad) <= min(good) and max(bad) >= max(good)."
            )
        # Pre-compute per-side k and midpoints
        half_left = (self._g_lo - self._b_lo) / 2.0
        self._k_left = self.k if self.k is not None else _auto_k(half_left)
        self._mid_left = (self._b_lo + self._g_lo) / 2.0
        half_right = (self._b_hi - self._g_hi) / 2.0
        self._k_right = self.k if self.k is not None else _auto_k(half_right)
        self._mid_right = (self._g_hi + self._b_hi) / 2.0

    def _compute(self, value: float) -> float:
        f_left = _safe_sigmoid(self._k_left * (value - self._mid_left))
        f_right = _safe_sigmoid(-self._k_right * (value - self._mid_right))
        return min(f_left, f_right)

    def _plot_range(self) -> tuple[float, float]:
        pad = max((self._b_hi - self._b_lo) * 0.15, 0.1)
        return (self._b_lo - pad, self._b_hi + pad)

    def _annotate(self, ax):
        ax.axvspan(self._g_lo, self._g_hi, alpha=0.15, color="green", label=f"good ({self._g_lo}–{self._g_hi})")
        ax.axvline(self._b_lo, color="red", linestyle="--", alpha=0.7, label=f"bad lo ({self._b_lo})")
        ax.axvline(self._b_hi, color="red", linestyle="--", alpha=0.7, label=f"bad hi ({self._b_hi})")
        ax.legend(fontsize=8)
