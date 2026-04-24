import pytest
import numpy as np
import pandas as pd

from pdchemchain.links.chemistry import DockingEfficiency, DockingEfficiencyAnalyzer
from pdchemchain.typing import Partitionable
from tests.basetest import BaseTest


@pytest.fixture
def docking_dataframe():
    """Synthetic dataset with known linear frontier: slope=-0.3, intercept=-4.0"""
    np.random.seed(42)
    n = 200
    ha = np.random.randint(10, 35, size=n)
    # Known frontier + positive noise (scores worse than frontier)
    frontier_score = -0.3 * ha - 4.0
    noise = np.random.uniform(0, 5, size=n)
    scores = frontier_score + noise
    return pd.DataFrame({"ha": ha, "score": scores})


@pytest.fixture
def docking_dataframe_with_elbow():
    """Dataset where HA > 30 has a clear non-linear elbow."""
    np.random.seed(42)
    n = 300
    ha = np.random.randint(10, 45, size=n)
    scores = np.where(
        ha <= 30,
        -0.3 * ha - 4.0 + np.random.uniform(0, 3, size=n),  # linear region
        -0.1 * ha - 8.0 + np.random.uniform(0, 8, size=n),  # elbow: flatter, noisier
    )
    return pd.DataFrame({"ha": ha, "score": scores})


class TestDockingEfficiency(BaseTest):
    _Link = DockingEfficiency
    _classparams = {"score_column": "score", "ha_column": "ha", "slope": -0.3, "intercept": -4.0}
    _alt_classparams = {"score_column": "score2", "ha_column": "ha2", "slope": -0.5, "intercept": -3.0}

    @pytest.fixture
    def sample_dataframe(self, docking_dataframe):
        return docking_dataframe

    def test_calculation(self, docking_dataframe):
        link = DockingEfficiency(score_column="score", ha_column="ha",
                                 slope=-0.3, intercept=-4.0)
        result = link(docking_dataframe)

        row = result.iloc[0]
        score = float(docking_dataframe.iloc[0]["score"])
        ha = float(docking_dataframe.iloc[0]["ha"])

        assert pytest.approx(row["dock_eff_sub"]) == score - (-0.3) * ha
        assert pytest.approx(row["dock_eff_per_ha"]) == score / ha
        assert pytest.approx(row["dock_eff_ratio"]) == score / (-0.3 * ha + -4.0)

    def test_ha_zero(self):
        df = pd.DataFrame({"score": [-5.0], "ha": [0]})
        link = DockingEfficiency(score_column="score", ha_column="ha")
        result = link(df)
        assert np.isnan(result.iloc[0]["dock_eff_per_ha"])

    def test_custom_prefix(self, docking_dataframe):
        link = DockingEfficiency(score_column="score", ha_column="ha",
                                 out_prefix="my_eff")
        result = link(docking_dataframe)
        assert "my_eff_sub" in result.columns
        assert "my_eff_per_ha" in result.columns
        assert "my_eff_ratio" in result.columns

    def test_default_params(self):
        link = DockingEfficiency()
        assert link.slope == -0.264
        assert link.intercept == -5.11


class TestDockingEfficiencyAnalyzer(BaseTest):
    _Link = DockingEfficiencyAnalyzer
    _classparams = {"score_column": "score", "ha_column": "ha", "ha_min": 10}
    _alt_classparams = {"score_column": "score2", "ha_column": "ha2", "ha_min": 15}

    @pytest.fixture
    def sample_dataframe(self, docking_dataframe):
        return docking_dataframe

    def test_partitionable_no(self):
        assert DockingEfficiencyAnalyzer._partitionable == Partitionable.NO

    def test_frontier_fit(self, docking_dataframe):
        analyzer = DockingEfficiencyAnalyzer(
            score_column="score", ha_column="ha", ha_min=10
        )
        analyzer(docking_dataframe)

        # Fitted slope should be close to the known -0.3
        assert -0.5 < analyzer.slope_ < -0.1
        # Fitted intercept should be close to -4.0
        assert -8.0 < analyzer.intercept_ < 0.0
        assert 0.0 < analyzer.r_squared_ <= 1.0

    def test_auto_detection(self, docking_dataframe_with_elbow):
        analyzer = DockingEfficiencyAnalyzer(
            score_column="score", ha_column="ha",
            ha_min=10, r_squared_threshold=0.7,
        )
        analyzer(docking_dataframe_with_elbow)

        # Auto-detection should cap below the elbow region
        assert analyzer.ha_range_[1] <= 35

    def test_explicit_ha_max(self, docking_dataframe):
        analyzer = DockingEfficiencyAnalyzer(
            score_column="score", ha_column="ha",
            ha_min=10, ha_max=25,
        )
        analyzer(docking_dataframe)

        assert analyzer.ha_range_ == (10, 25)
        # No R² sweep when ha_max is explicit
        assert not hasattr(analyzer, '_r_squared_by_cutoff')

    def test_output_columns(self, docking_dataframe):
        analyzer = DockingEfficiencyAnalyzer(
            score_column="score", ha_column="ha",
        )
        result = analyzer(docking_dataframe)

        assert "dock_eff_sub" in result.columns
        assert "dock_eff_per_ha" in result.columns
        assert "dock_eff_ratio" in result.columns

    def test_docking_efficiency_property(self, docking_dataframe):
        analyzer = DockingEfficiencyAnalyzer(
            score_column="score", ha_column="ha",
        )
        analyzer(docking_dataframe)

        scorer = analyzer.docking_efficiency
        assert isinstance(scorer, DockingEfficiency)
        assert scorer.slope == analyzer.slope_
        assert scorer.intercept == analyzer.intercept_

    def test_docking_efficiency_before_fit(self):
        analyzer = DockingEfficiencyAnalyzer(
            score_column="score", ha_column="ha",
        )
        with pytest.raises(AttributeError, match="not been fitted"):
            _ = analyzer.docking_efficiency

    def test_too_few_points(self):
        df = pd.DataFrame({"score": [-5.0, -6.0], "ha": [15, 16]})
        analyzer = DockingEfficiencyAnalyzer(
            score_column="score", ha_column="ha",
            min_frontier_points=5,
        )
        result = analyzer(df)
        # Should return without efficiency columns
        assert "dock_eff_sub" not in result.columns

    def test_plot_generation(self, docking_dataframe, tmp_path):
        analyzer = DockingEfficiencyAnalyzer(
            score_column="score", ha_column="ha",
            plot_dir=str(tmp_path),
        )
        analyzer(docking_dataframe)

        plot_file = tmp_path / "docking_efficiency_diagnostic.png"
        assert plot_file.exists()
        assert plot_file.stat().st_size > 0

    def test_repr_png(self, docking_dataframe):
        analyzer = DockingEfficiencyAnalyzer(
            score_column="score", ha_column="ha",
        )
        analyzer(docking_dataframe)

        png_data = analyzer._repr_png_()
        assert png_data is not None
        assert png_data[:8] == b'\x89PNG\r\n\x1a\n'  # PNG magic bytes
