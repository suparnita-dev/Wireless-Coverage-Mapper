import numpy as np
import pytest

from signal_mapper import (
    SurveyPoint,
    compare_locations,
    grid_dataframe,
    grid_summary,
    load_survey_csv,
    pairwise_distances,
    weak_zones,
)


def test_load_csv_rejects_missing_coordinates_without_imputation():
    points, rejected = load_survey_csv("survey_points.csv")

    assert len(points) == 11
    assert len(rejected) == 1
    assert rejected.loc[0, "reason"] == "missing x or y coordinate"


def test_invalid_signal_is_rejected():
    with pytest.raises(ValueError, match="signal_dbm"):
        SurveyPoint("bad", 1.0, 2.0, -121.0)


def test_csv_header_and_row_shape_are_validated(tmp_path):
    missing_header = tmp_path / "missing.csv"
    missing_header.write_text("location_id,x\nA,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column"):
        load_survey_csv(missing_header)

    extra_field = tmp_path / "extra.csv"
    extra_field.write_text(
        "location_id,x,y,signal_dbm\nA,1,2,-60,unexpected\n",
        encoding="utf-8",
    )
    points, rejected = load_survey_csv(extra_field)
    assert not points
    assert rejected.loc[0, "reason"] == "row contains more fields than the CSV header"


def test_numpy_distances_and_grid_summary():
    points = [SurveyPoint("a", 0, 0, -50), SurveyPoint("b", 3, 4, -70)]

    assert np.array_equal(pairwise_distances(points), [[0.0, 5.0], [5.0, 0.0]])
    means, x_edges, y_edges = grid_summary(points, bins=(1, 1))
    assert means.shape == (1, 1)
    assert means[0, 0] == pytest.approx(-60.0)
    assert len(x_edges) == len(y_edges) == 2
    assert list(grid_dataframe(means, x_edges, y_edges).columns) == [
        "x_center",
        "y_center",
        "mean_signal_dbm",
    ]


def test_pandas_comparison_and_weak_filter():
    points = [SurveyPoint("a", 0, 0, -60), SurveyPoint("a", 1, 1, -70), SurveyPoint("b", 2, 2, -80)]

    comparison = compare_locations(points)
    assert comparison.loc[0, "location_id"] == "b"
    assert comparison.loc[1, "weak_rate"] == pytest.approx(0.5)
    assert len(weak_zones(points)) == 2


def test_custom_threshold_is_used_consistently():
    points = [SurveyPoint("a", 0, 0, -66), SurveyPoint("b", 1, 1, -70)]

    comparison = compare_locations(points, threshold_dbm=-65)

    assert comparison["weak_rate"].tolist() == [1.0, 1.0]
    assert len(weak_zones(points, threshold_dbm=-65)) == 2
