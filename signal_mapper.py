"""Analysis of recorded wireless-signal samples for floor or campus points."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_WEAK_THRESHOLD_DBM = -67.0
MIN_SIGNAL_DBM = -120.0
MAX_SIGNAL_DBM = 0.0
DEFAULT_GRID_BINS = (10, 10)
REQUIRED_COLUMNS = frozenset(("location_id", "x", "y", "signal_dbm"))
SIGNAL_BANDS = (
    ("Excellent", -50.0, 0.0),
    ("Good", -67.0, -50.0),
    ("Fair", -75.0, -67.0),
    ("Poor", MIN_SIGNAL_DBM, -75.0),
)


@dataclass(frozen=True, slots=True)
class SurveyPoint:
    """A measured signal strength at a two-dimensional survey location."""

    location_id: str
    x: float
    y: float
    signal_dbm: float
    label: str = ""

    def __post_init__(self) -> None:
        if not self.location_id.strip():
            raise ValueError("location_id must not be empty")
        for name, value in (("x", self.x), ("y", self.y), ("signal_dbm", self.signal_dbm)):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not MIN_SIGNAL_DBM <= self.signal_dbm <= MAX_SIGNAL_DBM:
            raise ValueError(
                f"signal_dbm must be between {MIN_SIGNAL_DBM:g} and {MAX_SIGNAL_DBM:g} dBm"
            )

    @property
    def is_weak(self) -> bool:
        """Return whether this point is weaker than the standard -67 dBm threshold."""

        return self.signal_dbm < DEFAULT_WEAK_THRESHOLD_DBM


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Immutable, reusable result of one survey analysis run."""

    points: list[SurveyPoint]
    rejected: pd.DataFrame
    location_comparison: pd.DataFrame
    weak_points: pd.DataFrame
    grid_means: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    threshold_dbm: float

    @property
    def frame(self) -> pd.DataFrame:
        return points_dataframe(self.points, self.threshold_dbm)

    @property
    def statistics(self) -> dict[str, float | int]:
        signals = self.frame["signal_dbm"]
        return {
            "valid_points": len(self.points),
            "rejected_rows": len(self.rejected),
            "locations": int(self.frame["location_id"].nunique()),
            "mean_dbm": round(float(signals.mean()), 2),
            "median_dbm": round(float(signals.median()), 2),
            "std_dbm": round(float(signals.std(ddof=0)), 2),
            "minimum_dbm": round(float(signals.min()), 2),
            "maximum_dbm": round(float(signals.max()), 2),
            "weak_points": len(self.weak_points),
            "weak_rate": round(float(self.frame["is_weak"].mean()), 4),
            "measured_cells": int(np.isfinite(self.grid_means).sum()),
            "grid_cells": int(self.grid_means.size),
            "measured_cell_rate": round(
                float(np.isfinite(self.grid_means).sum() / self.grid_means.size), 4
            ),
        }


def _is_weak(signal_dbm: float, threshold_dbm: float) -> bool:
    if not math.isfinite(threshold_dbm):
        raise ValueError("threshold_dbm must be finite")
    return signal_dbm < threshold_dbm


def load_survey_csv(path: str | Path) -> tuple[list[SurveyPoint], pd.DataFrame]:
    """Load valid points and a report of rows excluded from spatial analysis.

    Missing coordinates are not imputed: without a trustworthy location, a row
    cannot be assigned to a grid cell or used in a distance calculation.
    """

    path = Path(path)
    valid: list[SurveyPoint] = []
    rejected: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = frozenset(reader.fieldnames or ())
        missing_columns = REQUIRED_COLUMNS - columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV is missing required column(s): {missing}")
        for row_number, row in enumerate(reader, start=2):
            try:
                if None in row:
                    raise ValueError("row contains more fields than the CSV header")
                location_id = (row.get("location_id") or "").strip()
                x_text = (row.get("x") or "").strip()
                y_text = (row.get("y") or "").strip()
                signal_text = (row.get("signal_dbm") or "").strip()
                if not x_text or not y_text:
                    raise ValueError("missing x or y coordinate")
                point = SurveyPoint(
                    location_id=location_id,
                    x=float(x_text),
                    y=float(y_text),
                    signal_dbm=float(signal_text),
                    label=(row.get("label") or "").strip(),
                )
            except (TypeError, ValueError) as error:
                rejected.append(
                    {
                        "row_number": row_number,
                        "location_id": row.get("location_id", ""),
                        "reason": str(error),
                    }
                )
            else:
                valid.append(point)
    return valid, pd.DataFrame(rejected, columns=("row_number", "location_id", "reason"))


def points_dataframe(
    points: Iterable[SurveyPoint],
    threshold_dbm: float = DEFAULT_WEAK_THRESHOLD_DBM,
) -> pd.DataFrame:
    """Convert points to a tabular representation suitable for Pandas analysis."""

    if not math.isfinite(threshold_dbm):
        raise ValueError("threshold_dbm must be finite")
    return pd.DataFrame(
        [
            {
                "location_id": point.location_id,
                "x": point.x,
                "y": point.y,
                "signal_dbm": point.signal_dbm,
                "label": point.label,
                "is_weak": _is_weak(point.signal_dbm, threshold_dbm),
            }
            for point in points
        ]
    )


def compare_locations(
    points: Iterable[SurveyPoint],
    threshold_dbm: float = DEFAULT_WEAK_THRESHOLD_DBM,
) -> pd.DataFrame:
    """Compare each location, including repeated measurements by location ID."""

    frame = points_dataframe(points, threshold_dbm)
    if frame.empty:
        return pd.DataFrame(
            columns=(
                "location_id",
                "samples",
                "mean_dbm",
                "minimum_dbm",
                "maximum_dbm",
                "weak_rate",
            )
        )
    return (
        frame.groupby("location_id", as_index=False)
        .agg(
            samples=("signal_dbm", "size"),
            mean_dbm=("signal_dbm", "mean"),
            minimum_dbm=("signal_dbm", "min"),
            maximum_dbm=("signal_dbm", "max"),
            weak_rate=("is_weak", "mean"),
        )
        .sort_values("mean_dbm", ascending=True, ignore_index=True)
    )


def pairwise_distances(points: Iterable[SurveyPoint]) -> np.ndarray:
    """Return the Euclidean distance matrix for valid survey points."""

    coordinates = np.asarray([(point.x, point.y) for point in points], dtype=float)
    if coordinates.size == 0:
        return np.empty((0, 0), dtype=float)
    deltas = coordinates[:, None, :] - coordinates[None, :, :]
    return np.linalg.norm(deltas, axis=2)


def grid_summary(
    points: Iterable[SurveyPoint], bins: tuple[int, int] = DEFAULT_GRID_BINS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean signal per rectangular grid cell and its x/y edges."""

    point_list = list(points)
    if not point_list:
        raise ValueError("at least one valid point is required")
    if len(bins) != 2 or any(not isinstance(size, int) or size < 1 for size in bins):
        raise ValueError("bins must contain two positive integers")
    x = np.asarray([point.x for point in point_list])
    y = np.asarray([point.y for point in point_list])
    signal = np.asarray([point.signal_dbm for point in point_list])
    sums, x_edges, y_edges = np.histogram2d(x, y, bins=bins, weights=signal)
    counts, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    return means, x_edges, y_edges


def grid_dataframe(means: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> pd.DataFrame:
    """Convert a NumPy grid into a readable table of cell centers and means."""

    if means.ndim != 2 or means.shape != (len(x_edges) - 1, len(y_edges) - 1):
        raise ValueError("means shape must match the supplied grid edges")
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing="ij")
    return pd.DataFrame(
        {
            "x_center": x_grid.ravel(),
            "y_center": y_grid.ravel(),
            "mean_signal_dbm": means.ravel(),
        }
    )


def weak_zones(
    points: Iterable[SurveyPoint], threshold_dbm: float = DEFAULT_WEAK_THRESHOLD_DBM
) -> pd.DataFrame:
    """Return points below the chosen weak-signal threshold."""

    if not math.isfinite(threshold_dbm):
        raise ValueError("threshold_dbm must be finite")
    frame = points_dataframe(points, threshold_dbm)
    return frame.loc[frame["is_weak"]].reset_index(drop=True)


def signal_band(signal_dbm: float) -> str:
    """Classify a measured RSSI into an engineering-friendly signal band."""

    for name, lower, upper in SIGNAL_BANDS:
        if lower <= signal_dbm < upper or name == "Excellent" and signal_dbm == upper:
            return name
    raise ValueError(f"signal_dbm is outside supported range: {signal_dbm}")


def analyse_points(
    points: Iterable[SurveyPoint],
    rejected: pd.DataFrame | None = None,
    threshold_dbm: float = DEFAULT_WEAK_THRESHOLD_DBM,
    bins: tuple[int, int] = DEFAULT_GRID_BINS,
) -> AnalysisResult:
    """Compute all reusable measurements and estimates without writing files."""

    point_list = list(points)
    if not point_list:
        raise ValueError("at least one valid survey point is required")
    if not math.isfinite(threshold_dbm):
        raise ValueError("threshold_dbm must be finite")
    means, x_edges, y_edges = grid_summary(point_list, bins)
    return AnalysisResult(
        points=point_list,
        rejected=rejected if rejected is not None else pd.DataFrame(),
        location_comparison=compare_locations(point_list, threshold_dbm),
        weak_points=weak_zones(point_list, threshold_dbm),
        grid_means=means,
        x_edges=x_edges,
        y_edges=y_edges,
        threshold_dbm=threshold_dbm,
    )


def write_report(result: AnalysisResult, output_dir: str | Path, input_name: str) -> None:
    """Write portable CSV, JSON, Markdown, NumPy, and image report artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.location_comparison.to_csv(output / "location_comparison.csv", index=False)
    result.rejected.to_csv(output / "rejected_rows.csv", index=False)
    result.weak_points.to_csv(output / "weak_zones.csv", index=False)
    result.frame.assign(signal_band=result.frame["signal_dbm"].map(signal_band)).to_csv(
        output / "validated_points.csv", index=False
    )
    grid_dataframe(result.grid_means, result.x_edges, result.y_edges).to_csv(
        output / "grid_summary.csv", index=False
    )
    np.savez(
        output / "grid_summary.npz",
        mean_signal=result.grid_means,
        x_edges=result.x_edges,
        y_edges=result.y_edges,
    )
    plot_signal_map(result.points, output / "signal_map.png", result.threshold_dbm)
    plot_signal_histogram(result.points, output / "signal_histogram.png", result.threshold_dbm)
    plot_grid_heatmap(
        result.grid_means,
        result.x_edges,
        result.y_edges,
        output / "grid_heatmap.png",
        result.threshold_dbm,
    )
    summary = {
        "input_file": input_name,
        "threshold_dbm": result.threshold_dbm,
        "grid_bins": [len(result.x_edges) - 1, len(result.y_edges) - 1],
        **result.statistics,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    weakest = result.location_comparison.iloc[0]
    report = f"""# Wireless coverage analysis report

> This report analyses recorded CSV sample data only. It does not represent
> live or real-world network measurements unless the user supplies and
> documents those measurements separately.

## Executive summary

- Input: `{input_name}`
- Valid measured points: **{result.statistics["valid_points"]}**
- Rejected rows: **{result.statistics["rejected_rows"]}**
- Weak threshold: **{result.threshold_dbm:g} dBm**
- Weak measured points: **{result.statistics["weak_points"]}**
  (**{result.statistics["weak_rate"]:.2%}**)
- Weakest measured location: **{weakest["location_id"]} ({weakest["mean_dbm"]:.1f} dBm)**
- Grid cells with measurements: **{result.statistics["measured_cell_rate"]:.2%}**

## Engineering interpretation

RSSI values are measured samples. Grid-cell means are spatial summaries, not
direct measurements of every location inside a cell. Empty cells remain
unmeasured and are not treated as weak. The -67 dBm threshold is a practical
planning boundary; production designs should replace it with the application's
validated service-level target.
"""
    (output / "analysis_report.md").write_text(report, encoding="utf-8")


def plot_signal_map(
    points: Iterable[SurveyPoint],
    output_path: str | Path,
    threshold_dbm: float = DEFAULT_WEAK_THRESHOLD_DBM,
) -> None:
    """Save a scatter map with weak points highlighted."""

    frame = points_dataframe(points, threshold_dbm)
    if frame.empty:
        raise ValueError("at least one valid point is required")
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        frame["x"],
        frame["y"],
        c=frame["signal_dbm"],
        cmap="RdYlGn",
        vmin=MIN_SIGNAL_DBM,
        vmax=MAX_SIGNAL_DBM,
        s=90,
    )
    weak = frame["signal_dbm"] < threshold_dbm
    if weak.any():
        ax.scatter(
            frame.loc[weak, "x"],
            frame.loc[weak, "y"],
            facecolors="none",
            edgecolors="black",
            s=170,
            linewidths=1.5,
            label=f"Weak (< {threshold_dbm:g} dBm)",
        )
        ax.legend()
    ax.set(title="Wireless signal coverage map", xlabel="X coordinate", ylabel="Y coordinate")
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="Signal strength (dBm)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_signal_histogram(
    points: Iterable[SurveyPoint],
    output_path: str | Path,
    threshold_dbm: float = DEFAULT_WEAK_THRESHOLD_DBM,
) -> None:
    """Save a histogram of signal readings with the weak threshold marked."""

    frame = points_dataframe(points, threshold_dbm)
    if frame.empty:
        raise ValueError("at least one valid point is required")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        frame["signal_dbm"],
        bins=min(12, max(4, len(frame))),
        color="#4c78a8",
        edgecolor="white",
    )
    ax.axvline(
        threshold_dbm,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label=f"Threshold: {threshold_dbm:g} dBm",
    )
    ax.set(
        title="Signal-strength distribution",
        xlabel="Signal strength (dBm)",
        ylabel="Samples",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_grid_heatmap(
    means: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    output_path: str | Path,
    threshold_dbm: float = DEFAULT_WEAK_THRESHOLD_DBM,
) -> None:
    """Save a visual heatmap of the NumPy grid summary."""

    if means.size == 0 or not np.isfinite(means).any():
        raise ValueError("grid summary contains no measured cells")
    if not math.isfinite(threshold_dbm):
        raise ValueError("threshold_dbm must be finite")
    masked_means = np.ma.masked_invalid(means.T)
    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.pcolormesh(
        x_edges,
        y_edges,
        masked_means,
        cmap="RdYlGn",
        vmin=MIN_SIGNAL_DBM,
        vmax=MAX_SIGNAL_DBM,
        shading="auto",
    )
    if means.shape[0] >= 2 and means.shape[1] >= 2:
        ax.contour(
            (x_edges[:-1] + x_edges[1:]) / 2,
            (y_edges[:-1] + y_edges[1:]) / 2,
            np.ma.masked_invalid(means.T < threshold_dbm),
            levels=[0.5],
            colors="black",
            linewidths=1,
        )
    ax.set(
        title="Mean signal by grid cell",
        xlabel="X coordinate",
        ylabel="Y coordinate",
        aspect="equal",
    )
    ax.grid(alpha=0.2)
    fig.colorbar(image, ax=ax, label="Mean signal strength (dBm)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def analyse(
    input_path: str | Path,
    output_dir: str | Path,
    threshold_dbm: float,
    bins: tuple[int, int] = DEFAULT_GRID_BINS,
) -> None:
    """Run the complete analysis and write tables and plots."""

    points, rejected = load_survey_csv(input_path)
    if not points:
        raise ValueError("the input contains no valid survey points")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = analyse_points(points, rejected, threshold_dbm, bins)
    write_report(result, output, str(Path(input_path).resolve()))
    print(f"Valid points: {result.statistics['valid_points']}")
    print(f"Rejected rows: {result.statistics['rejected_rows']}")
    print(f"Weak points (< {threshold_dbm:g} dBm): {result.statistics['weak_points']}")
    print(f"Results written to {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_csv",
        type=Path,
        help="CSV with location_id,x,y,signal_dbm,label columns",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_output"))
    parser.add_argument("--threshold", type=float, default=DEFAULT_WEAK_THRESHOLD_DBM)
    parser.add_argument(
        "--x-bins",
        type=int,
        default=DEFAULT_GRID_BINS[0],
        help="number of grid cells along the x axis",
    )
    parser.add_argument(
        "--y-bins",
        type=int,
        default=DEFAULT_GRID_BINS[1],
        help="number of grid cells along the y axis",
    )
    args = parser.parse_args()
    if args.x_bins < 1 or args.y_bins < 1:
        parser.error("--x-bins and --y-bins must be positive integers")
    try:
        analyse(args.input_csv, args.output_dir, args.threshold, (args.x_bins, args.y_bins))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
