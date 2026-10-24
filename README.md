# Wireless-signal coverage mapper

This project analyses signal-strength samples collected at locations on a floor
or campus. It validates readings, keeps invalid rows in a rejection report,
uses NumPy for spatial calculations, uses Pandas for location comparisons, and
produces a scatter map and signal histogram.

## Input format

CSV files must contain `location_id`, `x`, `y`, and `signal_dbm` columns. The
optional `label` column describes the surveyed area. Coordinates must be
finite numbers, and readings must be between -120 dBm and 0 dBm. A missing
coordinate is rejected rather than imputed: assigning an unknown location to a
grid cell or distance calculation would create a misleading coverage result.

The included [survey_points.csv](./survey_points.csv) contains one intentionally
incomplete row to demonstrate this policy.

## Run

Install dependencies, then run:

```text
python -m pip install -r requirements.txt
python signal_mapper.py survey_points.csv --output-dir analysis_output
```

The output directory contains:

- `location_comparison.csv`: Pandas summary by location ID
- `weak_zones.csv`: points below the configured threshold
- `rejected_rows.csv`: rows excluded from spatial analysis and the reason
- `signal_map.png`: coordinate scatter map with weak points outlined
- `signal_histogram.png`: reading distribution and threshold
- `grid_heatmap.png`: visual mean-signal heatmap by grid cell
- `grid_summary.csv`: readable grid-cell centers and mean signal
- `grid_summary.npz`: NumPy mean signal per grid cell and cell edges for code
- `summary.json`: machine-readable run metrics and configuration

Run tests with `pytest`.

## Analysis choices

The default weak-zone threshold is **-67 dBm**, a practical planning value:
clients near this level can still connect, but have less margin for movement,
interference, and higher-throughput traffic. The threshold is a parameter, so
the analysis can be repeated with a site-specific service-level target.
`weak_zones.csv` identifies individual weak samples; the grid summary reveals
spatial clusters instead of over-interpreting one noisy reading. Empty grid
cells remain `NaN` in `grid_summary.npz`, preserving the distinction between
“not surveyed” and “measured weak.”

The `.npz` file is a NumPy data container, not an image, so it will not
visualize when opened directly. Open `grid_heatmap.png` for the visual result
or `grid_summary.csv` for a spreadsheet-friendly view. The `.npz` remains
available for downstream Python, NumPy, or machine-learning workflows:

```python
import numpy as np

grid = np.load("analysis_output/grid_summary.npz")
mean_signal = grid["mean_signal"]
```

## Production-style options

Use smaller or larger grid cells for the size of the surveyed area:

```text
python signal_mapper.py survey_points.csv --output-dir analysis_output --x-bins 20 --y-bins 15
```

The program validates required CSV headers before reading rows, reports
row-level data-quality failures, and exits with a clear command-line error for
invalid input. `summary.json` can be consumed by an automated reporting or
dashboard pipeline. The heatmap works with any positive grid size, including a
single-cell summary.

## Quality checks

Run the complete verification suite from the project directory:

```text
.\.venv\Scripts\python.exe -m pytest -q
```

The test suite covers validation, missing locations, threshold behavior,
NumPy distances and grids, Pandas summaries, and malformed CSV rows.
