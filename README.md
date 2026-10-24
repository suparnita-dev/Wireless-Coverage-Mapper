# Wireless-signal coverage mapper

This project analyses **recorded sample signal-strength data** at locations on
a floor or campus. It validates readings, keeps invalid rows in a rejection report,
uses NumPy for spatial calculations, uses Pandas for location comparisons, and
produces measured-point maps, grid summaries, and a human-readable report.
It includes a reusable analysis engine and a command-line workflow.

> **Data scope:** This portfolio project uses the recorded rows in
> `survey_points.csv` as demonstration data. The values are not claimed to be
> live, current, or representative measurements from a real building or
> campus. Replace them only with your own recorded survey data when performing
> a real survey.

## Architecture

```text
CSV survey -> validation -> SurveyPoint model -> NumPy/Pandas analysis
                                      |             |
                                      +--> plots ---+--> CSV/JSON/Markdown report
```

Measured RSSI samples and grid summaries are kept distinct. A grid cell is an
estimate of the mean of samples inside that cell; it is not a claim that every
point in the cell was measured.

## Input format

CSV files must contain `location_id`, `x`, `y`, and `signal_dbm` columns. The
optional `label` column describes the surveyed area. Coordinates must be
finite numbers, and readings must be between -120 dBm and 0 dBm. A missing
coordinate is rejected rather than imputed: assigning an unknown location to a
grid cell or distance calculation would create a misleading coverage result.

The included [survey_points.csv](./survey_points.csv) is a small recorded
demonstration dataset and contains one intentionally incomplete row to
demonstrate this policy.

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
- `analysis_report.md`: concise report generated from actual computed results
- `validated_points.csv`: accepted points with engineering signal bands

Run tests with `pytest`. The CI workflow also runs Ruff linting and formatting.

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
invalid input. `summary.json` can be consumed by an automated reporting pipeline. The heatmap
works with any positive grid size, including a single-cell summary.

## Quality checks

Run the complete verification suite from the project directory:

```text
.\.venv\Scripts\python.exe -m pytest -q
```

The test suite covers validation, missing locations, threshold behavior,
NumPy distances and grids, Pandas summaries, malformed CSV rows, and report
generation.

The CI workflow runs the same tests and Ruff checks automatically on every
push and pull request.

## Signal interpretation

The default weak threshold is **-67 dBm**, a configurable planning target.
Signal bands are descriptive engineering categories, not a guarantee of
throughput: Excellent (-50 to 0 dBm), Good (-67 to -50 dBm), Fair (-75 to
-67 dBm), and Poor (-120 to -75 dBm). Actual performance also depends on
noise, interference, channel width, client hardware, modulation, and traffic.

## Portfolio limitations

This tool does not perform RF propagation simulation, floor-plan obstruction
modelling, channel planning, or throughput prediction. Distances use the
coordinate units supplied by the survey, and heatmap cells with no samples
remain unmeasured. These boundaries are intentional so estimates are not
presented as measurements.
