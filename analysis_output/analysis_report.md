# Wireless coverage analysis report

> This report analyses recorded CSV sample data only. It does not represent
> live or real-world network measurements unless the user supplies and
> documents those measurements separately.

## Executive summary

- Input: `C:\Users\User\Python_Project2026\survey_points.csv`
- Valid measured points: **11**
- Rejected rows: **1**
- Weak threshold: **-67 dBm**
- Weak measured points: **6**
  (**54.55%**)
- Weakest measured location: **C-03 (-83.0 dBm)**
- Grid cells with measurements: **44.00%**

## Engineering interpretation

RSSI values are measured samples. Grid-cell means are spatial summaries, not
direct measurements of every location inside a cell. Empty cells remain
unmeasured and are not treated as weak. The -67 dBm threshold is a practical
planning boundary; production designs should replace it with the application's
validated service-level target.
