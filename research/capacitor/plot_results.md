# plot_results.py

## Purpose

Create a standalone scientific figure from `results.json` without presenting
model curves as measured data. Requires Matplotlib and NumPy; no GUI backend.

## Components

`render` writes PNG/SVG panels for internal plate load, candidate force versus
the single vertical measurement, and independent stress-integral convergence.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| README.md | force_comparison.png and .svg | Output filename |
| benchmark.py outputs | Named SI quantities and convergence rows | Input schema |

## Notes

The plotted experimental error bar is three times the paper's reported
uncertainty (`u`), following its decision criterion; it is not a newly derived
confidence interval. Positive voltages shown; full signed model curves are
saved in JSON. Force residuals bottom out at floating-point roundoff.
The verification panel explicitly labels its synthetic sources; annotations
are positioned clear of the model curves.
SVG output uses stable element IDs, omits the generation date, and strips
insignificant trailing whitespace for reproducible, clean version control.
