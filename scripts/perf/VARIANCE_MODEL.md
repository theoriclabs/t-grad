# Prepared-runtime variance model

The release performance observation is the paired log ratio
`log(Tgrad prepared runtime / tinygrad TinyJit replay)` from
`scripts/perf/paired_runtime.py`. Pair order is balanced within a logical
session. The estimand gives each logical session equal weight. Uncertainty is
estimated by resampling sessions first and paired observations second.

Release calibration and release evaluation are different artifact sets.
Calibration must contain at least three independent process invocations, each
with at least three logical sessions of at least 30 paired samples. A reviewed
decision rule may be derived from those calibration artifacts only. Evaluation
then uses at least three new independent invocations, the same minimum session
and sample counts, at least 2,000 hierarchical bootstrap resamples, and a 95%
interval. No minimum sample, frozen tinygrad fixture, or single-run extremum
may be used as a release denominator.

The model distinguishes within-session dispersion, between-session
dispersion, and the hierarchical bootstrap interval. It never describes this
prepared-runtime boundary as isolated kernel speed.

The release statistic is the maximum, over independent evaluation runs, of
the upper endpoint of each run's hierarchical-bootstrap geometric-mean ratio.
The threshold is absent until calibration has measured its variability and a
human review promotes that rule. The certificate only names completed artifact
sets; run/session counts, statistics, and acceptance are re-derived from their
raw observation streams.
