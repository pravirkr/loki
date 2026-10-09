# Preprocessing

`loki::io::preprocess()` turns a raw dedispersed timeseries `T` into the two
arrays every fold and score kernel consumes:

```text
ts_e[n] = c * (T[n] - mu[n]) * g[n] / sigma[n]^2
ts_v[n] = c^2 * g[n]^2 / sigma[n]^2
```

- `mu` is the local baseline and `sigma^2` the local noise variance.
- `g` is the gain model. It is `1` for additive noise, and `mu` when the
  signal scales with the baseline (multiplicative).
- Masked or clipped samples get `ts_e = ts_v = 0`.
- `c` normalises the median nonzero `ts_v` to 1. A folded profile is scored
  per bin as `P_e / sqrt(P_v)`, which does not depend on `c`.

These are the sufficient statistics of a matched filter under heteroscedastic
Gaussian noise. Folding them with the existing kernels gives inverse-variance
weighting and exact masking, with no change to the fold or score math.

The search CLI runs it on the host CPU in `load_search_timeseries`. It reads
the raw series, then calls `TimeSeries::preprocess` with the `[preprocessing]`
table.

## Methods

| `method` | Output | Use |
| --- | --- | --- |
| `"robust"` (default) | dual arrays, as above | searches |
| `"zscore"` | legacy running-median detrend, one global z-score, `ts_v = 1` | backup and verification. It reproduces the pre-robust pipeline bit for bit. |

## Robust algorithm

All passes are `O(N)` and run in parallel over fixed blocks. The output does
not depend on the thread count.

1. **Geometry.**
   - The baseline window `W` is in seconds; see "Baseline window" below.
   - Statistics blocks have `B = W / window_blocks` samples (at least 32).
     Each running filter therefore works on about `window_blocks` block
     values, not on `N` samples.
2. **Statistics.**
   - Block medians over valid samples feed a masked running median over
     `window_blocks` blocks, then a short running mean. The mean removes the
     staircase that a running median leaves on a drifting baseline.
   - The result is interpolated linearly between block centres.
   - Edges use point reflection, so a linear trend has no edge bias.
   - The variance per block is the squared MAD, with a small-sample
     correction. It is averaged over the variance window and floored at
     `1e-6` times the median variance.
3. **Bad blocks.** These are computed on whitened residuals
   `z = (T - mu) / sigma`.
   - At each `block_scales` length, each block is scored on two statistics:
     - its mean, `sqrt(n) * mean(z)`;
     - its variance, the Wilson–Hilferty cube-root transform of
       `mean(z^2)`.
   - Single-sample outliers (`|z| > clip_sigma`) are left out of these
     statistics, so they are clipped rather than masked.
   - Each statistic is converted to a robust z-score across all blocks of
     that scale (median, 1.4826 MAD).
   - A block is masked when either z-score exceeds `block_sigma`, or when
     its valid fraction is below `min_good_fraction`.
   - Stages 2 and 3 are repeated `n_iter` times, so the baseline and the
     mask converge together.
4. **Periodic zap** (optional).
   - The valid `z` is Fourier transformed.
   - The power is whitened by its running median over `zap_whiten_bins`
     (divided by `ln 2`, so it follows Exp(1)).
   - Flagged bins are those above the Gaussian-equivalent `zap_sigma`,
     plus the `birdies`.
   - A flagged bin's amplitude is scaled down to the local noise level, and
     its phase is kept.
   - The cleaned series goes back through stage 2.
5. **Output.**
   - Samples with `|T - mu| > clip_sigma * sigma` are zeroed.
   - Masked samples are zeroed.
   - All other samples get the formulas above.

## Configuration (`[preprocessing]`)

| Key | Default | Meaning |
| --- | --- | --- |
| `preprocess` | `true` | `false` folds the raw series with `ts_v = 1` |
| `method` | `"robust"` | `"robust"` or `"zscore"` |
| `gain_model` | `"additive"` | `"additive"` or `"multiplicative"`. Multiplicative needs a positive baseline. |
| `filter_window` | `1.0` | baseline window, in s. `0` means one baseline for the whole series. |
| `min_window_periods` | `10.0` | the effective window is at least this divided by `f_min`. `0` disables this. |
| `variance_window` | `0.0` | robust: variance window in s. `0` means the effective baseline window. |
| `window_blocks` | `101` | robust: statistics blocks per baseline window |
| `fast_median` | `true` | zscore: block-averaged running median |
| `fast_median_min_points` | `101` | zscore: short-series width of `fast_median` |
| `n_iter` | `2` | robust: rounds of statistics followed by bad-block flagging |
| `block_scales` | `[0.016, 0.065, 0.26, 1.05]` | robust: bad-block lengths, in s. `[]` disables masking. |
| `block_sigma` | `6.0` | robust: robust z-score threshold for flagging a block |
| `min_good_fraction` | `0.3` | robust: blocks with fewer valid samples than this fraction are masked |
| `clip_sigma` | `6.0` | robust: per-sample clip, in local sigmas. `0` disables it. |
| `zap_periodic` | `false` | robust: threshold zap of the whitened spectrum |
| `zap_sigma` | `8.0` | robust: zap threshold, as a Gaussian-equivalent significance |
| `zap_whiten_bins` | `1001` | robust: running-median width of the spectrum, in bins |
| `birdies` | `[]` | robust: `[[freq_hz, width_hz], ...]`, always zapped |

Moved keys are rejected with a hint. The old `[input]` keys `preprocess`,
`filter_window`, `fast_median` and `fast_median_min_points` must now be
written in `[preprocessing]`.

The CLI flags are `--preprocess/--no-preprocess`, `--preproc-method
robust|zscore`, `--filter-window`, `--fast-median/--no-fast-median` and
`--fast-median-min-points`. Every other key is set in the TOML file.

## Guidance

- **Baseline window.**
  - A running median of width `W` absorbs any periodic signal with period
    comparable to `W`.
  - `min_window_periods = 10` keeps `W >= 10 / f_min`. With the default
    `f_min = 0.5 Hz`, a 1 s request becomes 20 s.
  - Raise the window when red noise is weak. Lower `min_window_periods`
    only when slow pulsars are not of interest.
- **Periodic zap.**
  - Leave `zap_periodic` off when searching for bright pulsars. Their
    harmonics are strong, narrow lines and can exceed `zap_sigma`.
  - Prefer `birdies` for known mains and instrumental lines.
- **Masking scales.**
  - `block_scales` should bracket the RFI durations you expect.
  - Scales shorter than 8 samples, or with fewer than 8 blocks in the
    series, are skipped.
- **Red noise.**
  - Wander slower than the variance window is absorbed by the baseline.
  - Wander faster than it shows up as sub-window offsets. Very strong
    red noise can therefore get masked; lengthen `filter_window` or raise
    `block_sigma` if the report's masked fraction is high.
- **Report.**
  - The CLI logs the effective windows, the masked percentage, the clipped
    sample count, the zapped bins and the longest masked run.
  - `io::PreprocessReport` also holds the per-block baseline, sigma and
    valid fraction.

## Validity contract

The fold and score kernels assume finite inputs, and `P_v > 0` in every fold
bin that they score. They do not check this. Preprocessing guarantees, once at
load time:

- the outputs are finite and `ts_v >= 0`;
- only masked or clipped samples have `ts_v = 0`, and they have `ts_e = 0`;
- the variance is floored relative to the global variance;
- an all-masked or constant series throws.

`TimeSeries` checks the same rules on construction.

A fold bin can still have zero weight if a masked run covers almost a whole
brute-fold segment. The CLI warns when the longest masked run reaches half a
brute-fold segment.
