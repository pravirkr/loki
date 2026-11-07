# Segment layout

The FFA folds a time series in a dyadic tree. The brute fold splits the
series into short segments and folds each one; every FFA level then merges
neighbouring pairs, halving the number of segments and doubling their
length. This page describes how Loki chooses that layout when `nsamps` is
not a power of two, and how the FFA search and the EP search treat the
samples that do not fill it.

Code: `FFASearchConfig::Impl` in `lib/search/configs.cpp` (`resolve_bseg_brute`,
`candidate_bseg_brute`, `truncate_to_whole_segments`) and `ep_input_series` in
`lib/search/ep_support.hpp`.

## Two settings

| Setting | Meaning | Default |
|---|---|---|
| `bseg_brute` (`[performance]`, `--bseg-brute`) | **Minimum** brute-fold segment length, in samples | Chosen per frequency chunk from a cycles-per-segment target at `f_max` (or a cost model for the direct-DFT Fourier fold) |
| `nsegments` (`[ep]`, `--nsegments`, EP only) | Segments the EP pruning starts from: the FFA stops when it has merged the brute segments down to this many. A power of 2, at least 2 | 128 |

`loki search ffa` always merges down to one segment and has no
`nsegments` setting. `bseg_ffa` is no longer a setting: it is derived as
`bseg_brute * 2^niters_ffa`, the length of the segments the FFA stops at,
and is still reported by `FFASearchConfig::get_bseg_ffa()`, the Python
`cfg.bseg_ffa` property and the result files. A TOML file that still sets
`performance.bseg_ffa` is rejected with a hint to use `ep.nsegments`.

## Equal segments

The brute fold always produces 2^L segments of the same whole number of
samples, so every FFA merge pairs two equal halves and every level shares one
set of resolve tables. For a minimum length `b`, L is the largest value whose
segment length is still at least `b`:

- **FFA search** (one final segment): segment length `ceil(nsamps / 2^L)`.
  The tree covers `2^L * ceil(nsamps / 2^L)` samples; the last fewer than
  2^L of them are zero-filled.
- **EP search** (`nsegments > 1`): segment length `floor(nsamps / 2^L)`. The
  search analyses the leading `2^L * floor(nsamps / 2^L)` samples and drops
  the last fewer than 2^L (see [EP truncation](#ep-truncation)).

Either way, the samples padded or dropped are fewer than 2^L, a fraction
below `1 / bseg_brute` of the series. A power-of-two `nsamps` with a
power-of-two `bseg_brute` pads or drops nothing.

Examples with `bseg_brute = 512`:

| nsamps | Brute segments | FFA search | EP search |
|---|---|---|---|
| 2^23 | 16384 x 512 | no padding | nothing dropped |
| 2^23 + 1 | 16384 x 513 (FFA), 16384 x 512 (EP) | 16,383 zero samples | 1 sample dropped |
| 12,000,000 | 16384 x 733 (FFA), 16384 x 732 (EP) | 9,472 zero samples | 6,912 samples dropped |

Before this layout the tree had `bit_ceil(ceil(nsamps / bseg_brute))`
segments of exactly `bseg_brute` samples, which could pad a series to nearly
twice its length (2^23 + 1 samples became 32768 x 512 = 2^24).

The effective length maps to itself: a config built from its own
`get_bseg_brute()` is the same config.

## FFA search: zero-filling costs no S/N

The brute fold reads only the `nsamps` samples of the series, so the
segments past it stay zero in both the signal (E) and the variance (V)
channel. Every merge adds those zeros, so the final profile has the same
signal and variance sums as an unpadded fold, and the score E / sqrt(V) is
unchanged. No mask is needed. The two side effects are harmless:

- The tree is slightly longer than the series, so its midpoint is later
  than the observation midpoint. Reported candidates are shifted back to
  the observation midpoint (`delta_t_epoch`, `lib/search/ffa_sweep_candidates.hpp`).
- The parameter grid is sized for the nominal segment length, a little
  longer than the data in the last segments, so it is slightly finer than
  needed: a few more trials, never a coarser grid.

## EP truncation

The EP pruning thresholds assume every segment brings the same expected
signal. A partly zero-filled segment would bring less, and a run that starts
next to it would be pruned against thresholds calibrated for more signal
than the data holds. EP therefore keeps whole segments only: with
`nsegments > 1` the config analyses the leading `2^L * floor(nsamps / 2^L)`
samples, so all `nsegments` EP segments are full and of equal length.

- `FFASearchConfig::get_nsamps()` is the analysed length, and
  `get_tobs()`, the FFA plan and the epoch reference all follow it, so the
  tree midpoint is the midpoint of the analysed data.
  `get_nsamps_input()` (Python `cfg.nsamps_input`) is the length the config
  was built for.
- The EP entry points (`EPFreqSweep::execute`, `EPMultiPass::execute`, CPU and
  CUDA) take a series of either length and use its leading `get_nsamps()`
  samples, before the brute fold. When they drop samples they log it at
  info level, for example:

  ```text
  EPFreqSweep::execute: EP analyses the first 34816 of 35800 samples
  (128 segments of 272); the last 984 samples fill no brute segment
  ```

- The candidate file still records the input length (`original_nsamps`).

All frequency chunks of an EP sweep share one EP segment length: a chunk
chooses its own brute length, but only among `bseg_ffa >> k`. Segment k
therefore covers the same samples in every chunk, and every chunk drops the
same tail.

### How much is dropped

The finest tree of the sweep sets the drop: the chunk at the top of the band
has the shortest brute segments and so the largest 2^L. A 1–500 Hz
acceleration search at 64 us with `nbins = 31` uses brute segments of about
256 samples there:

| nsamps | Dropped (worst case) |
|---|---|
| any power of two | 0 |
| 2^24 - 1 | 65,535 samples (0.39%) |
| 2^27 - 1 | 524,287 samples (33.6 s, 0.39%) |
| random lengths in [2^26, 2^27) | 0.21% on average |

The S/N cost is about half the dropped fraction: at most ~0.2% here.
