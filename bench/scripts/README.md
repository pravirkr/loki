# Benchmark and validation scripts

Python tools for timing loki and checking that a change keeps the science.
They write under `$LOKI_BENCH_WORK` (default: the current directory), never
next to the scripts. Run them with plain `python` and, for a package copy,
`PYTHONPATH`: `python -I` ignores `PYTHONPATH` and silently imports an
installed loki instead.

## Numerics contract

A change that is meant to keep the arithmetic (layout, fusion, host work)
must be bit-exact. A change that moves floats (phases, FFTs, summation
order) is acceptable when the top-N candidates are identical, the Jaccard
overlap at the production threshold is at least 99.9%, matched scores agree
within 1e-5 relative (or within the CPU-GPU spread for Fourier folds), and
injected S/N moves by under 0.1%. A change that moves chunk widths or accel
trials needs an injection study at 2^25 samples. Time-domain GPU outputs
jitter run to run (float atomics in the brute fold), so compare a change
against the old-vs-old jitter, not against zero.

## FFA (`ffa/`)

| Script | Purpose |
|---|---|
| `workloads.py` | The W1-W4 workloads (freq-only / accel, time / Fourier, 2^23 / 2^25, nbins 64.. or 50..): `data` simulates the inputs, `config` writes one TOML per workload from `loki search ffa -g` (`--backend`, `--memgb`, `--nthreads`), `run` times a binary on a list of workloads. |
| `compare_ffa.py` | Two FFA result files under the contract: Jaccard, matched dS/N, top-N, best candidate. |
| `injection_study.py` | Random accelerated pulsars searched by two builds; recovered S/N per case and the mean difference. |

```bash
export LOKI_BENCH_WORK=/scratch/loki-bench
python bench/scripts/ffa/workloads.py data   build-dev/applications/loki
python bench/scripts/ffa/workloads.py config build-dev/applications/loki --backend cuda
python bench/scripts/ffa/workloads.py run    base/loki base W3 W4
python bench/scripts/ffa/workloads.py run    build-dev/applications/loki new W3 W4
python bench/scripts/ffa/compare_ffa.py $LOKI_BENCH_WORK/runs/{base,new}_W4/W4_ffa_results.h5
```

## EP (`ep/`)

| Script | Purpose |
|---|---|
| `jerk_test.py` | `simulate` the 2^25-sample jerk pulsar (needs pyloki); `run` a fixed-threshold `EPMultiPass` (3 runs, 140-145 Hz) on a backend, domain and nbins, and print the execute time. |
| `compare_multipass.py` | Two `EPMultiPass` results run by run: bit-exactness, level counts, Jaccard, matched dS/N, top-10. |
| `compare_sweep.py` | Two `EPFreqSweep` results: candidate sets and level statistics within a tolerance. |
| `../ep_production_cuda.py`, `../measure_ep_peak_rss.py`, `../thresholds_check_cuda.py`, `../golden_schemes.py`, `../bias_study.py` (top level) | Planned EP sweeps, the memory model against peak RSS / device memory, and threshold-scheme checks (see each file's docstring). |

## Profiling (`common/`)

`analyze_nsys.py REPORT.sqlite [--cpu]`: GPU busy fraction over the EP prune
window from an nsys export, and with `--cpu` the hottest host symbols.
Build with `-DLOKI_ENABLE_NVTX=ON` for named ranges. For kernel timing with
ncu, use `--cache-control none --clock-control none`.
