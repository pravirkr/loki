# Example notebooks

Short, self-contained examples of loki's searches on simulated data. Each
notebook simulates its data with a fixed seed, so a rerun reproduces the
saved outputs, runs one search and checks the result: the recovered
parameters against the injected pulsar, or, for pure noise (notebook 5),
the scores against the expected null distribution.

| notebook | what it shows | runtime |
| --- | --- | --- |
| [01_ffa_frequency](01_ffa_frequency.ipynb) | P-FFA over frequency; time- vs Fourier-domain shifts | 10 s |
| [02_ffa_acceleration](02_ffa_acceleration.ipynb) | P-FFA over acceleration and frequency | 10 s |
| [03_ffa_jerk](03_ffa_jerk.ipynb) | P-FFA over jerk, acceleration and frequency | 10 s |
| [04_ffa_pipeline](04_ffa_pipeline.ipynb) | the production P-FFA sweep: band planning, candidate clustering, harmonic flagging, PulsarX export | 10 s |
| [05_noise_statistics](05_noise_statistics.ipynb) | false alarms: P-FFA scores on pure noise against the null distribution | 10 s |
| [06_pruning_thresholds](06_pruning_thresholds.ipynb) | designing the EP pruning thresholds | 30 s |
| [07_ep_acceleration](07_ep_acceleration.ipynb) | EP search, 1500 m/s² at S/N 8.5; Taylor basis | 1.5 min |
| [08_ep_jerk](08_ep_jerk.ipynb) | EP search over jerk, acceleration and frequency | 3 min |
| [09_ep_circular](09_ep_circular.ipynb) | EP search over five parameters for a 20-minute circular orbit | 7 min |

Runtimes in the table are on an Apple M5 with 4 threads (`NTHREADS` in each
notebook), CPU only. The search times printed inside the notebooks are from
the x86 machine that produced the saved outputs (next section).
The EP notebooks (7–9) need up to about 11 GB of memory, the others at most 6 GB.

## Reading the saved outputs

The saved outputs come from one machine: an AMD EPYC 9334 (x86-64, GCC 12,
`-march=native`) with 4 threads, CPU only. Every random draw is seeded
(`SEED`: the simulated pulsar and noise, the threshold design and its
evaluation), so a rerun on the same machine, build and `NTHREADS`
reproduces them exactly.

On another CPU or compiler, expect small differences. Release builds use
`-ffast-math` and SIMD, so floating-point results can differ in the last
bits between platforms (see "Numerics" in `docs/architecture.md`):

- FFA notebooks (1–5): best-fit parameters and S/N can differ in the last
  printed digits.
- EP notebooks (6–9): the threshold design and the trimming at `max_sugg`
  amplify those last bits. Expect slightly different P_d and cost figures
  in notebook 6, and in notebook 9 a different best run and a different
  count of runs that kept the pulsar (3 of 16 on an Apple M5, 2 of 16
  here).

Compare such results by what they show (the pulsar recovered near the
injected parameters, the score distribution against the null model), not
digit for digit. Search times printed in the notebooks are wall time on
the machine that ran them.

## Running

Install loki with the notebook extras, then start Jupyter in this
directory:

```bash
uv pip install ".[notebooks]"
```

```bash
jupyter lab notebooks/
```

The notebooks write their result files to `notebooks/_output/`, which git
ignores. They use [pyloki](https://github.com/pravirkr/pyloki), a loki
dependency, to simulate the data and plot the results.

To re-execute all notebooks and save their outputs:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```
