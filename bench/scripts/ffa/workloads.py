"""FFA benchmark workloads W1-W4 (and the L / n variants): data, configs, runs.

Usage::

    python workloads.py data   LOKI
    python workloads.py config LOKI [--backend cuda] [--memgb 10] [--nthreads 1]
    python workloads.py run    LOKI TAG W1 W3L ...

LOKI is the path of an ``applications/loki`` binary. Everything is written
under ``$LOKI_BENCH_WORK`` (default: the current directory): ``data/`` (the
simulated inputs), ``cfg/`` (one TOML per workload, from ``loki search ffa
-g``), ``runs/<TAG>_<W>/`` (results and logs). ``run`` prints the sweep timer,
the wall time, the peak RSS of the run and the chunk count. Compare two result
files with ``compare_ffa.py``.

| Name | Search | Domain | Samples |
|---|---|---|---|
| W1 / W2 | freq-only, 0.5-100 Hz | time / Fourier | 2^23 |
| W3 / W4 | accel +-100 m/s^2, 10-100 Hz | time / Fourier | 2^23 |
| W1L-W4L | as above | as above | 2^25 |
| W2n, W4n, W2Ln, W4Ln | nbins 50, nbins_max 800 | Fourier | |
"""

from __future__ import annotations

import argparse
import os
import re
import resource
import subprocess
import time
from pathlib import Path

WORK = Path(os.environ.get("LOKI_BENCH_WORK", "."))

# Per workload: input file, f_min, f_max, acc_min, acc_max, use_fourier.
WORKLOADS = {
    "W1": ("pulse23.tim", 0.5, 100.0, 0.0, 0.0, "false"),
    "W2": ("pulse23.tim", 0.5, 100.0, 0.0, 0.0, "true"),
    "W3": ("accel23.tim", 10.0, 100.0, -100.0, 100.0, "false"),
    "W4": ("accel23.tim", 10.0, 100.0, -100.0, 100.0, "true"),
    "W1L": ("pulse25.tim", 0.5, 100.0, 0.0, 0.0, "false"),
    "W2L": ("pulse25.tim", 0.5, 100.0, 0.0, 0.0, "true"),
    "W3L": ("accel25.tim", 10.0, 100.0, -100.0, 100.0, "false"),
    "W4L": ("accel25.tim", 10.0, 100.0, -100.0, 100.0, "true"),
}
# Non-power-of-two nbins variants (Bluestein paths).
for _w in ("W2", "W4", "W2L", "W4L"):
    WORKLOADS[_w + "n"] = WORKLOADS[_w]

# input: (period, nsamps, accel, seed); dt 64 us, S/N 12, duty cycle 0.05.
INPUTS = {
    "pulse23.tim": (0.0731, 2**23, 0.0, 7),
    "accel23.tim": (0.0123, 2**23, 50.0, 11),
    "pulse25.tim": (0.0731, 2**25, 0.0, 7),
    "accel25.tim": (0.0123, 2**25, 20.0, 11),
}


def set_key(text: str, key: str, value: object) -> str:
    """Replace the value of ``key = ...`` (exactly one line) in a TOML text."""
    out, n = re.subn(rf"^{key} = .*$", f"{key} = {value}", text, flags=re.MULTILINE)
    if n != 1:
        msg = f"{key}: {n} matches in the template"
        raise ValueError(msg)
    return out


def make_data(loki: str) -> None:
    data = WORK / "data"
    data.mkdir(parents=True, exist_ok=True)
    for name, (period, nsamps, acc, seed) in INPUTS.items():
        if (data / name).exists():
            continue
        cmd = [loki, "simulate", "--period", str(period), "--dt", "0.000064"]
        cmd += ["--nsamps", str(nsamps), "--snr", "12", "--ducy", "0.05"]
        cmd += ["--acc", str(acc), "--seed", str(seed), "-o", str(data / name)]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"wrote {data / name}")


def make_configs(loki: str, backend: str, memgb: float, nthreads: int) -> None:
    cfg_dir = WORK / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    template_path = cfg_dir / "template.toml"
    subprocess.run(
        [loki, "search", "ffa", "-g", str(template_path)],
        check=True,
        capture_output=True,
    )
    template = template_path.read_text()
    data = (WORK / "data").resolve()
    for name, (inp, f0, f1, a0, a1, fourier) in WORKLOADS.items():
        t = set_key(template, "timeseries", f'"{data / inp}"')
        for key, value in (
            ("f_min", f0),
            ("f_max", f1),
            ("acc_min", a0),
            ("acc_max", a1),
            ("use_fourier", fourier),
            ("nthreads", nthreads),
            ("max_process_memory_gb", memgb),
            ("backend", f'"{backend}"'),
            ("outdir", '"./"'),
            ("prefix", f'"{name}"'),
        ):
            t = set_key(t, key, value)
        if name.endswith("n"):
            t = set_key(set_key(t, "nbins", 50), "nbins_max", 800)
        (cfg_dir / f"{name}.toml").write_text(t)
    print(f"wrote {len(WORKLOADS)} configs to {cfg_dir} ({backend}, {memgb} GB)")


def run(loki: str, tag: str, names: list[str]) -> None:
    loki = str(Path(loki).resolve())
    for name in names:
        out = WORK / "runs" / f"{tag}_{name}"
        out.mkdir(parents=True, exist_ok=True)
        cfg = (WORK / "cfg" / f"{name}.toml").resolve()
        t0 = time.perf_counter()
        with (out / "log.txt").open("w") as log:
            subprocess.run(
                [loki, "search", "ffa", "-c", str(cfg)],
                cwd=out,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        wall = time.perf_counter() - t0
        # Peak RSS of the largest child so far (KiB on Linux).
        rss_gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20
        text = (out / "log.txt").read_text(errors="replace")
        timer = re.search(r"FFA Freq Sweep: timer.*", text)
        chunks = re.search(r"Total chunks: (\d+)", text)
        print(
            f"{tag} {name}: {timer.group(0) if timer else 'no timer (failed?)'}"
            f" | wall {wall:.2f} s | peak RSS so far {rss_gb:.2f} GB"
            f" | chunks {chunks.group(1) if chunks else '?'}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("data")
    p.add_argument("loki")
    p = sub.add_parser("config")
    p.add_argument("loki")
    p.add_argument("--backend", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--memgb", type=float, default=10.0)
    p.add_argument("--nthreads", type=int, default=1)
    p = sub.add_parser("run")
    p.add_argument("loki")
    p.add_argument("tag")
    p.add_argument("workloads", nargs="+", choices=sorted(WORKLOADS))
    args = parser.parse_args()
    if args.cmd == "data":
        make_data(args.loki)
    elif args.cmd == "config":
        make_configs(args.loki, args.backend, args.memgb, args.nthreads)
    else:
        run(args.loki, args.tag, args.workloads)


if __name__ == "__main__":
    main()
