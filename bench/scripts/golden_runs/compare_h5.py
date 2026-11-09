"""Compare two golden output trees dataset by dataset (bit-exact unless --rtol).

Usage: python compare_h5.py <dirA> <dirB> [--rtol R] [--ignore-attr NAME ...]
Attributes that record time, host, version, toolchain or run provenance (the
backend and config text) are skipped by default.
Exit 0 when every file, dataset and attribute matches.
"""

import argparse
import pathlib
import sys

import h5py
import numpy as np

VOLATILE = (
    "time",
    "date",
    "host",
    "version",
    "duration",
    "elapsed",
    "created",
    "wall",
    "compiler",
    "cuda",
    "build",
    "command",
    "config_path",
    "path",
    "outdir",
    "timeseries",
    "device_name",
    # Provenance: EP files gained these in ab3637c, and the effective config
    # names new keys as versions add them. What the run computed is compared
    # through the datasets.
    "backend",
    "config_toml",
    "config_effective",
)


def volatile(name: str, extra: list[str]) -> bool:
    low = name.lower()
    return any(v in low for v in VOLATILE) or name in extra


ROOTS: list[str] = ["", ""]


def norm(x, side: int):
    """Replace the tree's own root path in strings, so A and B compare equal."""
    if isinstance(x, bytes):
        x = x.decode(errors="replace")
    if isinstance(x, str) and ROOTS[side]:
        return x.replace(ROOTS[side], "<ROOT>")
    return x


def compare_values(a, b, rtol: float) -> bool:
    a = np.asarray(norm(a, 0) if isinstance(a, (str, bytes)) else a)
    b = np.asarray(norm(b, 1) if isinstance(b, (str, bytes)) else b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind in "fc" and rtol > 0:
        return bool(np.allclose(a, b, rtol=rtol, atol=0, equal_nan=True))
    if a.dtype.kind in "fc":
        return a.tobytes() == b.tobytes()
    if a.dtype.names:
        return (
            a.tobytes() == b.tobytes()
            if rtol == 0
            else all(compare_values(a[n], b[n], rtol) for n in a.dtype.names)
        )
    return bool(np.array_equal(a, b))


def compare_file(fa, fb, rtol, extra, report):
    ok = True

    def attrs(path, oa, ob):
        nonlocal ok
        for k in sorted(set(oa.attrs) | set(ob.attrs)):
            if volatile(k, extra):
                continue
            if k not in oa.attrs or k not in ob.attrs:
                report(f"  attr {path}@{k}: missing on one side")
                ok = False
            elif not compare_values(oa.attrs[k], ob.attrs[k], rtol):
                report(
                    f"  attr {path}@{k}: {str(oa.attrs[k])[:120]!r} != "
                    f"{str(ob.attrs[k])[:120]!r}"
                )
                ok = False

    names_a, names_b = set(), set()
    fa.visit(names_a.add)
    fb.visit(names_b.add)
    for n in sorted(names_a ^ names_b):
        report(f"  object {n}: only in {'A' if n in names_a else 'B'}")
        ok = False
    attrs("/", fa, fb)
    for n in sorted(names_a & names_b):
        oa, ob = fa[n], fb[n]
        attrs(n, oa, ob)
        if isinstance(oa, h5py.Dataset):
            if volatile(n.rsplit("/", 1)[-1], extra):
                continue
            if not compare_values(oa[()], ob[()], rtol):
                report(f"  dataset {n}: differs (shape {oa.shape})")
                ok = False
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--rtol", type=float, default=0.0)
    p.add_argument("--ignore-attr", nargs="*", default=[])
    args = p.parse_args()
    da, db = pathlib.Path(args.a).resolve(), pathlib.Path(args.b).resolve()
    ROOTS[0], ROOTS[1] = str(da), str(db)
    files = sorted(
        {f.relative_to(da) for f in da.rglob("*.h5")}
        | {f.relative_to(db) for f in db.rglob("*.h5")}
    )
    all_ok = True
    for rel in files:
        if not (da / rel).exists() or not (db / rel).exists():
            print(f"{rel}: missing on one side")
            all_ok = False
            continue
        lines = []
        with h5py.File(da / rel) as fa, h5py.File(db / rel) as fb:
            ok = compare_file(fa, fb, args.rtol, args.ignore_attr, lines.append)
        print(f"{rel}: {'identical' if ok else 'DIFFERENT'}")
        for line in lines[:20]:
            print(line)
        all_ok &= ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
