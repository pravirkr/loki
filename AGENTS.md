# Working on Loki

Read `docs/architecture.md` before changing code. It is the frozen layout and
the rules every change must follow (public API, dispatch, includes, tests,
namespaces, numerics). Its recipes cover adding an algorithm or a backend.

Before finishing a change:

- `scripts/check_architecture.sh` passes;
- `pre-commit run --files <changed files>` passes;
- `cmake --preset dev && cmake --build --preset dev && ctest --preset dev`
  passes; for Python, copy `build-dev/src/libloki*.so` into `src/loki/` and
  run `PYTHONPATH=src pytest tests/python`.

Do not change kernel math, RNG streams or output formats as a side effect of
a layout or API change. Do not stage or commit unless asked.
