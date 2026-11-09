# Working on Loki

Read `docs/architecture.md` before changing code. It is the frozen layout and
the rules every change must follow (public API, dispatch, includes, tests,
namespaces, numerics). Its recipes cover adding an algorithm or a backend.
Before any performance work read `docs/performance.md`: the numerics
contract, the engineering rules, and the ideas already measured and
rejected. Add every new measured result there.

Before finishing a change:

- `scripts/check_architecture.sh` passes;
- `pre-commit run --files <changed files>` passes;
- `cmake --preset release && cmake --build --preset release &&
  ctest --preset release -j 8` passes; for Python, run
  `PYTHONPATH=src pytest -n auto tests/python` (the build copies the module
  into `src/loki/`). The `release` preset builds into `build/` (with CUDA
  when a toolkit is found). Never run the C++ or Python tests on a Debug
  build (`ci-tidy`): they are far too slow there.

Do not change kernel math, RNG streams or output formats as a side effect of
a layout or API change. Do not stage or commit unless asked.
