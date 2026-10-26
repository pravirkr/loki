# loki logo source

The wordmark is LOKI built from geometry, with no font: cap height 100 and one
stroke weight. The O is a binary pulsar in a circular orbit. The ring is the
orbit, the orange pulsar sits on it with its two beams, and the dot at the
centre is the companion. The O alone is the mark and the icon. The orange is
the CLI progress-bar accent (`kDarkOrange` / `kOrange` in
`lib/detail/progress.hpp`).

## Regenerate

```bash
uv run docs/logo-src/make_logo.py            # rewrites docs/_static/logo/*.svg
uv run docs/logo-src/make_logo.py --exports  # also writes build/ (PNG, ICO, mono, social)
```

## Files in the repo

Only the SVGs are committed:

| File | Use |
| --- | --- |
| `loki-logo.svg`, `loki-logo-dark.svg` | `README.md` header |
| `loki-mark.svg`, `loki-mark-dark.svg` | The O alone (docs, slides) |
| `favicon.svg` | Favicon (follows the browser's light/dark preference) |

Everything else comes from `--exports`: high-res PNGs, mono variants,
rounded-tile icons (GitHub avatar, conda-forge), `favicon.ico`,
`apple-touch-icon.png`, and `social-preview.png` (1280×640, for GitHub
Settings → Social preview).

## Parameters

Lengths are in units of the cap height (100); pulsar sizes are in units of the
stroke weight, so they scale with it.

| Constant | Value | Meaning |
| --- | --- | --- |
| `STROKE` | 12 | Letter stroke weight |
| `SMALL_STROKE` | 15 | Heavier O for the favicon and touch icon, so it survives 16 px |
| `PULSAR_ANGLE` | −45° | Pulsar position on the orbit (upper right) |
| `PULSAR_R` | 0.95 | Pulsar radius |
| `BEAM_REACH` | 2.0 | Beam length beyond the pulsar's edge |
| `BEAM_TILT`, `BEAM_HALF` | 24°, 10.5° | Beam axis against the orbit tangent; half-opening |
| `COMPANION_R` | 0.6 | Companion radius |
| `ACCENT` / `ACCENT_D` | `#E6722A` / `#FF8C42` | Pulsar on light / dark |
| `INK` / `INK_D` | `#12151B` / `#EDF0F4` | Letters on light / dark |
