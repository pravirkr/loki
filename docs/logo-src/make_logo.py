# /// script
# requires-python = ">=3.10"
# dependencies = ["resvg-py", "pillow"]
# ///
"""Generate the loki logo.

The wordmark is LOKI built from geometry, no font: cap height 100, one stroke
weight. The O is a binary pulsar in a circular orbit: the ring is the orbit,
the orange pulsar sits on it with its beams, and the companion is the dot at
the centre. The O alone is the mark and the icon.

Usage::

    uv run docs/logo-src/make_logo.py            # SVGs used by README + docs
    uv run docs/logo-src/make_logo.py --exports  # also PNG/ICO/mono/social exports

The first form rewrites ``docs/_static/logo/``. Exports go to
``docs/logo-src/build/`` (git-ignored).
"""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path
from typing import TYPE_CHECKING

import resvg_py
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Sequence

HERE = Path(__file__).resolve().parent
STATIC = HERE.parent / "_static" / "logo"
BUILD = HERE / "build"

# Palette. Accent is the pulsar only: the TVA oranges of the CLI progress bar
# (kDarkOrange, kOrange in lib/detail/progress.hpp).
INK, ACCENT = "#12151B", "#E6722A"
INK_D, ACCENT_D = "#EDF0F4", "#FF8C42"
TILE_L, TILE_D = "#FFFFFF", "#0E1117"

# Letter geometry: cap line at y=0, baseline at y=100.
STROKE = 12.0
SMALL_STROKE = 15.0  # favicon and touch icon, so the O survives 16 px
L_WIDTH = 56
O_RADIUS = 52  # outer radius; overshoots the cap line and baseline by 2
GAP_LO, GAP_OK, GAP_KI = 15, 17, 17

# The pulsar and companion, in units of the stroke weight.
PULSAR_ANGLE = -45.0  # position on the orbit, degrees (y down: upper right)
PULSAR_R = 0.95
BEAM_REACH = 2.0  # beam length beyond the pulsar's edge
BEAM_TILT = 24.0  # beam axis relative to the orbit tangent, degrees
BEAM_HALF = 10.5  # half-opening of each beam, degrees
COMPANION_R = 0.6


def fmt(x: float) -> str:
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def poly(pts: Sequence[tuple[float, float]]) -> str:
    return "M" + "L".join(f"{fmt(x)},{fmt(y)}" for x, y in pts) + "Z"


def circle(cx: float, cy: float, r: float, fill: str) -> str:
    return f'<circle cx="{fmt(cx)}" cy="{fmt(cy)}" r="{fmt(r)}" fill="{fill}"/>'


# ---------------------------------------------------------------- letters
def pulsar(cx: float, t: float) -> tuple[float, float, float, list[list[tuple]]]:
    """Pulsar centre, radius and its two beam triangles for the O at (cx, 50)."""
    rc = O_RADIUS - t / 2
    a = math.radians(PULSAR_ANGLE)
    px, py = cx + rc * math.cos(a), 50 + rc * math.sin(a)
    rp = t * PULSAR_R
    axis = a + math.pi / 2 + math.radians(BEAM_TILT)
    reach, h = t * BEAM_REACH + rp, math.radians(BEAM_HALF)
    beams = [
        [(px, py)]
        + [
            (px + reach * math.cos(s + d), py + reach * math.sin(s + d))
            for d in (-h, h)
        ]
        for s in (axis, axis + math.pi)
    ]
    return px, py, rp, beams


def o_body(cx: float, ink: str, accent: str, t: float | None = None) -> str:
    """Draw the O centred on (cx, 50).

    ``t`` is the stroke weight (default ``STROKE``); beams, pulsar and companion
    scale with it, so a heavier ``t`` gives an icon for small sizes.
    """
    t = t or STROKE
    cy, ro = 50, O_RADIUS
    ri, rc = ro - t, ro - t / 2
    px, py, rp, beams = pulsar(cx, t)
    a = math.radians(PULSAR_ANGLE)
    da = (rp + t * 0.55) / rc  # half-gap in the orbit around the pulsar
    a0, a1 = a + da, a - da + 2 * math.pi

    def at(r: float, ang: float) -> str:
        return f"{fmt(cx + r * math.cos(ang))},{fmt(cy + r * math.sin(ang))}"

    ring = (
        f"M{at(ro, a0)}A{fmt(ro)},{fmt(ro)} 0 1 1 {at(ro, a1)}"
        f"L{at(ri, a1)}A{fmt(ri)},{fmt(ri)} 0 1 0 {at(ri, a0)}Z"
    )
    return (
        f'<path d="{ring}" fill="{ink}"/>'
        f'<path d="{"".join(poly(b) for b in beams)}" fill="{accent}"/>'
        + circle(px, py, rp, accent)
        + circle(cx, cy, t * COMPANION_R, ink)
    )


def stroke(a: tuple[float, float], b: tuple[float, float]) -> list[tuple[float, float]]:
    """Straight stroke from a to b, its far end cut flat on the line y = b[1]."""
    t = STROKE
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    nx, ny = -dy / length * t / 2, dx / length * t / 2
    near = [(a[0] - nx, a[1] - ny), (a[0] + nx, a[1] + ny)]
    far = []
    for x0, y0 in near[::-1]:
        k = (b[1] - y0) / dy
        far.append((x0 + dx * k, b[1]))
    return near + far


def word_body(ink: str, accent: str) -> tuple[str, float]:
    """LOKI and its width. Every letter sits on the baseline and the cap line."""
    t = STROKE
    el = poly(
        [(0, 0), (t, 0), (t, 100 - t), (L_WIDTH, 100 - t), (L_WIDTH, 100), (0, 100)]
    )
    ox = L_WIDTH + GAP_LO
    kx = ox + 2 * O_RADIUS + GAP_OK
    sx = kx + t / 2 + 0.5
    arm_end = (kx + 57, 0)
    k = 0.27  # the leg springs from the arm this far along it
    leg_start = (sx + (arm_end[0] - sx) * k, 64 * (1 - k))
    ix = kx + 66 + GAP_KI
    # The K strokes overlap, so each is its own path: one path with opposite
    # windings would punch holes where they cross.
    shapes = [
        el,
        poly([(kx, 0), (kx + t, 0), (kx + t, 100), (kx, 100)]),
        poly(stroke((sx, 64), arm_end)),
        poly(stroke(leg_start, (kx + 63, 100))),
        poly([(ix, 0), (ix + t, 0), (ix + t, 100), (ix, 100)]),
    ]
    paths = "".join(f'<path d="{d}"/>' for d in shapes)
    body = f'<g fill="{ink}">{paths}</g>' + o_body(ox + O_RADIUS, ink, accent)
    return body, ix + t


# ---------------------------------------------------------------- compositions
def svg(viewbox: Sequence[float], body: str, label: str) -> str:
    vb = " ".join(fmt(v) for v in viewbox)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" role="img" '
        f'aria-label="{label}"><title>{label}</title>{body}</svg>\n'
    )


def logo_svg(ink: str, accent: str, pad: float = 4.0) -> str:
    body, width = word_body(ink, accent)
    # Room for the O's overshoot, and for a beam tip that crosses the cap line.
    _, py, rp, beams = pulsar(0, STROKE)
    top = min([-2.0, py - rp] + [y for b in beams for _, y in b])
    return svg((-pad, top - pad, width + 2 * pad, 102 - top + 2 * pad), body, "loki")


def square(t: float | None = None) -> tuple[float, float, float]:
    """Centre (x, y) and side of the square holding the O at (0, 50) and its beams.

    Icons centre this box, not the ring, so the beams are not clipped.
    """
    t = t or STROKE
    px, py, rp, beams = pulsar(0, t)
    xs = [-O_RADIUS, O_RADIUS, px + rp] + [x for b in beams for x, _ in b]
    ys = [50 - O_RADIUS, 50 + O_RADIUS, py - rp] + [y for b in beams for _, y in b]
    side = max(max(xs) - min(xs), max(ys) - min(ys))
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, side


def mark_svg(ink: str, accent: str, pad: float = 4.0) -> str:
    cx, cy, side = square()
    side += 2 * pad
    vb = (cx - side / 2, cy - side / 2, side, side)
    return svg(vb, o_body(0, ink, accent), "loki")


def placed(t: float | None, size: float, fill: float, body: str) -> str:
    """Scale and centre ``body``, an O drawn at stroke ``t``, in a ``size`` box.

    The O's square (see ``square``) fills ``fill`` of the box.
    """
    cx, cy, side = square(t)
    k = size * fill / side
    tx, ty = size / 2 - cx * k, size / 2 - cy * k
    return f'<g transform="translate({fmt(tx)} {fmt(ty)}) scale({k:.5f})">{body}</g>'


def tile_svg(
    ink: str,
    accent: str,
    bg: str,
    inner: float = 0.74,
    radius: float = 0.22,
    t: float | None = None,
) -> str:
    s = 512
    body = (
        f'<rect width="{s}" height="{s}" rx="{fmt(s * radius)}" fill="{bg}"/>'
        + placed(t, s, inner, o_body(0, ink, accent, t))
    )
    return svg((0, 0, s, s), body, "loki")


def favicon_svg(t: float) -> str:
    """Bare O that follows the browser's light/dark preference.

    It is drawn heavier (``t``) so the orbit and the pulsar survive 16 px.
    """
    mark = (
        o_body(0, "INK", "ACC", t)
        .replace('fill="INK"', 'class="i"')
        .replace('fill="ACC"', 'class="a"')
    )
    body = (
        f"<style>.i{{fill:{INK}}}.a{{fill:{ACCENT}}}"
        "@media (prefers-color-scheme:dark)"
        f"{{.i{{fill:{INK_D}}}.a{{fill:{ACCENT_D}}}}}</style>"
        + placed(t, 100, 1.0, mark)
    )
    return svg((0, 0, 100, 100), body, "loki")


def social_svg(logo_dark: str) -> str:
    """1280x640 GitHub social preview: the wordmark centred on dark."""
    width = 760
    height = width / aspect(logo_dark)
    nested = logo_dark.replace(
        "<svg ",
        f'<svg x="{fmt((1280 - width) / 2)}" y="{fmt((640 - height) / 2)}" '
        f'width="{width}" height="{fmt(height)}" ',
        1,
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="640" '
        'viewBox="0 0 1280 640">'
        f'<rect width="1280" height="640" fill="{TILE_D}"/>{nested}</svg>\n'
    )


# ---------------------------------------------------------------- output
def write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content)
    else:
        path.write_bytes(content)
    print(f"  {path.relative_to(HERE.parent.parent)}")


def png(svg_text: str, width: int, height: int) -> bytes:
    return bytes(resvg_py.svg_to_bytes(svg_string=svg_text, width=width, height=height))


def aspect(svg_text: str) -> float:
    vb = [float(v) for v in svg_text.split('viewBox="', 1)[1].split('"', 1)[0].split()]
    return vb[2] / vb[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--exports",
        action="store_true",
        help="also write PNG/ICO/mono/social exports to build/",
    )
    args = parser.parse_args()

    # Files referenced by README.md and the docs.
    print("docs/_static/logo:")
    write(STATIC / "loki-logo.svg", logo_svg(INK, ACCENT))
    write(STATIC / "loki-logo-dark.svg", logo_svg(INK_D, ACCENT_D))
    write(STATIC / "loki-mark.svg", mark_svg(INK, ACCENT))
    write(STATIC / "loki-mark-dark.svg", mark_svg(INK_D, ACCENT_D))
    write(STATIC / "favicon.svg", favicon_svg(SMALL_STROKE))

    if not args.exports:
        return

    print("docs/logo-src/build:")
    tones = {"": (ACCENT, ACCENT_D), "-mono": (INK, INK_D)}
    for tone, (acc_l, acc_d) in tones.items():
        for dark, (ink, acc, tile) in {
            "": (INK, acc_l, TILE_L),
            "-dark": (INK_D, acc_d, TILE_D),
        }.items():
            assets = {
                f"loki-mark{tone}{dark}": (mark_svg(ink, acc), "h", 1024),
                f"loki-logo{tone}{dark}": (logo_svg(ink, acc), "w", 2400),
                f"loki-icon{tone}{dark}": (tile_svg(ink, acc, tile), "w", 1024),
            }
            for name, (text, axis, size) in assets.items():
                a = aspect(text)
                w, h = (
                    (size, round(size / a)) if axis == "w" else (round(size * a), size)
                )
                write(BUILD / f"{name}.svg", text)
                write(BUILD / f"{name}.png", png(text, w, h))

    # Favicon/touch icon use a tile with a larger, heavier O so it survives 16 px.
    fav = tile_svg(INK, ACCENT, TILE_L, inner=0.86, radius=0.2, t=SMALL_STROKE)
    write(BUILD / "apple-touch-icon.png", png(fav, 180, 180))
    frames = [
        Image.open(io.BytesIO(png(fav, s, s))).convert("RGBA") for s in (64, 48, 32, 16)
    ]
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="ICO",
        sizes=[(s, s) for s in (16, 32, 48, 64)],
        append_images=frames[1:],
    )
    write(BUILD / "favicon.ico", buf.getvalue())

    social = social_svg(logo_svg(INK_D, ACCENT_D))
    write(BUILD / "social-preview.png", png(social, 1280, 640))


if __name__ == "__main__":
    main()
