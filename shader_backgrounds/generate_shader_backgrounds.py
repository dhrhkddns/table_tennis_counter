#!/usr/bin/env python3
"""Generate seamless near-black shader background GIF loops (1920x1080).

All motion is phase-locked to a full 2π cycle with *integer* harmonics so
frame 0 and frame N meet without a visual jump.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

OUT_DIR = Path(__file__).resolve().parent
W, H = 1920, 1080
LW, LH = 192, 108  # compact render → upscale (keeps GIF size sane)
FRAMES = 20
DURATION_MS = 80  # 20 * 80ms = 1.6s loop
PALETTE = 20


def phase(frame: int) -> float:
    """Normalized angle in [0, 2π) for seamless loops."""
    return (2.0 * math.pi * frame) / FRAMES


def black_rgb(v: np.ndarray, tint: float = 0.0) -> np.ndarray:
    """Map luminance [0,1] → near-black RGB (keep highlights dark).

    tint > 0 = slight cool bias, tint < 0 = slight warm bias.
    Output stays in a black / charcoal family.
    """
    # Crush into a dark range: mostly 0–0.40 luminance (black / charcoal)
    v = np.clip(v, 0.0, 1.0)
    lum = 0.015 + 0.38 * (v ** 1.2)
    r = np.clip(lum * (1.0 - 0.04 * tint), 0.0, 1.0)
    g = np.clip(lum * (1.0 - 0.01 * abs(tint)), 0.0, 1.0)
    b = np.clip(lum * (1.0 + 0.05 * tint), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def hash2(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    n = (ix * 127.1 + iy * 311.7).astype(np.float64)
    return np.mod(np.sin(n) * 43758.5453, 1.0)


def value_noise(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    n00 = hash2(x0, y0)
    n10 = hash2(x0 + 1, y0)
    n01 = hash2(x0, y0 + 1)
    n11 = hash2(x0 + 1, y0 + 1)
    nx0 = n00 * (1 - ux) + n10 * ux
    nx1 = n01 * (1 - ux) + n11 * ux
    return nx0 * (1 - uy) + nx1 * uy


def fbm(x: np.ndarray, y: np.ndarray, octaves: int = 4) -> np.ndarray:
    amp = 0.5
    freq = 1.0
    total = np.zeros_like(x, dtype=np.float64)
    norm = 0.0
    for _ in range(octaves):
        total += amp * value_noise(x * freq, y * freq)
        norm += amp
        amp *= 0.5
        freq *= 2.0
    return total / norm


def coords() -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, 1.0, LW, endpoint=False, dtype=np.float64)
    ys = np.linspace(0.0, 1.0, LH, endpoint=False, dtype=np.float64)
    return np.meshgrid(xs, ys)


def vignette(x: np.ndarray, y: np.ndarray, strength: float = 0.55) -> np.ndarray:
    dx = x - 0.5
    dy = y - 0.5
    d = np.sqrt(dx * dx + dy * dy) / 0.72
    return 1.0 - strength * np.clip(d * d, 0.0, 1.0)


# --- New pattern set (integer harmonics only → seamless) ------------------


def shader_ink_ripple(frame: int) -> np.ndarray:
    """Concentric ink ripples expanding from slightly offset centers."""
    x, y = coords()
    th = phase(frame)
    field = np.zeros((LH, LW), dtype=np.float64)
    centers = [
        (0.42 + 0.08 * math.cos(th), 0.48 + 0.06 * math.sin(th), 14.0, 1),
        (0.62 + 0.07 * math.sin(th), 0.55 + 0.05 * math.cos(th), 10.0, -1),
        (0.50 + 0.05 * math.cos(2 * th), 0.40 + 0.04 * math.sin(2 * th), 18.0, 2),
    ]
    for cx, cy, freq, harm in centers:
        r = np.sqrt((x - cx) ** 2 + ((y - cy) * 1.15) ** 2)
        field += 0.5 + 0.5 * np.sin(r * freq - harm * th)
    field /= len(centers)
    field *= vignette(x, y, 0.5)
    return black_rgb(field, tint=0.3)


def shader_hex_crawl(frame: int) -> np.ndarray:
    """Soft hexagonal lattice that slowly breathes / crawls."""
    x, y = coords()
    th = phase(frame)
    # Skewed hex-ish lattice via two sine sets
    ox = 0.04 * math.sin(th)
    oy = 0.04 * math.cos(th)
    a = np.sin((x + ox) * math.pi * 8)
    b = np.sin(((x * 0.5 + y * 0.866) + oy) * math.pi * 8)
    c = np.sin(((x * 0.5 - y * 0.866) - ox) * math.pi * 8)
    cells = (np.abs(a) * np.abs(b) * np.abs(c)) ** 0.35
    edges = 1.0 - np.minimum(np.minimum(np.abs(a), np.abs(b)), np.abs(c))
    edges = np.clip(edges ** 4, 0, 1)
    n = fbm(x * 1.5 + ox, y * 1.5 + oy, 3)
    field = 0.55 * cells + 0.25 * edges + 0.20 * n
    field *= vignette(x, y, 0.48)
    return black_rgb(field, tint=0.15)


def shader_vortex_spiral(frame: int) -> np.ndarray:
    """Dark spiral arms rotating around center."""
    x, y = coords()
    th = phase(frame)
    dx = x - 0.5
    dy = (y - 0.5) * (LH / LW)
    r = np.sqrt(dx * dx + dy * dy) + 1e-6
    ang = np.arctan2(dy, dx)
    arms = 0.5 + 0.5 * np.sin(ang * 3.0 - r * 22.0 + th)
    arms2 = 0.5 + 0.5 * np.sin(ang * 5.0 + r * 14.0 - 2 * th)
    fall = np.exp(-r * 2.2)
    n = fbm(x * 2.0 + 0.12 * math.cos(th), y * 2.0 + 0.12 * math.sin(th), 3)
    field = (0.5 * arms + 0.3 * arms2 + 0.2 * n) * (0.35 + 0.65 * fall)
    return black_rgb(field, tint=0.4)


def shader_rain_streaks(frame: int) -> np.ndarray:
    """Diagonal rain / scratch streaks drifting seamlessly."""
    x, y = coords()
    th = phase(frame)
    # Toroidal vertical drift via sin phase (no hard wrap jump)
    drift = 0.35 * math.sin(th)
    streak = x * 0.35 + y + drift
    # Multiple streak layers at different frequencies
    s1 = 0.5 + 0.5 * np.sin(streak * math.pi * 28 + th)
    s2 = 0.5 + 0.5 * np.sin((x * 0.2 + y + 0.35 * math.sin(th + math.pi)) * math.pi * 18 - th)
    # Thin the streaks
    s1 = np.clip(s1 ** 4, 0, 1)
    s2 = np.clip(s2 ** 3.5, 0, 1)
    n = fbm(x * 1.8, y * 2.2 + 0.15 * math.cos(th), 3)
    field = 0.50 * s1 + 0.35 * s2 + 0.30 * n
    field *= vignette(x, y, 0.35)
    return black_rgb(np.clip(field, 0, 1), tint=0.2)


def shader_ember_embers(frame: int) -> np.ndarray:
    """Sparse rising ember-like dots on deep black (dark charcoal sparks)."""
    x, y = coords()
    th = phase(frame)
    field = np.zeros((LH, LW), dtype=np.float64)
    # Deterministic ember positions orbiting / rising on closed paths
    rng = np.random.RandomState(42)
    for i in range(28):
        base_x = float(rng.random())
        base_y = float(rng.random())
        speed = 1 if i % 2 == 0 else 2
        amp = 0.04 + 0.03 * (i % 3)
        cx = (base_x + amp * math.sin(th * speed + i * 0.7)) % 1.0
        # Rise with seamless sin so it doesn't jump at wrap
        cy = (base_y + 0.12 * math.sin(th * speed + i * 1.3)) % 1.0
        # Toroidal distance
        dx = np.abs(x - cx)
        dx = np.minimum(dx, 1.0 - dx)
        dy = np.abs(y - cy)
        dy = np.minimum(dy, 1.0 - dy)
        d2 = dx * dx + dy * dy
        sigma = 0.012 + 0.008 * (i % 4) / 3
        pulse = 0.55 + 0.45 * math.sin(th * speed + i)
        field += pulse * np.exp(-d2 / (2 * sigma * sigma))
    field = np.clip(field / 1.6, 0, 1)
    # Faint under-fog
    fog = fbm(x * 1.4 + 0.1 * math.cos(th), y * 1.4 + 0.1 * math.sin(th), 3)
    field = 0.85 * field + 0.15 * fog * 0.55
    field *= vignette(x, y, 0.45)
    return black_rgb(field, tint=-0.25)  # slight warm charcoal


def shader_cross_hatch(frame: int) -> np.ndarray:
    """Animated cross-hatch / etch lines on black."""
    x, y = coords()
    th = phase(frame)
    ox = 0.05 * math.sin(th)
    oy = 0.05 * math.cos(th)
    h1 = np.sin((x + y + ox) * math.pi * 16)
    h2 = np.sin((x - y + oy) * math.pi * 16)
    # Thin line look
    l1 = np.clip(1.0 - np.abs(h1) * 8.0, 0, 1) ** 2
    l2 = np.clip(1.0 - np.abs(h2) * 8.0, 0, 1) ** 2
    # Phase-modulated visibility
    m1 = 0.5 + 0.5 * math.sin(th)
    m2 = 0.5 + 0.5 * math.sin(th + math.pi / 2)
    n = fbm(x * 1.6 + ox, y * 1.6 + oy, 3)
    field = 0.55 * (m1 * l1 + m2 * l2) + 0.40 * n
    field *= vignette(x, y, 0.45)
    return black_rgb(np.clip(field, 0, 1), tint=0.1)


def shader_horizon_wave(frame: int) -> np.ndarray:
    """Stacked horizontal wave layers like dark dunes / signal waves."""
    x, y = coords()
    th = phase(frame)
    field = np.zeros((LH, LW), dtype=np.float64)
    for i, (amp, freq, harm, y0) in enumerate(
        [
            (0.06, 3.0, 1, 0.30),
            (0.05, 5.0, 1, 0.45),
            (0.04, 4.0, 2, 0.60),
            (0.035, 6.0, 2, 0.72),
        ]
    ):
        wave_y = y0 + amp * np.sin(x * math.pi * freq + harm * th + i)
        dist = np.abs(y - wave_y)
        field += np.exp(-((dist / 0.035) ** 2)) * (0.7 - i * 0.1)
        # Fill below wave faintly
        below = np.clip((wave_y - y) * 4.0, 0, 1)
        field += 0.08 * below
    n = fbm(x * 1.5 + 0.1 * math.cos(th), y * 2.0 + 0.1 * math.sin(th), 3)
    field = np.clip(0.7 * field + 0.3 * n * 0.5, 0, 1)
    field *= vignette(x, y, 0.45)
    return black_rgb(field, tint=0.35)


def shader_caustic_cells(frame: int) -> np.ndarray:
    """Dark caustic / cellular membrane look."""
    x, y = coords()
    th = phase(frame)
    tx = 0.2 * math.cos(th)
    ty = 0.2 * math.sin(th)
    # Domain warp then cellular-ish abs-noise
    wx = fbm(x * 1.8 + tx, y * 1.8 + ty, 3)
    wy = fbm(x * 1.8 - ty + 2.3, y * 1.8 + tx + 1.1, 3)
    qx = x * 3.5 + (wx - 0.5) * 1.8
    qy = y * 3.0 + (wy - 0.5) * 1.8
    n1 = value_noise(qx, qy)
    n2 = value_noise(qx + 3.7, qy + 1.9)
    cells = 1.0 - np.abs(n1 - n2) * 2.0
    cells = np.clip(cells, 0, 1) ** 1.6
    field = cells * vignette(x, y, 0.5)
    return black_rgb(field, tint=0.25)


def shader_radar_sweep(frame: int) -> np.ndarray:
    """Dark radar / clock-hand sweep with faint rings."""
    x, y = coords()
    th = phase(frame)
    dx = x - 0.5
    dy = (y - 0.5) * (LH / LW)
    r = np.sqrt(dx * dx + dy * dy)
    ang = np.arctan2(dy, dx)  # [-π, π]
    # Soft wedge following phase (seamless because ang wraps and th is 2π-periodic)
    # Distance on circle between ang and th
    delta = (ang - th + math.pi) % (2 * math.pi) - math.pi
    wedge = np.exp(-((delta / 0.35) ** 2))
    # Faint trail behind
    delta2 = (ang - th + math.pi + 0.5) % (2 * math.pi) - math.pi
    trail = 0.35 * np.exp(-((delta2 / 0.9) ** 2))
    rings = 0.5 + 0.5 * np.sin(r * 28.0 - 2 * th)
    rings = np.clip(rings ** 4, 0, 1) * 0.45
    n = fbm(x * 1.5, y * 1.5, 2)
    field = (0.55 * wedge + 0.25 * trail + rings + 0.15 * n) * np.clip(1.0 - r * 1.3, 0.15, 1)
    return black_rgb(field, tint=0.45)


def shader_pixel_static(frame: int) -> np.ndarray:
    """Coarse blocky static that slowly morphs (CRT / void noise)."""
    x, y = coords()
    th = phase(frame)
    # Block-quantize coords
    bx = np.floor(x * 24) / 24
    by = np.floor(y * 14) / 14
    # Loopable: mix two hashes with sin/cos weights
    w = 0.5 + 0.5 * math.sin(th)
    n_a = fbm(bx * 3.0 + 0.3 * math.cos(th), by * 3.0 + 0.3 * math.sin(th), 2)
    n_b = fbm(bx * 3.0 + 1.7 - 0.3 * math.sin(th), by * 3.0 + 2.1 + 0.3 * math.cos(th), 2)
    field = (1 - w) * n_a + w * n_b
    # Occasional darker bars
    bar = 0.5 + 0.5 * np.sin(y * math.pi * 6 + th)
    field *= 0.7 + 0.3 * bar
    field *= vignette(x, y, 0.6)
    return black_rgb(field, tint=0.05)


SHADERS = [
    ("ink_ripple", shader_ink_ripple),
    ("hex_crawl", shader_hex_crawl),
    ("vortex_spiral", shader_vortex_spiral),
    ("rain_streaks", shader_rain_streaks),
    ("ember_sparks", shader_ember_embers),
    ("cross_hatch", shader_cross_hatch),
    ("horizon_wave", shader_horizon_wave),
    ("caustic_cells", shader_caustic_cells),
    ("radar_sweep", shader_radar_sweep),
    ("pixel_static", shader_pixel_static),
]


def to_gif_frame(rgb: np.ndarray, palette_img: Image.Image | None = None) -> Image.Image:
    img = Image.fromarray(rgb, mode="RGB")
    img = img.resize((LW // 2, LH // 2), Image.BILINEAR).resize((LW, LH), Image.BILINEAR)
    up = img.resize((W, H), Image.BILINEAR)
    if palette_img is not None:
        return up.quantize(palette=palette_img, dither=Image.Dither.NONE)
    return up.quantize(colors=PALETTE, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)


def save_gif(path: Path, render_fn) -> None:
    mid = to_gif_frame(render_fn(FRAMES // 2))
    frames = [to_gif_frame(render_fn(i), palette_img=mid) for i in range(FRAMES)]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=DURATION_MS,
        loop=0,
        optimize=True,
        disposal=1,
    )
    try:
        tmp = path.with_suffix(".opt.gif")
        subprocess.run(
            [
                "gifsicle",
                "-O3",
                "--lossy=45",
                "--colors",
                str(PALETTE),
                "-o",
                str(tmp),
                str(path),
            ],
            check=True,
            capture_output=True,
        )
        tmp.replace(path)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass


def verify_loop_continuity(render_fn, name: str) -> float:
    a = render_fn(0).astype(np.float32)
    b = render_fn(FRAMES - 1).astype(np.float32)
    c = render_fn(1).astype(np.float32)
    adj = float(np.mean(np.abs(a - c)))
    seam = float(np.mean(np.abs(a - b)))
    ratio = seam / max(adj, 1e-6)
    status = "ok" if ratio < 1.35 else "WARN"
    print(f"  [{status}] {name}: seam={seam:.2f} adj={adj:.2f} ratio={ratio:.2f}")
    return ratio


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("shader_bg_*.gif"):
        old.unlink()

    bad = []
    for idx, (name, fn) in enumerate(SHADERS, start=1):
        ratio = verify_loop_continuity(fn, name)
        if ratio >= 1.35:
            bad.append(name)
        path = OUT_DIR / f"shader_bg_{idx:02d}_{name}.gif"
        save_gif(path, fn)
        kb = path.stat().st_size // 1024
        print(f"{path.name}  {kb}KB")

    total = sum(p.stat().st_size for p in OUT_DIR.glob("shader_bg_*.gif"))
    print(f"done: {len(SHADERS)} gifs, total {total // 1024}KB")
    if bad:
        raise SystemExit(f"loop continuity failed: {bad}")


if __name__ == "__main__":
    main()
