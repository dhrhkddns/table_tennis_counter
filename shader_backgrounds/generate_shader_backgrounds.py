#!/usr/bin/env python3
"""Generate seamless gray-tone shader background GIF loops (1920x1080).

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
PALETTE = 24


def phase(frame: int) -> float:
    """Normalized angle in [0, 2π) for seamless loops."""
    return (2.0 * math.pi * frame) / FRAMES


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def gray_rgb(v: np.ndarray, cool: float = 0.0) -> np.ndarray:
    """Map luminance [0,1] → soft gray RGB with optional cool bias."""
    v = np.clip(v, 0.0, 1.0)
    r = v * (1.0 - 0.02 * cool)
    g = v * (1.0 - 0.01 * cool)
    b = np.clip(v * (1.0 + 0.04 * cool), 0.0, 1.0)
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


def vignette(x: np.ndarray, y: np.ndarray, strength: float = 0.45) -> np.ndarray:
    dx = x - 0.5
    dy = y - 0.5
    d = np.sqrt(dx * dx + dy * dy) / 0.75
    return 1.0 - strength * np.clip(d * d, 0.0, 1.0)


# --- Shader styles (integer harmonics only → seamless) --------------------


def shader_soft_flow(frame: int) -> np.ndarray:
    x, y = coords()
    th = phase(frame)
    tx = 0.35 * math.cos(th)
    ty = 0.35 * math.sin(th)
    n1 = fbm(x * 2.4 + tx, y * 2.0 + ty, octaves=4)
    n2 = fbm(x * 3.2 - ty * 0.8, y * 2.6 + tx * 0.8, octaves=3)
    field = 0.55 * n1 + 0.45 * n2
    lum = 0.28 + 0.40 * field
    lum *= vignette(x, y, 0.35)
    breath = 0.04 * math.sin(th)  # harmonic 1
    return gray_rgb(lum + breath, cool=0.6)


def shader_drift_bands(frame: int) -> np.ndarray:
    x, y = coords()
    th = phase(frame)
    a = np.sin((x * 2.4 + y * 1.0) * math.pi * 2 + th) * 0.5 + 0.5
    b = np.sin((x * -1.2 + y * 2.0) * math.pi * 2 - th) * 0.5 + 0.5
    n = fbm(x * 1.5 + 0.2 * math.cos(th), y * 1.5 + 0.2 * math.sin(th), 3)
    field = 0.45 * a + 0.35 * b + 0.20 * n
    lum = 0.26 + 0.44 * field
    lum *= vignette(x, y, 0.4)
    return gray_rgb(lum, cool=0.4)


def shader_radial_pulse(frame: int) -> np.ndarray:
    x, y = coords()
    th = phase(frame)
    dx = x - 0.5
    dy = (y - 0.5) * (LH / LW)
    r = np.sqrt(dx * dx + dy * dy)
    rings = 0.5 + 0.5 * np.sin(r * 18.0 - th)       # harmonic 1
    soft = 0.5 + 0.5 * np.sin(r * 7.0 + 2.0 * th)   # harmonic 2 (was 0.5!)
    n = fbm(x * 2.0 + 0.15 * math.cos(th), y * 2.0 + 0.15 * math.sin(th), 3)
    field = 0.5 * rings + 0.3 * soft + 0.2 * n
    lum = 0.24 + 0.48 * field * (1.0 - 0.45 * np.clip(r * 1.6, 0, 1))
    return gray_rgb(lum, cool=0.7)


def shader_scan_wrap(frame: int) -> np.ndarray:
    x, y = coords()
    th = phase(frame)
    band_y = 0.5 + 0.42 * math.sin(th)
    dist = np.abs(y - band_y)
    dist = np.minimum(dist, 1.0 - dist)
    band = np.exp(-((dist / 0.08) ** 2))
    n = fbm(x * 2.5 + 0.25 * math.cos(th), y * 2.5 + 0.25 * math.sin(th), 3)
    base = 0.27 + 0.26 * n
    lum = base + 0.22 * band
    band2_y = 0.5 + 0.42 * math.sin(th + math.pi)
    dist2 = np.minimum(np.abs(y - band2_y), 1.0 - np.abs(y - band2_y))
    lum += 0.08 * np.exp(-((dist2 / 0.12) ** 2))
    lum *= vignette(x, y, 0.3)
    return gray_rgb(lum, cool=0.5)


def shader_smoke_warp(frame: int) -> np.ndarray:
    x, y = coords()
    th = phase(frame)
    cx = 0.22 * math.cos(th)
    cy = 0.22 * math.sin(th)
    wx = fbm(x * 1.4 + cx, y * 1.4 + cy, 3)
    wy = fbm(x * 1.4 - cy + 3.1, y * 1.4 + cx + 1.7, 3)
    qx = x * 2.6 + (wx - 0.5) * 1.4 + cx
    qy = y * 2.2 + (wy - 0.5) * 1.4 + cy
    field = fbm(qx, qy, 4)
    lum = 0.22 + 0.50 * (field ** 1.15)
    lum *= vignette(x, y, 0.38)
    return gray_rgb(lum, cool=0.8)


def shader_grid_shimmer(frame: int) -> np.ndarray:
    x, y = coords()
    th = phase(frame)
    ox = 0.05 * math.sin(th)
    oy = 0.05 * math.cos(th)
    # Coarser grid compresses far better in GIF
    gx = np.sin((x + ox) * math.pi * 10) ** 2
    gy = np.sin((y + oy) * math.pi * 6) ** 2
    lines = np.maximum((1.0 - gx) ** 6, (1.0 - gy) ** 6)
    n = fbm(x * 1.4 + ox * 2, y * 1.4 + oy * 2, 3)
    field = 0.65 * n + 0.35 * lines
    lum = 0.25 + 0.38 * field
    lum *= vignette(x, y, 0.42)
    return gray_rgb(lum, cool=0.55)


def shader_aurora_gray(frame: int) -> np.ndarray:
    x, y = coords()
    th = phase(frame)
    # Integer harmonics only
    r1 = np.sin(x * math.pi * 3 + th + 0.6 * np.sin(y * 4 + th))
    r2 = np.sin(x * math.pi * 5 - th + 0.4 * np.sin(y * 3 - th))
    ribbons = 0.5 * (0.5 + 0.5 * r1) + 0.5 * (0.5 + 0.5 * r2)
    height = 0.55 + 0.45 * np.sin(y * math.pi)
    n = fbm(x * 2.0 + 0.2 * math.cos(th), y * 1.5 + 0.2 * math.sin(th), 3)
    field = ribbons * height * 0.75 + n * 0.25
    lum = 0.24 + 0.48 * field
    lum *= vignette(x, y, 0.32)
    return gray_rgb(lum, cool=0.9)


def shader_soft_wash(frame: int) -> np.ndarray:
    """Large soft gray wash — replaces noisy grain (GIF-hostile)."""
    x, y = coords()
    th = phase(frame)
    n1 = fbm(x * 1.3 + 0.25 * math.cos(th), y * 1.1 + 0.25 * math.sin(th), 4)
    n2 = fbm(x * 2.0 - 0.2 * math.sin(th), y * 1.8 + 0.2 * math.cos(th), 3)
    field = 0.6 * n1 + 0.4 * n2
    breath = 0.05 * math.sin(th)
    lum = 0.30 + 0.36 * field + breath
    lum *= vignette(x, y, 0.28)
    return gray_rgb(lum, cool=0.35)


def shader_orbit_blobs(frame: int) -> np.ndarray:
    x, y = coords()
    th = phase(frame)
    field = np.zeros((LH, LW), dtype=np.float64)
    # Integer speed multipliers only (1 or 2) so orbits close after one loop
    blobs = [
        (0.28, 1, 0.0, 0.18),
        (0.20, 1, 2.1, 0.14),
        (0.34, 2, 4.0, 0.20),
        (0.16, 2, 1.2, 0.12),
    ]
    for radius, speed, offset, sigma in blobs:
        cx = 0.5 + radius * math.cos(th * speed + offset)
        cy = 0.5 + radius * math.sin(th * speed + offset) * 0.75
        d2 = (x - cx) ** 2 + ((y - cy) * 1.1) ** 2
        field += np.exp(-d2 / (2 * sigma * sigma))
    n = fbm(x * 2.0 + 0.15 * math.cos(th), y * 2.0 + 0.15 * math.sin(th), 3)
    field = 0.7 * np.clip(field / 1.8, 0, 1) + 0.3 * n
    lum = 0.23 + 0.48 * field
    lum *= vignette(x, y, 0.36)
    return gray_rgb(lum, cool=0.5)


def shader_metal_sheen(frame: int) -> np.ndarray:
    x, y = coords()
    th = phase(frame)
    diag = x * 0.85 + y * 0.55
    sheen = 0.5 + 0.5 * np.sin(diag * math.pi * 4 + th)
    sheen2 = 0.5 + 0.5 * np.sin(diag * math.pi * 2 - th)
    brush = 0.5 + 0.5 * np.sin(y * math.pi * 24 + math.sin(th))
    n = fbm(x * 1.3, y * 2.0 + 0.1 * math.sin(th), 3)
    field = 0.4 * sheen + 0.25 * sheen2 + 0.15 * brush + 0.20 * n
    lum = 0.26 + 0.42 * field
    lum *= vignette(x, y, 0.4)
    return gray_rgb(lum, cool=0.25)


SHADERS = [
    ("soft_flow", shader_soft_flow),
    ("drift_bands", shader_drift_bands),
    ("radial_pulse", shader_radial_pulse),
    ("scan_wrap", shader_scan_wrap),
    ("smoke_warp", shader_smoke_warp),
    ("grid_shimmer", shader_grid_shimmer),
    ("aurora_gray", shader_aurora_gray),
    ("soft_wash", shader_soft_wash),
    ("orbit_blobs", shader_orbit_blobs),
    ("metal_sheen", shader_metal_sheen),
]


def to_gif_frame(rgb: np.ndarray, palette_img: Image.Image | None = None) -> Image.Image:
    img = Image.fromarray(rgb, mode="RGB")
    # Mild blur before upscale → fewer unique colors → smaller GIF
    img = img.resize((LW // 2, LH // 2), Image.BILINEAR).resize((LW, LH), Image.BILINEAR)
    up = img.resize((W, H), Image.BILINEAR)
    if palette_img is not None:
        return up.quantize(palette=palette_img, dither=Image.Dither.NONE)
    return up.quantize(colors=PALETTE, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)


def save_gif(path: Path, render_fn) -> None:
    # Build a shared palette from a mid-loop frame for stabler deltas
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
    # Lossy gifsicle pass — big win on soft gradient shaders
    try:
        tmp = path.with_suffix(".opt.gif")
        subprocess.run(
            [
                "gifsicle",
                "-O3",
                "--lossy=40",
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
