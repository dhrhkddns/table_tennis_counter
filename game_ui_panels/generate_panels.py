#!/usr/bin/env python3
"""Generate 1000 varied game-style rectangular UI panel GIFs (transparent BG)."""

from __future__ import annotations

import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).resolve().parent
COUNT = 1000
FRAMES_CHOICES = (4, 6, 8)
DURATION_CHOICES = (80, 100, 120, 140)

STYLES = [
    "flat",
    "inset",
    "raised",
    "double",
    "neon",
    "bevel",
    "window",
    "bracket",
    "chunky",
    "scanbox",
    "rpg",
    "arcade",
    "holo",
    "metal",
    "pixel_thick",
]

ANIMS = ["pulse", "chase", "shimmer", "flicker", "scan", "static", "blink_corner", "breathe"]

# Game-ish palette families (base mid tones). Random variation applied per panel.
PALETTE_SEEDS = [
    ("navy", (30, 55, 110)),
    ("steel", (70, 85, 105)),
    ("forest", (28, 90, 55)),
    ("ember", (150, 55, 35)),
    ("gold", (170, 130, 40)),
    ("ice", (55, 130, 170)),
    ("violet", (95, 55, 140)),
    ("rose", (150, 60, 95)),
    ("olive", (95, 110, 45)),
    ("coal", (40, 42, 48)),
    ("sand", (140, 115, 75)),
    ("teal", (30, 120, 115)),
    ("blood", (120, 30, 40)),
    ("mint", (55, 140, 110)),
    ("sky", (70, 120, 190)),
]


def clamp(v: int, lo: int = 0, hi: int = 255) -> int:
    return max(lo, min(hi, v))


def jitter(c: tuple[int, int, int], amt: int, rng: random.Random) -> tuple[int, int, int]:
    return tuple(clamp(ch + rng.randint(-amt, amt)) for ch in c)


def shade(c: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(clamp(int(ch * factor)) for ch in c)


def lighten(c: tuple[int, int, int], amt: int) -> tuple[int, int, int]:
    return tuple(clamp(ch + amt) for ch in c)


def rgba(c: tuple[int, int, int], a: int = 255) -> tuple[int, int, int, int]:
    return (*c, a)


def pick_size(rng: random.Random) -> tuple[int, int, str]:
    """Non-uniform sizes: bars, cards, sidebars, big panels."""
    kind = rng.choices(
        ["card", "wide_bar", "tall_bar", "sidebar", "modal", "chip", "banner", "square"],
        weights=[18, 14, 10, 14, 12, 10, 12, 10],
        k=1,
    )[0]
    if kind == "card":
        w = rng.randrange(96, 321, 8)
        h = rng.randrange(72, 241, 8)
    elif kind == "wide_bar":
        w = rng.randrange(180, 641, 8)
        h = rng.randrange(28, 81, 4)
    elif kind == "tall_bar":
        w = rng.randrange(28, 81, 4)
        h = rng.randrange(120, 401, 8)
    elif kind == "sidebar":
        w = rng.randrange(80, 201, 8)
        h = rng.randrange(180, 481, 8)
    elif kind == "modal":
        w = rng.randrange(220, 521, 8)
        h = rng.randrange(140, 361, 8)
    elif kind == "chip":
        w = rng.randrange(48, 161, 4)
        h = rng.randrange(24, 65, 4)
    elif kind == "banner":
        w = rng.randrange(240, 641, 8)
        h = rng.randrange(48, 121, 4)
    else:  # square
        s = rng.randrange(64, 257, 8)
        w = h = s
    return w, h, kind


def draw_corner_brackets(draw, x0, y0, x1, y1, color, length, thick):
    # TL
    draw.rectangle([x0, y0, x0 + length, y0 + thick - 1], fill=color)
    draw.rectangle([x0, y0, x0 + thick - 1, y0 + length], fill=color)
    # TR
    draw.rectangle([x1 - length, y0, x1, y0 + thick - 1], fill=color)
    draw.rectangle([x1 - thick + 1, y0, x1, y0 + length], fill=color)
    # BL
    draw.rectangle([x0, y1 - thick + 1, x0 + length, y1], fill=color)
    draw.rectangle([x0, y1 - length, x0 + thick - 1, y1], fill=color)
    # BR
    draw.rectangle([x1 - length, y1 - thick + 1, x1, y1], fill=color)
    draw.rectangle([x1 - thick + 1, y1 - length, x1, y1], fill=color)


def draw_dashed_border(draw, x0, y0, x1, y1, color, dash=4, gap=3, thick=1):
    # top/bottom
    x = x0
    while x <= x1:
        xe = min(x + dash - 1, x1)
        draw.rectangle([x, y0, xe, y0 + thick - 1], fill=color)
        draw.rectangle([x, y1 - thick + 1, xe, y1], fill=color)
        x += dash + gap
    y = y0
    while y <= y1:
        ye = min(y + dash - 1, y1)
        draw.rectangle([x0, y, x0 + thick - 1, ye], fill=color)
        draw.rectangle([x1 - thick + 1, y, x1, ye], fill=color)
        y += dash + gap


def draw_panel(draw, w, h, style, fill, border, accent, hi, lo, rng, frame, frames, anim):
    pad = 0
    x0, y0, x1, y1 = pad, pad, w - 1, h - 1

    # Fill alpha slightly varied for game panels (mostly solid)
    fill_a = 255 if style != "holo" else rng.choice([170, 190, 210])
    if style == "holo":
        fill_a = 160 + int(30 * (0.5 + 0.5 * math.sin(frame / frames * math.pi * 2)))

    # Base fill
    if style == "inset":
        draw.rectangle([x0, y0, x1, y1], fill=rgba(lo, fill_a))
        draw.rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], fill=rgba(fill, fill_a))
    elif style == "raised":
        draw.rectangle([x0, y0, x1, y1], fill=rgba(fill, fill_a))
        draw.rectangle([x0, y0, x1, y0], fill=rgba(hi))
        draw.rectangle([x0, y0, x0, y1], fill=rgba(hi))
        draw.rectangle([x0, y1, x1, y1], fill=rgba(lo))
        draw.rectangle([x1, y0, x1, y1], fill=rgba(lo))
    elif style == "metal":
        for y in range(h):
            t = y / max(1, h - 1)
            c = tuple(int(fill[i] * (1 - t) + lo[i] * t) for i in range(3))
            # subtle horizontal bands
            if (y // 3) % 2 == 0:
                c = lighten(c, 8)
            draw.line([(0, y), (w - 1, y)], fill=rgba(c, fill_a))
    elif style == "rpg":
        draw.rectangle([x0, y0, x1, y1], fill=rgba(fill, fill_a))
        # parchment-ish noise dots
        for _ in range(max(8, w * h // 900)):
            px = rng.randint(2, max(2, w - 3))
            py = rng.randint(2, max(2, h - 3))
            draw.point((px, py), fill=rgba(lighten(fill, rng.randint(-12, 18)), fill_a))
    else:
        draw.rectangle([x0, y0, x1, y1], fill=rgba(fill, fill_a))

    # Borders / chrome
    thick = 1
    if style in ("chunky", "pixel_thick", "arcade"):
        thick = rng.choice([2, 3, 4])
    elif style == "neon":
        thick = rng.choice([1, 2])
    elif style == "double":
        thick = 1
    elif style == "bevel":
        thick = 2

    # Animate border color
    bcol = border
    if anim == "pulse":
        t = (math.sin(frame / frames * math.pi * 2) + 1) / 2
        bcol = tuple(int(border[i] * (1 - 0.35 * t) + hi[i] * (0.35 * t)) for i in range(3))
    elif anim == "flicker":
        if frame % 3 == 0:
            bcol = shade(border, 0.7)
    elif anim == "breathe":
        t = (math.sin(frame / frames * math.pi * 2) + 1) / 2
        bcol = lighten(border, int(25 * t))

    if style == "double":
        draw.rectangle([x0, y0, x1, y1], outline=rgba(bcol), width=1)
        draw.rectangle([x0 + 3, y0 + 3, x1 - 3, y1 - 3], outline=rgba(accent), width=1)
    elif style == "neon":
        draw.rectangle([x0, y0, x1, y1], outline=rgba(bcol), width=thick)
        draw.rectangle([x0 + thick + 1, y0 + thick + 1, x1 - thick - 1, y1 - thick - 1], outline=rgba(lighten(accent, 40)), width=1)
    elif style == "bracket":
        draw.rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], outline=rgba(shade(bcol, 0.75)), width=1)
        length = max(6, min(w, h) // 5)
        draw_corner_brackets(draw, x0, y0, x1, y1, rgba(bcol), length, max(2, thick))
    elif style == "window":
        title_h = max(12, min(28, h // 5))
        draw.rectangle([x0, y0, x1, y0 + title_h], fill=rgba(shade(fill, 0.65), fill_a))
        draw.rectangle([x0, y0, x1, y1], outline=rgba(bcol), width=max(1, thick))
        draw.line([(x0, y0 + title_h), (x1, y0 + title_h)], fill=rgba(accent))
        # fake window buttons
        bx = x1 - 8
        for _ in range(min(3, max(1, w // 80))):
            draw.rectangle([bx - 5, y0 + 4, bx, y0 + 9], fill=rgba(accent))
            bx -= 10
    elif style == "bevel":
        draw.rectangle([x0, y0, x1, y1], outline=rgba(hi), width=1)
        draw.rectangle([x0 + 1, y0 + 1, x1 - 1, y1 - 1], outline=rgba(lo), width=1)
        draw.rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], outline=rgba(bcol), width=1)
    elif style == "scanbox":
        draw.rectangle([x0, y0, x1, y1], outline=rgba(bcol), width=thick)
        # scan line
        sy = 2 + int((frame / frames) * max(1, h - 4))
        draw.rectangle([x0 + 2, sy, x1 - 2, min(y1 - 2, sy + 1)], fill=rgba(accent, 180))
    elif style == "arcade":
        draw.rectangle([x0, y0, x1, y1], outline=rgba(bcol), width=thick)
        draw_dashed_border(draw, x0 + thick + 2, y0 + thick + 2, x1 - thick - 2, y1 - thick - 2, rgba(accent), dash=3, gap=2)
    elif style == "holo":
        draw.rectangle([x0, y0, x1, y1], outline=rgba(bcol), width=1)
        draw_corner_brackets(draw, x0, y0, x1, y1, rgba(hi), max(5, min(w, h) // 6), 1)
    else:
        draw.rectangle([x0, y0, x1, y1], outline=rgba(bcol), width=thick)

    # Optional inner margin line
    if style in ("flat", "rpg", "metal", "chunky", "pixel_thick") and min(w, h) > 40 and rng.random() < 0.55:
        m = 3 + thick
        if x1 - m > x0 + m and y1 - m > y0 + m:
            draw.rectangle([x0 + m, y0 + m, x1 - m, y1 - m], outline=rgba(accent, 200), width=1)

    # Anim overlays
    if anim == "chase":
        perim = 2 * ((w - 1) + (h - 1))
        pos = int((frame / frames) * perim)
        for t in range(10):
            p = (pos + t) % max(1, perim)
            if p < w:
                px, py = p, 0
            elif p < w + h - 1:
                px, py = w - 1, p - (w - 1)
            elif p < 2 * w + h - 2:
                px, py = (w - 1) - (p - (w + h - 2)), h - 1
            else:
                px, py = 0, (h - 1) - (p - (2 * w + h - 3))
            draw.point((px, py), fill=rgba(hi))
    elif anim == "shimmer":
        band = int((frame / frames) * (w + 20)) - 10
        for x in range(max(0, band), min(w, band + 12)):
            alpha = int(90 * (1 - abs(x - band - 6) / 6))
            for y in range(2, h - 2):
                # lighten existing look with translucent streak
                draw.point((x, y), fill=rgba(hi, alpha))
    elif anim == "scan" and style != "scanbox":
        sy = int((frame / frames) * h)
        draw.rectangle([1, sy, w - 2, min(h - 2, sy + 2)], fill=rgba(accent, 100))
    elif anim == "blink_corner":
        on = frame % 2 == 0
        if on:
            L = max(4, min(w, h) // 8)
            draw_corner_brackets(draw, 0, 0, w - 1, h - 1, rgba(hi), L, 2)


def to_gif_frame(img: Image.Image) -> Image.Image:
    """RGBA -> palette GIF frame with transparency index 255."""
    alpha = img.getchannel("A")
    key = (1, 0, 2)
    base = Image.new("RGBA", img.size, (*key, 255))
    composed = Image.alpha_composite(base, img)
    pal = composed.convert("RGB").quantize(colors=48, method=Image.Quantize.MEDIANCUT)
    mask = alpha.point(lambda a: 255 if a < 20 else 0)
    data = list(pal.getdata())
    mdata = list(mask.getdata())
    pal.putdata([255 if m else (254 if v == 255 else v) for v, m in zip(data, mdata)])
    pal.info["transparency"] = 255
    return pal


def generate_one(idx: int, out_dir: str) -> str:
    rng = random.Random(idx * 10007 + 13)
    style = rng.choice(STYLES)
    anim = rng.choice(ANIMS)
    # Bias: some static panels
    if rng.random() < 0.18:
        anim = "static"

    w, h, kind = pick_size(rng)
    pname, base = rng.choice(PALETTE_SEEDS)
    fill = jitter(shade(base, rng.uniform(0.35, 0.75)), 18, rng)
    border = jitter(lighten(base, rng.randint(40, 110)), 20, rng)
    accent = jitter(lighten(base, rng.randint(20, 90)), 25, rng)
    hi = lighten(border, rng.randint(20, 60))
    lo = shade(fill, rng.uniform(0.45, 0.75))

    # Occasional high-contrast accent swap (still game-like)
    if rng.random() < 0.12:
        accent = jitter(rng.choice([(220, 180, 60), (80, 220, 160), (230, 90, 70), (120, 200, 255)]), 15, rng)

    frames_n = rng.choice(FRAMES_CHOICES)
    if anim == "static":
        frames_n = 2
    duration = rng.choice(DURATION_CHOICES)

    frames = []
    for f in range(frames_n):
        # re-seed style-stable random for rpg noise consistency across frames
        local = random.Random(idx * 10007 + 13 + (0 if style != "rpg" else f))
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw_panel(draw, w, h, style, fill, border, accent, hi, lo, local, f, frames_n, anim)
        frames.append(to_gif_frame(img))

    name = f"panel_{idx:04d}_{style}_{kind}_{w}x{h}.gif"
    path = Path(out_dir) / name
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        transparency=255,
        disposal=2,
        optimize=True,
    )
    return name


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("panel_*.gif"):
        old.unlink()

    # Process pool for speed
    workers = max(2, min(8, (Path("/proc/cpuinfo").read_text().count("processor\n") or 4)))
    print(f"generating {COUNT} panels with {workers} workers...")
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(generate_one, i, str(OUT_DIR)) for i in range(1, COUNT + 1)]
        for fut in as_completed(futs):
            name = fut.result()
            done += 1
            if done % 100 == 0 or done == COUNT:
                print(f"  {done}/{COUNT}  last={name}")

    files = sorted(OUT_DIR.glob("panel_*.gif"))
    total = sum(f.stat().st_size for f in files)
    print(f"done: {len(files)} files, {total / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()
