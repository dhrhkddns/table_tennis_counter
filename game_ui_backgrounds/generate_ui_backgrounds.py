#!/usr/bin/env python3
"""Generate 100 animated 1920x1080 game UI panel backgrounds as GIFs."""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).resolve().parent
W, H = 1920, 1080
LW, LH = 480, 270  # logical render size (4x upscale)
FRAMES = 10
DURATION = 120

THEMES = [
    ("navy", (12, 20, 42), (24, 38, 72), (70, 120, 220), (140, 190, 255)),
    ("purple", (22, 10, 36), (44, 20, 68), (160, 80, 220), (220, 170, 255)),
    ("emerald", (8, 28, 22), (16, 52, 40), (50, 200, 140), (150, 255, 210)),
    ("crimson", (36, 10, 16), (72, 18, 28), (220, 70, 90), (255, 170, 180)),
    ("amber", (32, 22, 6), (64, 42, 10), (240, 170, 50), (255, 220, 140)),
    ("slate", (18, 22, 28), (36, 42, 54), (120, 140, 170), (200, 210, 225)),
    ("cyan", (6, 26, 36), (12, 50, 70), (40, 200, 240), (170, 245, 255)),
    ("rose", (34, 14, 28), (68, 26, 52), (230, 100, 170), (255, 190, 220)),
    ("olive", (24, 26, 10), (48, 50, 18), (170, 190, 60), (230, 240, 150)),
    ("void", (8, 8, 14), (18, 18, 28), (90, 90, 120), (170, 170, 200)),
]

LAYOUTS = [
    ("center", "center_panel"),
    ("topbar", "top_hud"),
    ("left", "left_sidebar"),
    ("right", "right_sidebar"),
    ("dual", "dual_sidebar"),
    ("bottom", "bottom_dock"),
    ("triple", "triple_zone"),
    ("grid", "card_grid"),
    ("corner", "corner_frame"),
    ("banner", "wide_banner"),
]

ANIM = [
    "pulse",
    "scan",
    "grid",
    "stars",
    "sweep",
    "shimmer",
    "noise",
    "chase",
    "wave",
    "flicker",
]


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_c(c1, c2, t):
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def rect_panel(draw, box, fill, border, accent, thickness=2):
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=fill)
    draw.rectangle([x0, y0, x1, y1], outline=border, width=thickness)
    # inner accent line
    draw.rectangle([x0 + 3, y0 + 3, x1 - 3, y1 - 3], outline=accent, width=1)


def layout_panels(layout: str) -> list[tuple[int, int, int, int]]:
    m = 14
    if layout == "center_panel":
        return [(m + 70, m + 20, LW - m - 70, LH - m - 20)]
    if layout == "top_hud":
        return [(m, m, LW - m, m + 42), (m, m + 52, LW - m, LH - m)]
    if layout == "left_sidebar":
        return [(m, m, m + 110, LH - m), (m + 120, m, LW - m, LH - m)]
    if layout == "right_sidebar":
        return [(LW - m - 110, m, LW - m, LH - m), (m, m, LW - m - 120, LH - m)]
    if layout == "dual_sidebar":
        return [
            (m, m, m + 78, LH - m),
            (m + 88, m, LW - m - 88, LH - m),
            (LW - m - 78, m, LW - m, LH - m),
        ]
    if layout == "bottom_dock":
        return [(m, m, LW - m, LH - m - 48), (m, LH - m - 40, LW - m, LH - m)]
    if layout == "triple_zone":
        return [
            (m, m, LW - m, m + 36),
            (m, m + 46, LW - m, LH - m - 46),
            (m, LH - m - 36, LW - m, LH - m),
        ]
    if layout == "card_grid":
        cols, rows = 3, 2
        gap = 8
        pw = (LW - 2 * m - gap * (cols - 1)) // cols
        ph = (LH - 2 * m - gap * (rows - 1)) // rows
        panels = []
        for r in range(rows):
            for c in range(cols):
                x0 = m + c * (pw + gap)
                y0 = m + r * (ph + gap)
                panels.append((x0, y0, x0 + pw, y0 + ph))
        return panels
    if layout == "corner_frame":
        return [(m + 40, m + 24, LW - m - 40, LH - m - 24)]
    if layout == "wide_banner":
        return [(m + 30, m + 70, LW - m - 30, m + 130), (m, m + 145, LW - m, LH - m)]
    return [(m, m, LW - m, LH - m)]


def draw_bg_base(draw, bg0, bg1, frame: int, anim: str, seed: int):
    # vertical gradient background
    for y in range(LH):
        t = y / (LH - 1)
        if anim == "wave":
            t += 0.04 * math.sin((y / 18.0) + frame * 0.55)
            t = max(0, min(1, t))
        c = lerp_c(bg0, bg1, t)
        draw.line([(0, y), (LW, y)], fill=c)

    if anim == "stars":
        rng = random.Random(seed)
        for _ in range(55):
            x = rng.randint(0, LW - 1)
            y = rng.randint(0, LH - 1)
            phase = (x * 7 + y * 13 + seed) % FRAMES
            bright = 1.0 if (frame + phase) % FRAMES < FRAMES // 3 else 0.35
            if bright > 0.7:
                draw.point((x, y), fill=(220, 230, 255))
            else:
                draw.point((x, y), fill=(70, 80, 110))

    if anim == "grid":
        step = 18
        off = (frame * 2) % step
        grid_c = lerp_c(bg1, (255, 255, 255), 0.08)
        for x in range(-step, LW + step, step):
            draw.line([(x + off, 0), (x + off, LH)], fill=grid_c)
        for y in range(-step, LH + step, step):
            draw.line([(0, y + off), (LW, y + off)], fill=grid_c)

    if anim == "scan":
        sy = int((frame / FRAMES) * (LH + 40)) - 20
        draw.rectangle([0, sy, LW, sy + 6], fill=lerp_c(bg1, (255, 255, 255), 0.12))

    if anim == "noise":
        rng = random.Random(seed + frame * 991)
        for _ in range(120):
            x = rng.randint(0, LW - 1)
            y = rng.randint(0, LH - 1)
            draw.point((x, y), fill=lerp_c(bg0, bg1, rng.random()))

    if anim == "shimmer":
        band = int((frame / FRAMES) * (LW + 80)) - 40
        for x in range(max(0, band), min(LW, band + 30)):
            alpha = 1 - abs(x - band - 15) / 15
            c = lerp_c(bg1, (255, 255, 255), 0.15 * alpha)
            draw.line([(x, 0), (x, LH)], fill=c)


def border_pulse(accent, frame: int, anim: str):
    if anim == "pulse":
        t = (math.sin(frame / FRAMES * math.pi * 2) + 1) / 2
        return lerp_c(accent, (255, 255, 255), 0.25 * t)
    if anim == "flicker":
        t = 0.85 if frame % 3 else 1.0
        return tuple(int(c * t) for c in accent)
    return accent


def draw_chase(draw, panels, accent, frame: int):
    for i, box in enumerate(panels):
        x0, y0, x1, y1 = box
        perim = 2 * ((x1 - x0) + (y1 - y0))
        pos = int((frame / FRAMES) * perim) + i * 20
        seg = 18
        coords = []
        w, h = x1 - x0, y1 - y0
        for t in range(seg):
            p = (pos + t) % perim
            if p < w:
                coords.append((x0 + p, y0))
            elif p < w + h:
                coords.append((x1, y0 + p - w))
            elif p < 2 * w + h:
                coords.append((x1 - (p - w - h), y1))
            else:
                coords.append((x0, y1 - (p - 2 * w - h)))
        for pt in coords:
            draw.point(pt, fill=accent)
            draw.point((pt[0], pt[1] + 1), fill=accent)


def draw_sweep(draw, accent, frame: int):
    cx, cy = LW // 2, LH // 2
    angle = (frame / FRAMES) * math.pi * 2
    for r in range(20, max(LW, LH)):
        x = int(cx + math.cos(angle) * r)
        y = int(cy + math.sin(angle) * r)
        if 0 <= x < LW and 0 <= y < LH:
            draw.point((x, y), fill=lerp_c(accent, (255, 255, 255), 0.4))


def draw_corner_accents(draw, accent, frame: int):
    pulse = (math.sin(frame / FRAMES * math.pi * 2) + 1) / 2
    c = lerp_c(accent, (255, 255, 255), 0.3 * pulse)
    s = 16
    for x0, y0 in [(8, 8), (LW - 8 - s, 8), (8, LH - 8 - s), (LW - 8 - s, LH - 8 - s)]:
        draw.rectangle([x0, y0, x0 + s, y0 + 3], fill=c)
        draw.rectangle([x0, y0, x0 + 3, y0 + s], fill=c)


def render_frame(layout: str, theme, anim: str, frame: int, seed: int) -> Image.Image:
    bg0, bg1, accent, hi = theme[1], theme[2], theme[3], theme[4]
    img = Image.new("RGB", (LW, LH), bg0)
    draw = ImageDraw.Draw(img)
    draw_bg_base(draw, bg0, bg1, frame, anim, seed)

    panels = layout_panels(layout)
    panel_fill = lerp_c(bg0, bg1, 0.35)
    border = border_pulse(accent, frame, anim)

    for box in panels:
        rect_panel(draw, box, panel_fill, border, hi, thickness=2)

    if anim == "chase":
        draw_chase(draw, panels, hi, frame)
    if anim == "sweep":
        draw_sweep(draw, accent, frame)

    draw_corner_accents(draw, accent, frame)

    # upscale
    up = img.resize((W, H), Image.BILINEAR)
    pal = up.quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    return pal


def save_gif(path: Path, layout: str, theme, anim: str, seed: int):
    frames = [render_frame(layout, theme, anim, i, seed) for i in range(FRAMES)]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=DURATION,
        loop=0,
        optimize=True,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("ui_bg_*.gif"):
        old.unlink()

    idx = 0
    for li, (lshort, lname) in enumerate(LAYOUTS):
        for ti, theme in enumerate(THEMES):
            idx += 1
            tname = theme[0]
            anim = ANIM[(li + ti) % len(ANIM)]
            fname = f"ui_bg_{idx:03d}_{lshort}_{tname}.gif"
            path = OUT_DIR / fname
            seed = idx * 1337 + li * 17 + ti
            save_gif(path, lname, theme, anim, seed)
            print(f"{fname}  {path.stat().st_size // 1024}KB  anim={anim}")

    assert idx == 100
    total = sum(f.stat().st_size for f in OUT_DIR.glob("ui_bg_*.gif"))
    print(f"done: {idx} gifs, total {total // (1024*1024)}MB")


if __name__ == "__main__":
    main()
