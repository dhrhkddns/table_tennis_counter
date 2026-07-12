#!/usr/bin/env python3
"""Generate 50 fast pixel-art bouncing ball GIFs (200x200, transparent BG)."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).resolve().parent
SIZE = 200
LOGICAL = 40  # 40 * 5 = 200
SCALE = SIZE // LOGICAL
FLOOR_Y = 34  # ball contact line in logical px
FRAMES = 12  # fast loop
DURATION_MS = 40  # fast playback


# (name, body, highlight, shadow/outline)
PALETTES = [
    ("orange", (255, 120, 40), (255, 220, 170), (170, 70, 20)),
    ("white", (245, 245, 250), (255, 255, 255), (160, 160, 175)),
    ("yellow", (255, 220, 50), (255, 250, 180), (190, 150, 20)),
    ("cyan", (60, 220, 255), (200, 250, 255), (20, 130, 170)),
    ("lime", (120, 255, 80), (220, 255, 180), (40, 150, 40)),
    ("magenta", (255, 70, 200), (255, 200, 240), (160, 30, 120)),
    ("red", (255, 70, 70), (255, 190, 180), (160, 30, 30)),
    ("blue", (70, 130, 255), (180, 210, 255), (30, 60, 160)),
    ("peach", (255, 180, 140), (255, 230, 210), (190, 110, 80)),
    ("mint", (140, 255, 210), (230, 255, 245), (50, 160, 130)),
]

# (style_id, style_name)
STYLES = [
    ("classic", "classic"),
    ("outline", "outline"),
    ("chunky", "chunky"),
    ("halftone", "halftone"),
    ("spark", "spark"),
]


def bounce(t: int, period: int, amp: float) -> tuple[float, float]:
    """Return (y_center, squash). Fast parabolic bounce."""
    phase = (t % period) / period
    height = abs(math.sin(phase * math.pi))
    y = FLOOR_Y - 3 - amp * height
    near = 1.0 - min(height * 2.8, 1.0)
    squash = 1.0 - 0.42 * near
    return y, squash


def ellipse_pixels(cx: float, cy: float, rx: float, ry: float):
    rx = max(0.6, rx)
    ry = max(0.6, ry)
    x0, x1 = int(math.floor(cx - rx - 1)), int(math.ceil(cx + rx + 1))
    y0, y1 = int(math.floor(cy - ry - 1)), int(math.ceil(cy + ry + 1))
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if ((x + 0.5 - cx) / rx) ** 2 + ((y + 0.5 - cy) / ry) ** 2 <= 1.0:
                yield x, y


def ring_pixels(cx: float, cy: float, rx: float, ry: float, thick: float = 1.15):
    inner_rx, inner_ry = max(0.4, rx - thick), max(0.4, ry - thick)
    outer = set(ellipse_pixels(cx, cy, rx, ry))
    inner = set(ellipse_pixels(cx, cy, inner_rx, inner_ry))
    return outer - inner


def draw_ball(
    px: dict[tuple[int, int], tuple[int, int, int, int]],
    cx: float,
    cy: float,
    r: float,
    squash: float,
    body: tuple[int, int, int],
    hi: tuple[int, int, int],
    sh: tuple[int, int, int],
    style: str,
    frame: int,
):
    ry = max(1.0, r * squash)
    rx = max(1.0, r / max(squash, 0.55))

    if style == "chunky":
        # Bigger blocky radius
        rx *= 1.15
        ry *= 1.15

    # Shadow/outline ring first
    for x, y in ring_pixels(cx, cy, rx + 0.9, ry + 0.9, 1.1):
        px[(x, y)] = (*sh, 255)

    # Body
    body_pts = list(ellipse_pixels(cx, cy, rx, ry))
    for x, y in body_pts:
        if style == "halftone":
            # two-tone split
            if x < cx:
                px[(x, y)] = (*body, 255)
            else:
                mix = tuple(int((a + b) / 2) for a, b in zip(body, sh))
                px[(x, y)] = (*mix, 255)
        elif style == "outline":
            px[(x, y)] = (*body, 255)
        else:
            px[(x, y)] = (*body, 255)

    if style == "outline":
        # thicker dark rim already; fill slightly inset highlight ring none
        pass

    # Highlight
    hx, hy = cx - rx * 0.35, cy - ry * 0.35
    if style == "chunky":
        for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1), (-1, 0)]:
            px[(int(hx) + dx, int(hy) + dy)] = (*hi, 255)
    elif style == "classic":
        for dx, dy in [(0, 0), (1, 0), (0, -1), (-1, 0), (0, 1)]:
            px[(int(hx) + dx, int(hy) + dy)] = (*hi, 255)
    elif style == "outline":
        for dx, dy in [(0, 0), (1, 0), (0, -1)]:
            px[(int(hx) + dx, int(hy) + dy)] = (*hi, 255)
    elif style == "halftone":
        px[(int(hx), int(hy))] = (*hi, 255)
        px[(int(hx) + 1, int(hy))] = (*hi, 255)
    elif style == "spark":
        for dx, dy in [(0, 0), (1, 0), (0, -1), (-1, 0), (0, 1)]:
            px[(int(hx) + dx, int(hy) + dy)] = (*hi, 255)
        # rotating spark dots
        ang = frame / FRAMES * math.pi * 2
        for k in range(3):
            a = ang + k * (2 * math.pi / 3)
            sx = int(round(cx + math.cos(a) * (rx + 2.2)))
            sy = int(round(cy + math.sin(a) * (ry + 2.2)))
            px[(sx, sy)] = (*hi, 255)


def rgba_from_pixels(px: dict[tuple[int, int], tuple[int, int, int, int]]) -> Image.Image:
    img = Image.new("RGBA", (LOGICAL, LOGICAL), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for (x, y), color in px.items():
        if 0 <= x < LOGICAL and 0 <= y < LOGICAL:
            draw.point((x, y), fill=color)
    return img.resize((SIZE, SIZE), Image.NEAREST)


def to_transparent_gif_frame(img: Image.Image) -> Image.Image:
    """Convert RGBA frame to palette image with index 0 transparent."""
    # Force transparent pixels to a unique key color then map
    key = (1, 2, 3)  # unlikely ball color after remapping; we'll use alpha mask instead
    base = Image.new("RGBA", img.size, (*key, 255))
    composed = Image.alpha_composite(base, img)
    alpha = img.split()[-1]
    # Build palette image
    rgb = composed.convert("RGB")
    pal = rgb.convert("P", palette=Image.ADAPTIVE, colors=255)
    # Find/create transparent index: paste 255 where alpha is low
    mask = Image.eval(alpha, lambda a: 255 if a < 16 else 0)
    # Ensure palette has 256 slots; use index 255 as transparent
    pal.info["transparency"] = 255
    # Where mask is white (transparent), set pixel to 255
    datas = list(pal.getdata())
    mask_data = list(mask.getdata())
    out = [
        255 if m == 255 else (v if v != 255 else 254)
        for v, m in zip(datas, mask_data)
    ]
    pal.putdata(out)
    return pal


def make_one(
    path: Path,
    body: tuple[int, int, int],
    hi: tuple[int, int, int],
    sh: tuple[int, int, int],
    style: str,
    amp: float,
    radius: float,
    wobble: float,
):
    frames: list[Image.Image] = []
    for i in range(FRAMES):
        px: dict[tuple[int, int], tuple[int, int, int, int]] = {}
        y, squash = bounce(i, FRAMES, amp)
        x = LOGICAL / 2 + math.sin(i / FRAMES * math.pi * 2) * wobble
        draw_ball(px, x, y, radius, squash, body, hi, sh, style, i)
        rgba = rgba_from_pixels(px)
        frames.append(to_transparent_gif_frame(rgba))

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=DURATION_MS,
        loop=0,
        transparency=255,
        disposal=2,
        optimize=False,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Clear old numbered outputs if regenerating
    for old in OUT_DIR.glob("ball_*.gif"):
        old.unlink()

    idx = 0
    # 10 palettes × 5 styles = 50
    for pi, (pname, body, hi, sh) in enumerate(PALETTES):
        for si, (sid, sname) in enumerate(STYLES):
            idx += 1
            # Vary amp / radius / wobble slightly per combo for uniqueness
            amp = 18 + (pi % 3) * 2 + (si % 2)
            radius = 3.2 + (si % 3) * 0.55 + (pi % 2) * 0.25
            wobble = 0.0 if si % 2 == 0 else 1.2 + (pi % 3) * 0.4
            name = f"ball_{idx:02d}_{pname}_{sname}.gif"
            path = OUT_DIR / name
            make_one(path, body, hi, sh, sid, amp, radius, wobble)
            print(f"wrote {name} ({path.stat().st_size} bytes)")

    assert idx == 50, idx
    print(f"done: {idx} gifs in {OUT_DIR}")


if __name__ == "__main__":
    main()
