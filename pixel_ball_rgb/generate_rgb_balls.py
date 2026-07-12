#!/usr/bin/env python3
"""Generate many transparent pixel ping-pong balls across an RGB color grid."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).resolve().parent
SIZE = 200
LOGICAL = 40
SCALE = SIZE // LOGICAL
FLOOR_Y = 34
FRAMES = 10
DURATION_MS = 45

# 10 levels per channel -> 1000 colors
LEVELS = [0, 28, 56, 84, 112, 140, 168, 196, 224, 252]


def bounce(t: int, period: int, amp: float) -> tuple[float, float]:
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


def ring_pixels(cx, cy, rx, ry, thick=1.1):
    outer = set(ellipse_pixels(cx, cy, rx, ry))
    inner = set(ellipse_pixels(cx, cy, max(0.4, rx - thick), max(0.4, ry - thick)))
    return outer - inner


def clamp(v: int) -> int:
    return max(0, min(255, v))


def shade(c: tuple[int, int, int], f: float) -> tuple[int, int, int]:
    return tuple(clamp(int(ch * f)) for ch in c)


def lighten(c: tuple[int, int, int], amt: int) -> tuple[int, int, int]:
    return tuple(clamp(ch + amt) for ch in c)


def draw_ball(px, cx, cy, r, squash, body):
    ry = max(1.0, r * squash)
    rx = max(1.0, r / max(squash, 0.55))
    sh = shade(body, 0.55)
    hi = lighten(body, 70)
    # ensure highlight visible on bright colors
    if sum(body) > 600:
        hi = (255, 255, 255)
        sh = shade(body, 0.75)

    for x, y in ring_pixels(cx, cy, rx + 0.85, ry + 0.85, 1.05):
        px[(x, y)] = (*sh, 255)
    for x, y in ellipse_pixels(cx, cy, rx, ry):
        px[(x, y)] = (*body, 255)

    hx, hy = int(cx - rx * 0.35), int(cy - ry * 0.35)
    for dx, dy in [(0, 0), (1, 0), (0, -1), (-1, 0), (0, 1)]:
        px[(hx + dx, hy + dy)] = (*hi, 255)


def rgba_from_pixels(px) -> Image.Image:
    img = Image.new("RGBA", (LOGICAL, LOGICAL), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for (x, y), color in px.items():
        if 0 <= x < LOGICAL and 0 <= y < LOGICAL:
            draw.point((x, y), fill=color)
    return img.resize((SIZE, SIZE), Image.NEAREST)


def to_gif_frame(img: Image.Image) -> Image.Image:
    alpha = img.getchannel("A")
    key = (1, 0, 2)
    base = Image.new("RGBA", img.size, (*key, 255))
    composed = Image.alpha_composite(base, img)
    pal = composed.convert("RGB").quantize(colors=32, method=Image.Quantize.MEDIANCUT)
    mask = alpha.point(lambda a: 255 if a < 16 else 0)
    data = list(pal.getdata())
    mdata = list(mask.getdata())
    pal.putdata([255 if m else (254 if v == 255 else v) for v, m in zip(data, mdata)])
    pal.info["transparency"] = 255
    return pal


def make_one(args):
    idx, r, g, b, out_dir = args
    body = (r, g, b)
    # Near-black: bump slightly so outline still readable
    if sum(body) < 40:
        body = (max(r, 20), max(g, 20), max(b, 20))

    frames = []
    for i in range(FRAMES):
        px = {}
        y, squash = bounce(i, FRAMES, 20)
        x = LOGICAL / 2
        draw_ball(px, x, y, 4.2, squash, body)
        frames.append(to_gif_frame(rgba_from_pixels(px)))

    name = f"ball_{idx:04d}_r{r:03d}_g{g:03d}_b{b:03d}.gif"
    path = Path(out_dir) / name
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=DURATION_MS,
        loop=0,
        transparency=255,
        disposal=2,
        optimize=True,
    )
    return name


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("ball_*.gif"):
        old.unlink()

    jobs = []
    idx = 0
    for r in LEVELS:
        for g in LEVELS:
            for b in LEVELS:
                idx += 1
                jobs.append((idx, r, g, b, str(OUT_DIR)))

    print(f"generating {len(jobs)} transparent RGB pixel balls...")
    workers = 4
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(make_one, job) for job in jobs]
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}")

    files = list(OUT_DIR.glob("ball_*.gif"))
    total = sum(f.stat().st_size for f in files)
    print(f"done: {len(files)} files, {total/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
