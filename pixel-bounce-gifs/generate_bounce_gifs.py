#!/usr/bin/env python3
"""Generate 3 pixel-art style bouncing ping-pong ball GIFs."""

from pathlib import Path

from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).resolve().parent

# Pixel palette (ping-pong orange on dark game-like backgrounds)
BALL = (255, 120, 40)
BALL_HI = (255, 210, 160)
BALL_SHADOW = (180, 70, 20)
FLOOR = (90, 90, 100)
FLOOR_HI = (120, 120, 130)
BG_A = (18, 22, 36)
BG_B = (12, 40, 28)
BG_C = (36, 16, 28)
STAR = (70, 80, 110)
DUST = (255, 200, 80)


def nearest_scale(img: Image.Image, scale: int) -> Image.Image:
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)


def draw_pixel_ball(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, squash: float = 1.0):
    """Draw a tiny pixel ping-pong ball with highlight. squash < 1 flattens vertically."""
    ry = max(1, int(round(r * squash)))
    rx = max(1, int(round(r / max(squash, 0.55))))

    # Shadow ring
    for dy in range(-ry - 1, ry + 2):
        for dx in range(-rx - 1, rx + 2):
            if (dx / (rx + 0.8)) ** 2 + (dy / (ry + 0.8)) ** 2 <= 1.05:
                if (dx / rx) ** 2 + (dy / ry) ** 2 > 1.0:
                    draw.point((cx + dx, cy + dy), fill=BALL_SHADOW)

    # Body
    for dy in range(-ry, ry + 1):
        for dx in range(-rx, rx + 1):
            if (dx / rx) ** 2 + (dy / ry) ** 2 <= 1.0:
                draw.point((cx + dx, cy + dy), fill=BALL)

    # Specular highlight (pixel shine)
    hx, hy = cx - max(1, rx // 3), cy - max(1, ry // 3)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if abs(dx) + abs(dy) <= 1:
                draw.point((hx + dx, hy + dy), fill=BALL_HI)


def draw_floor(draw: ImageDraw.ImageDraw, w: int, h: int, y: int):
    draw.rectangle([0, y, w, h], fill=FLOOR)
    # Pixel checker / edge
    for x in range(0, w, 4):
        draw.point((x, y), fill=FLOOR_HI)
        draw.point((x + 1, y), fill=FLOOR_HI)
    draw.line([(0, y), (w, y)], fill=FLOOR_HI)


def bounce_y(t: float, floor_y: int, amp: float, period: int) -> tuple[float, float]:
    """Return (y, squash). Parabolic bounce with squash on impact."""
    # t in [0, period)
    phase = (t % period) / period
    # triangle-ish bounce: up then down with ease
    # use absolute sine for bounce feel
    import math

    height = abs(math.sin(phase * math.pi))
    y = floor_y - amp * height
    # squash near floor
    near = 1.0 - min(height * 2.5, 1.0)
    squash = 1.0 - 0.35 * near
    return y, squash


def make_gif_classic(path: Path):
    """1) Classic vertical bounce on dark blue stage."""
    W, H = 48, 48
    FLOOR_Y = 40
    AMP = 26
    PERIOD = 16
    frames = []
    for i in range(PERIOD * 2):
        img = Image.new("RGB", (W, H), BG_A)
        draw = ImageDraw.Draw(img)
        # stars
        for sx, sy in [(6, 8), (20, 4), (34, 10), (42, 6), (12, 16), (28, 14)]:
            draw.point((sx, sy), fill=STAR)
        draw_floor(draw, W, H, FLOOR_Y)
        y, squash = bounce_y(i, FLOOR_Y - 4, AMP, PERIOD)
        # soft shadow under ball
        sh = int(4 + (1 - squash) * 4)
        draw.ellipse([22 - sh, FLOOR_Y - 2, 26 + sh, FLOOR_Y], fill=(40, 44, 58))
        draw_pixel_ball(draw, 24, int(round(y)), 4, squash)
        frames.append(nearest_scale(img, 6))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=70,
        loop=0,
        optimize=False,
    )


def make_gif_sideways(path: Path):
    """2) Side-to-side bounce with green court vibe."""
    W, H = 64, 40
    FLOOR_Y = 32
    AMP = 18
    PERIOD = 20
    frames = []
    import math

    for i in range(PERIOD * 2):
        img = Image.new("RGB", (W, H), BG_B)
        draw = ImageDraw.Draw(img)
        # court lines
        draw.line([(0, 8), (W, 8)], fill=(30, 70, 50))
        draw.line([(W // 2, 8), (W // 2, FLOOR_Y)], fill=(50, 110, 70))
        draw_floor(draw, W, H, FLOOR_Y)
        # horizontal travel
        x = 8 + (W - 16) * ((i % PERIOD) / PERIOD)
        # bounce twice per travel
        y, squash = bounce_y(i * 2, FLOOR_Y - 4, AMP, PERIOD // 2)
        # dust puffs near bounce
        height = abs(math.sin(((i * 2) % (PERIOD // 2)) / (PERIOD // 2) * math.pi))
        if height < 0.12:
            for dx, dy in [(-3, 0), (3, 0), (-5, -1), (5, -1)]:
                draw.point((int(x) + dx, FLOOR_Y - 1 + dy), fill=DUST)
        draw.ellipse(
            [int(x) - 3, FLOOR_Y - 2, int(x) + 3, FLOOR_Y],
            fill=(20, 50, 35),
        )
        draw_pixel_ball(draw, int(round(x)), int(round(y)), 4, squash)
        frames.append(nearest_scale(img, 5))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=60,
        loop=0,
        optimize=False,
    )


def make_gif_spin_trail(path: Path):
    """3) High bounce with spin trail / afterimages."""
    W, H = 40, 56
    FLOOR_Y = 48
    AMP = 34
    PERIOD = 18
    frames = []
    import math

    history = []
    for i in range(PERIOD * 2):
        img = Image.new("RGB", (W, H), BG_C)
        draw = ImageDraw.Draw(img)
        # vignette pixels
        for sx, sy in [(4, 6), (36, 8), (10, 20), (30, 18), (6, 30)]:
            draw.point((sx, sy), fill=(60, 30, 50))
        draw_floor(draw, W, H, FLOOR_Y)

        # slight left-right wobble
        wobble = math.sin(i / PERIOD * math.pi * 2) * 3
        y, squash = bounce_y(i, FLOOR_Y - 4, AMP, PERIOD)
        x = 20 + wobble
        history.append((x, y, squash))
        if len(history) > 5:
            history.pop(0)

        # trail (fading afterimages)
        trail_colors = [
            (120, 50, 30),
            (160, 70, 35),
            (200, 90, 40),
            (230, 100, 40),
        ]
        for ti, (tx, ty, ts) in enumerate(history[:-1]):
            c = trail_colors[min(ti, len(trail_colors) - 1)]
            ry = max(1, int(3 * ts))
            rx = max(1, int(3 / max(ts, 0.55)))
            for dy in range(-ry, ry + 1):
                for dx in range(-rx, rx + 1):
                    if (dx / rx) ** 2 + (dy / ry) ** 2 <= 1.0:
                        draw.point((int(tx) + dx, int(ty) + dy), fill=c)

        draw.ellipse(
            [int(x) - 4, FLOOR_Y - 2, int(x) + 4, FLOOR_Y],
            fill=(50, 25, 40),
        )
        draw_pixel_ball(draw, int(round(x)), int(round(y)), 5, squash)
        frames.append(nearest_scale(img, 6))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=55,
        loop=0,
        optimize=False,
    )


def main():
    outs = [
        (OUT_DIR / "pingpong-bounce-classic.gif", make_gif_classic),
        (OUT_DIR / "pingpong-bounce-sideways.gif", make_gif_sideways),
        (OUT_DIR / "pingpong-bounce-trail.gif", make_gif_spin_trail),
    ]
    for path, fn in outs:
        fn(path)
        print(f"wrote {path.name} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
