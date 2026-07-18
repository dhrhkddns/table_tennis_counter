"""_gen_readable_digits.py 기반 — 원본과 같은 검은 셰비 배경 위에 패널 2개.

배경은 shader_bg + 바운스 볼(make_base_frame)이며,
세로형 패널 2개를 나란히 올린다. 텍스트·내부 사각형 없음.
"""

from PIL import Image, ImageDraw, ImageChops
import math
import os

W, H = 1920, 1080  # 최종 GIF 해상도
FPS = 30
N = FPS * 5
FRAME_MS = int(1000 / FPS)

# 논리 캔버스 (검은 배경이 채움). 패널은 그 위에 1:3으로 올린다.
PX_W, PX_H = 320, 180

# 패널 크기 (논리 캔버스 기준, 최종은 W×H로 업스케일)
#   PW = 가로  /  PH = 세로  ← 세로 줄이려면 PH만 낮추면 됨
PW, PH = 80, 160

PANEL_COUNT = 2
PANEL_GAP = 24  # 두 패널 사이 간격
_total_w = PW * PANEL_COUNT + PANEL_GAP * (PANEL_COUNT - 1)
PX0S = [(PX_W - _total_w) // 2 + i * (PW + PANEL_GAP) for i in range(PANEL_COUNT)]
PY0 = (PX_H - PH) // 2

R = 6
BORDER_W = 3

PANELS = [
    ("01_red", (235, 95, 95), (175, 45, 50)),
    ("02_orange", (245, 155, 70), (195, 95, 30)),
    ("03_gold", (245, 205, 80), (190, 145, 35)),
    ("04_lime", (160, 225, 85), (95, 160, 40)),
    ("05_green", (70, 195, 120), (35, 130, 75)),
    ("06_teal", (60, 205, 195), (30, 140, 135)),
    ("07_cyan", (80, 200, 245), (35, 130, 185)),
    ("08_blue", (90, 140, 240), (45, 80, 180)),
    ("09_pink", (245, 120, 180), (185, 55, 120)),
    ("10_slate", (140, 155, 175), (85, 100, 120)),
]
PANEL_COLOR = "green"
PREVIEW = True  # True면 pygame으로 미리보기

GRAYS = [
    (60, 62, 68),
    (110, 114, 122),
    (180, 184, 190),
    (230, 232, 236),
    (150, 154, 160),
    (90, 94, 100),
]
BALL_COLORS = [
    ((255, 70, 70), (190, 30, 30)),
    ((255, 150, 50), (200, 90, 20)),
    ((255, 220, 60), (200, 160, 25)),
    ((80, 210, 90), (40, 140, 50)),
    ((70, 140, 255), (35, 80, 190)),
    ((180, 80, 220), (120, 40, 160)),
    ((245, 248, 255), (175, 185, 205)),
]


def clamp(v, a=0.0, b=1.0):
    return max(a, min(b, v))


def mix(c1, c2, t):
    t = clamp(t)
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def gray_at(u):
    n = len(GRAYS)
    x = (u % 1.0) * n
    i = int(x) % n
    f = x - int(x)
    return mix(GRAYS[i], GRAYS[(i + 1) % n], f)


def hsv(h, s, v):
    h = h % 1.0
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i %= 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return (int(clamp(r) * 255), int(clamp(g) * 255), int(clamp(b) * 255))


def rounded_mask(w, h, r):
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=255)
    return m


OUTER_M = rounded_mask(PW, PH, R)
INNER_M = Image.new("L", (PW, PH), 0)
ImageDraw.Draw(INNER_M).rounded_rectangle(
    [BORDER_W, BORDER_W, PW - 1 - BORDER_W, PH - 1 - BORDER_W],
    radius=max(1, R - BORDER_W),
    fill=255,
)
BORDER_M = ImageChops.subtract(OUTER_M, INNER_M)
border_meta = []
cx, cy = (PW - 1) / 2.0, (PH - 1) / 2.0
bm = BORDER_M.load()
for ly in range(PH):
    for lx in range(PW):
        if bm[lx, ly] > 0:
            ang = (math.atan2(ly - cy, lx - cx) + math.pi) / (2 * math.pi)
            border_meta.append((lx, ly, ang))


def gravity_bounce(u):
    return (4.0 * u * (1.0 - u)) ** 0.92


def make_balls():
    ground = PX_H - 18
    xs = [36, 80, 124, 168, 212, 256, 290]
    amps = [46, 40, 52, 44, 48, 38, 50]
    cycles = [3, 3, 3, 4, 3, 3, 4]
    phases = [0.00, 0.12, 0.28, 0.40, 0.55, 0.70, 0.85]
    rs = [1.9, 1.7, 2.0, 1.8, 1.85, 1.75, 1.95]
    x_amps = [4, 5, 3, 5, 4, 5, 3]
    return [
        {
            "x": xs[i],
            "ground": ground,
            "amp": amps[i],
            "cycles": cycles[i],
            "phase": phases[i],
            "r": rs[i],
            "x_amp": x_amps[i],
            "x_cycles": 1,
            "light": BALL_COLORS[i][0],
            "dark": BALL_COLORS[i][1],
        }
        for i in range(7)
    ]


def ball_at(b, frame_i):
    u = (frame_i / N * b["cycles"] + b["phase"]) % 1.0
    h = gravity_bounce(u)
    if h < 0.16:
        k = h / 0.16
        squash = 0.55 + 0.45 * k
        stretch = 1.35 - 0.35 * k
    elif h > 0.82:
        squash, stretch = 1.1, 0.9
    else:
        squash, stretch = 1.0, 1.0
    y = b["ground"] - b["r"] * squash - h * b["amp"]
    xu = (frame_i / N * b["x_cycles"] + b["phase"] * 0.5) % 1.0
    x = b["x"] + math.sin(xu * math.pi * 2) * b["x_amp"]
    return x, y, b["r"] * stretch, b["r"] * squash, h, b["light"], b["dark"]


def draw_ball(draw, x, y, rx, ry, h, light, dark):
    hi = mix(light, (255, 255, 255), 0.45)
    draw.ellipse([x - rx, y - ry, x + rx, y + ry], fill=light)
    draw.ellipse([x - rx + 0.5, y, x + rx - 0.5, y + ry], fill=dark)
    draw.ellipse(
        [x - rx * 0.55, y - ry * 0.7, x - rx * 0.05, y - ry * 0.12], fill=hi
    )
    sw = rx * (1.25 - 0.6 * h)
    sh = max(1, 1.2 + (1 - h))
    gy = y + ry + 2
    draw.ellipse([x - sw, gy - sh * 0.3, x + sw, gy + sh], fill=(18, 18, 22))


def shader_bg(t):
    """원본과 동일한 다크 셰비 배경."""
    img = Image.new("RGB", (PX_W, PX_H), (0, 0, 0))
    pix = img.load()
    for y in range(PX_H):
        for x in range(PX_W):
            nx, ny = x / PX_W, y / PX_H
            cell = 10
            cx = (x % cell) / cell
            cy = (y % cell) / cell
            diag = abs(cx - 0.5) + abs(cy - 0.5)
            grid = 1.0 if diag < 0.35 else 0.0
            wave1 = 0.5 + 0.5 * math.sin((nx * 6 + ny * 2 + t * 2) * math.pi * 2)
            wave2 = 0.5 + 0.5 * math.sin((nx * -3 + ny * 7 - t * 1.5) * math.pi * 2)
            ring = math.sin(math.hypot(nx - 0.5, ny - 0.5) * 18 - t * 2.5 * math.pi)
            glow = 0.12 * wave1 * wave2 + 0.06 * max(0, ring) + 0.08 * grid * wave1
            hh = 0.58 + 0.12 * wave2 + 0.05 * math.sin(t * math.pi * 2)
            val = glow * 1.4
            if y % 4 == 0:
                val += 0.012 * (0.5 + 0.5 * math.sin(t * math.pi * 2 * 0.4))
            dx, dy = nx - 0.5, ny - 0.5
            vig = 1 - clamp((dx * dx + dy * dy) * 1.2, 0, 0.5)
            val *= vig
            pix[x, y] = (0, 0, 0) if val < 0.02 else hsv(hh % 1.0, 0.7, clamp(val))
    return img


def make_base_frame(i, ball_hist):
    """원본과 동일: shader 배경 + 상하 액센트 + 코너 + 볼."""
    t = i / N
    img = shader_bg(t)
    pix = img.load()
    draw = ImageDraw.Draw(img)
    bar = 8
    accent = (40, 70, 90)
    for y in range(bar):
        a = 0.22 * (1 - y / bar)
        for x in range(PX_W):
            pix[x, y] = mix(pix[x, y], accent, a)
            pix[x, PX_H - 1 - y] = mix(pix[x, PX_H - 1 - y], accent, a)
    blink = 0.55 + 0.35 * (0.5 + 0.5 * math.sin(t * math.pi * 2))
    line = mix((50, 120, 140), (120, 230, 245), blink)
    for x in range(PX_W):
        pix[x, bar] = mix(pix[x, bar], line, 0.5)
        pix[x, PX_H - 1 - bar] = mix(pix[x, PX_H - 1 - bar], line, 0.5)
    WHITE = (220, 230, 240)
    m, L = 5, 14
    for ox, oy, sx, sy in [
        (m, m, 1, 1),
        (PX_W - m - 1, m, -1, 1),
        (m, PX_H - m - 1, 1, -1),
        (PX_W - m - 1, PX_H - m - 1, -1, -1),
    ]:
        for j in range(L):
            pix[ox + j * sx, oy] = WHITE
            pix[ox, oy + j * sy] = WHITE
    for x, y, rx, ry, h, light, dark in sorted(ball_hist[i], key=lambda p: p[1]):
        draw_ball(draw, x, y, rx, ry, h, light, dark)
    return img, t


def blit_one_panel(layer, px0, py0, t, light, dark):
    """레이어에 패널 하나(그라데이션+테두리)만 그린다."""
    sh = Image.new("RGBA", (PW, PH), (0, 0, 0, 70))
    sh.putalpha(OUTER_M)
    layer.paste(sh, (px0 + 2, py0 + 3), sh)

    im = INNER_M.load()
    shaded = Image.new("RGBA", (PW, PH), (0, 0, 0, 0))
    sp = shaded.load()
    for ly in range(PH):
        ny = ly / max(1, PH - 1)
        col = mix(light, dark, ny * 0.7)
        for lx in range(PW):
            if im[lx, ly] > 0:
                sp[lx, ly] = (*col, 235)
    layer.paste(shaded, (px0, py0), shaded)

    border = Image.new("RGBA", (PW, PH), (0, 0, 0, 0))
    bp = border.load()
    for lx, ly, ang in border_meta:
        gcol = gray_at(ang + t)
        shimmer = 0.85 + 0.15 * math.sin((ang * 4 + t * 2) * math.pi * 2)
        gcol = tuple(int(clamp(c * shimmer, 0, 255)) for c in gcol)
        bp[lx, ly] = (*gcol, 255)
    layer.paste(border, (px0, py0), border)


def draw_panels_on_bg(base_img, t, light, dark):
    """원본 배경 위에 1:3 패널 2개를 나란히 합성한다."""
    layer = Image.new("RGBA", (PX_W, PX_H), (0, 0, 0, 0))
    for px0 in PX0S:
        blit_one_panel(layer, px0, PY0, t, light, dark)
    return Image.alpha_composite(base_img.convert("RGBA"), layer).convert("RGB")


def resolve_panels(color_key):
    key = (color_key or "").strip().lower()
    if key in ("", "all", "*"):
        return list(PANELS)
    matched = [p for p in PANELS if p[0].split("_", 1)[-1] == key or p[0] == key]
    if not matched:
        names = ", ".join(p[0].split("_", 1)[-1] for p in PANELS)
        raise SystemExit(f"Unknown PANEL_COLOR={color_key!r}. Use one of: {names}, all")
    return matched


def preview_gif(path):
    import pygame

    path = os.fspath(path)
    frames = []
    durations = []
    with Image.open(path) as img:
        w, h = img.size
        try:
            while True:
                rgba = img.convert("RGBA")
                frames.append(rgba)
                durations.append(img.info.get("duration", FRAME_MS))
                img.seek(img.tell() + 1)
        except EOFError:
            pass

    pygame.init()
    info = pygame.display.Info()
    scale = min(1.0, info.current_w / w * 0.9, info.current_h / h * 0.9)
    vw, vh = max(1, int(w * scale)), max(1, int(h * scale))
    screen = pygame.display.set_mode((vw, vh))
    pygame.display.set_caption(os.path.basename(path))

    surfs = []
    for fr in frames:
        s = pygame.image.fromstring(fr.tobytes(), fr.size, fr.mode).convert_alpha()
        if scale != 1.0:
            s = pygame.transform.scale(s, (vw, vh))
        surfs.append(s)

    clock = pygame.time.Clock()
    idx = 0
    acc = 0
    running = True
    print(f"Preview: {path}  (ESC to close)", flush=True)
    while running:
        dt = clock.tick(60)
        acc += dt
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        if acc >= durations[idx]:
            acc = 0
            idx = (idx + 1) % len(surfs)
        screen.fill((0, 0, 0))
        screen.blit(surfs[idx], (0, 0))
        pygame.display.flip()
    pygame.quit()


def main():
    out_dir = os.path.join(os.path.expanduser("~"), "Desktop", "ui_panel_gifs_1x3")
    os.makedirs(out_dir, exist_ok=True)
    panels = resolve_panels(PANEL_COLOR)
    print(
        f"Shader BG {PX_W}x{PX_H} | "
        f"{PANEL_COUNT}× Panel {PW}x{PH} (1:3) @ x={PX0S}, y={PY0} | "
        f"PANEL_COLOR={PANEL_COLOR!r} → {[p[0] for p in panels]}",
        flush=True,
    )

    print("Building shader backgrounds + balls...", flush=True)
    balls = make_balls()
    ball_hist = [[ball_at(b, i) for b in balls] for i in range(N)]
    bases = [make_base_frame(i, ball_hist) for i in range(N)]

    saved = []
    for name, light, dark in panels:
        print(f"Rendering {name}...", flush=True)
        frames_p = []
        for img, t in bases:
            framed = draw_panels_on_bg(img.copy(), t, light, dark)
            big = framed.resize((W, H), Image.Resampling.NEAREST)
            q = big.convert(
                "P", palette=Image.Palette.ADAPTIVE, colors=192, dither=Image.Dither.NONE
            )
            frames_p.append(q)
        path = os.path.join(out_dir, f"game_ui_panel_1x3_{name}.gif")
        frames_p[0].save(
            path,
            save_all=True,
            append_images=frames_p[1:],
            duration=FRAME_MS,
            loop=0,
            optimize=False,
            disposal=2,
        )
        print(f"  saved {path}", flush=True)
        saved.append(path)
    print("done", flush=True)

    if PREVIEW and saved:
        try:
            preview_gif(saved[0])
        except Exception as e:
            print(f"Preview skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
