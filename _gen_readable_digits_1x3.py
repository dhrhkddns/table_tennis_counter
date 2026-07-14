"""_gen_readable_digits.py 기반 — 검은 배경을 따로 두고 1:3 패널을 올린다.

검은색 배경은 make_black_background()로 별도 생성(또는 shader_backgrounds GIF에서
로드)한 뒤, width:height = 1:3 직사각형 랭킹 패널을 가운데 합성한다.
"""

from PIL import Image, ImageDraw, ImageChops
import math
import os

W, H = 1920, 1080  # 최종 GIF 해상도
FPS = 12
N = FPS * 5
FRAME_MS = int(1000 / FPS)

# 논리 캔버스 (검은 배경이 채움). 패널은 그 위에 1:3으로 올린다.
PX_W, PX_H = 320, 180

# width:height = 1:3 직사각형 패널 (세로로 긴 사이드바형)
PW, PH = 60, 180  # 60:180 = 1:3
assert PW * 3 == PH, f"panel must be 1:3, got {PW}x{PH}"

# 패널을 검은 캔버스 가로 중앙·세로 풀높이에 배치
PX0 = (PX_W - PW) // 2
PY0 = (PX_H - PH) // 2

R = 6
BORDER_W = 3

# 검은 배경 소스:
#   "solid"  — 순수 검정 Image.new
#   "shader" — shader_backgrounds/ 의 GIF 프레임을 따로 로드
BLACK_BG_SOURCE = "solid"
SHADER_BG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shader_backgrounds")
SHADER_BG_FILE = "shader_bg_01_ink_ripple.gif"  # BLACK_BG_SOURCE == "shader" 일 때

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
PREVIEW = False  # True면 pygame으로 미리보기 (헤드리스에선 False 권장)

GRAYS = [
    (60, 62, 68),
    (110, 114, 122),
    (180, 184, 190),
    (230, 232, 236),
    (150, 154, 160),
    (90, 94, 100),
]

RANK_COUNT = 16
COLOR_BLUE = (50, 130, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (255, 55, 55)
NUM_COL_W = 22
DIGIT_W, DIGIT_H = 6, 7
DIGIT_GAP = 1
TEXT_NUDGE_X = 2


def rank_color(rank):
    if rank <= 4:
        return COLOR_BLUE
    if rank <= 13:
        return COLOR_WHITE
    return COLOR_RED


DIGIT = {
    "0": [
        "011110",
        "110011",
        "110011",
        "110011",
        "110011",
        "110011",
        "011110",
    ],
    "1": [
        "001100",
        "011100",
        "001100",
        "001100",
        "001100",
        "001100",
        "011110",
    ],
    "2": [
        "011110",
        "110011",
        "000011",
        "001110",
        "011000",
        "110000",
        "111111",
    ],
    "3": [
        "011110",
        "110011",
        "000011",
        "001110",
        "000011",
        "110011",
        "011110",
    ],
    "4": [
        "001100",
        "011100",
        "110100",
        "110100",
        "111111",
        "000100",
        "000100",
    ],
    "5": [
        "111111",
        "110000",
        "111110",
        "000011",
        "000011",
        "110011",
        "011110",
    ],
    "6": [
        "011110",
        "110000",
        "110000",
        "111110",
        "110011",
        "110011",
        "011110",
    ],
    "7": [
        "111111",
        "000011",
        "000110",
        "001100",
        "011000",
        "011000",
        "011000",
    ],
    "8": [
        "011110",
        "110011",
        "110011",
        "011110",
        "110011",
        "110011",
        "011110",
    ],
    "9": [
        "011110",
        "110011",
        "110011",
        "011111",
        "000011",
        "000011",
        "011110",
    ],
}


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


def make_black_background(w=PX_W, h=PX_H):
    """검은색 배경을 패널과 분리해서 따로 만든다."""
    return Image.new("RGB", (w, h), (0, 0, 0))


def load_black_background_frames(n=N, w=PX_W, h=PX_H):
    """
    검은 배경을 따로 가져온다.
    - solid: 순수 검정 프레임 n장
    - shader: shader_backgrounds GIF를 로드해 리사이즈·루프
    """
    source = (BLACK_BG_SOURCE or "solid").strip().lower()
    if source == "solid":
        return [make_black_background(w, h) for _ in range(n)]

    if source != "shader":
        raise SystemExit(f"Unknown BLACK_BG_SOURCE={BLACK_BG_SOURCE!r}. Use solid or shader")

    path = os.path.join(SHADER_BG_DIR, SHADER_BG_FILE)
    if not os.path.isfile(path):
        print(f"shader bg missing ({path}), falling back to solid black")
        return [make_black_background(w, h) for _ in range(n)]

    frames = []
    with Image.open(path) as img:
        try:
            while True:
                frames.append(img.convert("RGB").resize((w, h), Image.Resampling.NEAREST))
                img.seek(img.tell() + 1)
        except EOFError:
            pass
    if not frames:
        return [make_black_background(w, h) for _ in range(n)]
    return [frames[i % len(frames)].copy() for i in range(n)]


def blit_digit(mask, ch, x, y):
    pattern = DIGIT[ch]
    pix = mask.load()
    for dy, row in enumerate(pattern):
        for dx, bit in enumerate(row):
            if bit == "1":
                xx, yy = x + dx, y + dy
                if 0 <= xx < PW and 0 <= yy < PH:
                    pix[xx, yy] = 255


def build_rank_text_layers(_frame_i=0):
    mask = Image.new("L", (PW, PH), 0)
    blue_m = Image.new("L", (PW, PH), 0)
    white_m = Image.new("L", (PW, PH), 0)
    red_m = Image.new("L", (PW, PH), 0)
    left = BORDER_W + 3
    top = BORDER_W + 2
    bottom = PH - BORDER_W - 2
    row_h = (bottom - top) / RANK_COUNT

    for i in range(RANK_COUNT):
        rank = i + 1
        label = str(rank)
        row_cy = top + (i + 0.5) * row_h
        total_w = len(label) * DIGIT_W + (len(label) - 1) * DIGIT_GAP
        tx = left + NUM_COL_W - total_w - 1 - TEXT_NUDGE_X
        ty = int(row_cy - DIGIT_H / 2)
        ty = max(int(top + i * row_h) + 1, min(ty, int(top + (i + 1) * row_h) - DIGIT_H - 1))
        row_mask = Image.new("L", (PW, PH), 0)
        x = tx
        for ch in label:
            blit_digit(row_mask, ch, x, ty)
            x += DIGIT_W + DIGIT_GAP
        row_mask = ImageChops.multiply(row_mask, INNER_M)
        mask = ImageChops.lighter(mask, row_mask)
        col = rank_color(rank)
        if col == COLOR_BLUE:
            blue_m = ImageChops.lighter(blue_m, row_mask)
        elif col == COLOR_RED:
            red_m = ImageChops.lighter(red_m, row_mask)
        else:
            white_m = ImageChops.lighter(white_m, row_mask)

    rgb = Image.new("RGB", (PW, PH), (0, 0, 0))
    rgb.paste(Image.new("RGB", (PW, PH), COLOR_BLUE), mask=blue_m)
    rgb.paste(Image.new("RGB", (PW, PH), COLOR_WHITE), mask=white_m)
    rgb.paste(Image.new("RGB", (PW, PH), COLOR_RED), mask=red_m)
    return rgb, mask, blue_m, white_m, red_m


def draw_rank_chrome(panel_img, frame_i):
    draw = ImageDraw.Draw(panel_img)
    left = BORDER_W + 2
    right = PW - BORDER_W - 3
    top = BORDER_W + 2
    bottom = PH - BORDER_W - 2
    row_h = (bottom - top) / RANK_COUNT
    line_x0 = left + NUM_COL_W

    draw.rectangle([left, top, line_x0 - 1, bottom], fill=(0, 0, 0, 160))

    sel = int(frame_i * RANK_COUNT / N) % RANK_COUNT
    y0 = int(top + sel * row_h)
    y1 = int(top + (sel + 1) * row_h)
    draw.rectangle([left, y0, right, max(y0 + 1, y1 - 1)], fill=(0, 0, 0, 255))

    for i in range(RANK_COUNT + 1):
        yy = int(top + i * row_h)
        draw.line([(line_x0, yy), (right, yy)], fill=(0, 0, 0, 255), width=1)


def draw_panel_on_black(black_bg, t, light, dark, frame_i):
    """따로 가져온 검은 배경 위에 1:3 패널을 합성한다."""
    layer = Image.new("RGBA", (PX_W, PX_H), (0, 0, 0, 0))

    # 패널 그림자
    sh = Image.new("RGBA", (PW, PH), (0, 0, 0, 70))
    sh.putalpha(OUTER_M)
    layer.paste(sh, (PX0 + 2, PY0 + 3), sh)

    # 패널 본체(세로 그라데이션)
    im = INNER_M.load()
    shaded = Image.new("RGBA", (PW, PH), (0, 0, 0, 0))
    sp = shaded.load()
    for ly in range(PH):
        ny = ly / max(1, PH - 1)
        col = mix(light, dark, ny * 0.7)
        for lx in range(PW):
            if im[lx, ly] > 0:
                sp[lx, ly] = (*col, 235)
    layer.paste(shaded, (PX0, PY0), shaded)

    # 랭킹 크롬
    panel_local = Image.new("RGBA", (PW, PH), (0, 0, 0, 0))
    draw_rank_chrome(panel_local, frame_i)
    r, g, b, a = panel_local.split()
    a = ImageChops.multiply(a, INNER_M)
    panel_local = Image.merge("RGBA", (r, g, b, a))
    layer.paste(panel_local, (PX0, PY0), panel_local)

    # 테두리 쉬머
    border = Image.new("RGBA", (PW, PH), (0, 0, 0, 0))
    bp = border.load()
    for lx, ly, ang in border_meta:
        gcol = gray_at(ang + t)
        shimmer = 0.85 + 0.15 * math.sin((ang * 4 + t * 2) * math.pi * 2)
        gcol = tuple(int(clamp(c * shimmer, 0, 255)) for c in gcol)
        bp[lx, ly] = (*gcol, 255)
    layer.paste(border, (PX0, PY0), border)

    return Image.alpha_composite(black_bg.convert("RGBA"), layer).convert("RGB")


def stamp_rank_text(rgb_img, text_rgb_full, mask_full):
    rgb_img.paste(text_rgb_full, mask=mask_full)
    return rgb_img


def force_palette_rank_colors(img_p):
    pal = img_p.getpalette()
    if not pal:
        return img_p
    pal = list(pal) + [0] * (768 - len(pal))
    for i in range(256):
        r, g, b = pal[i * 3], pal[i * 3 + 1], pal[i * 3 + 2]
        if r >= 250 and g >= 250 and b >= 250:
            pal[i * 3 : i * 3 + 3] = list(COLOR_WHITE)
    pal[253 * 3 : 253 * 3 + 3] = list(COLOR_BLUE)
    pal[254 * 3 : 254 * 3 + 3] = list(COLOR_RED)
    pal[255 * 3 : 255 * 3 + 3] = list(COLOR_WHITE)
    img_p.putpalette(pal[:768])
    return img_p


def stamp_rank_colors_on_p(img_p, blue_big, white_big, red_big):
    pix = list(img_p.getdata())
    bdat = blue_big.getdata()
    wdat = white_big.getdata()
    rdat = red_big.getdata()
    out = []
    for p, b, w, r in zip(pix, bdat, wdat, rdat):
        if b:
            out.append(253)
        elif r:
            out.append(254)
        elif w:
            out.append(255)
        else:
            out.append(p)
    img_p.putdata(out)
    return img_p


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
    print(f"Preview: {path}  (ESC to close)")
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
        f"Black BG ({BLACK_BG_SOURCE}) {PX_W}x{PX_H} | "
        f"Panel {PW}x{PH} (1:3) @ ({PX0},{PY0}) | "
        f"PANEL_COLOR={PANEL_COLOR!r} → {[p[0] for p in panels]}"
    )

    # 검은 배경을 패널과 분리해서 미리 가져옴
    print("Loading black backgrounds separately...")
    black_bgs = load_black_background_frames(N, PX_W, PX_H)

    print("Building digit layers...")
    text_rgb, mask, blue_m, white_m, red_m = build_rank_text_layers(0)

    def to_full(layer):
        full = Image.new(layer.mode, (PX_W, PX_H), 0 if layer.mode == "L" else (0, 0, 0))
        full.paste(layer, (PX0, PY0))
        return full

    text_full = to_full(text_rgb)
    mask_full = to_full(mask)
    blue_full = to_full(blue_m)
    white_full = to_full(white_m)
    red_full = to_full(red_m)

    text_big = text_full.resize((W, H), Image.Resampling.NEAREST)
    mask_big = mask_full.resize((W, H), Image.Resampling.NEAREST)
    blue_big = blue_full.resize((W, H), Image.Resampling.NEAREST)
    white_big = white_full.resize((W, H), Image.Resampling.NEAREST)
    red_big = red_full.resize((W, H), Image.Resampling.NEAREST)

    saved = []
    for name, light, dark in panels:
        print(f"Rendering {name}...")
        frames_p = []
        for i, black_bg in enumerate(black_bgs):
            t = i / N
            framed = stamp_rank_text(
                draw_panel_on_black(black_bg.copy(), t, light, dark, i),
                text_full,
                mask_full,
            )
            big = framed.resize((W, H), Image.Resampling.NEAREST)
            big.paste(text_big, mask=mask_big)
            q = big.convert(
                "P", palette=Image.Palette.ADAPTIVE, colors=192, dither=Image.Dither.NONE
            )
            q = force_palette_rank_colors(q)
            q = stamp_rank_colors_on_p(q, blue_big, white_big, red_big)
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
        print(f"  saved {path}")
        saved.append(path)
    print("done")

    if PREVIEW and saved:
        try:
            preview_gif(saved[0])
        except Exception as e:
            print(f"Preview skipped: {e}")


if __name__ == "__main__":
    main()
