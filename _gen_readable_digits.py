from PIL import Image, ImageDraw, ImageChops  # PIL 라이브러리에서 이미지 처리 관련 모듈 임포트
import math  # 수학 함수 사용을 위한 math 모듈 임포트
import os  # 시스템 경로 및 파일 관리를 위한 os 모듈 임포트

W, H = 1920, 1080  # 최종 GIF 해상도 (가로, 세로)
FPS = 12  # 초당 프레임 수
N = FPS * 5  # 총 프레임 수 (5초 분량)
FRAME_MS = int(1000 / FPS)  # 각 프레임의 지속 시간(ms)
PX_W, PX_H = 320, 180  # 패널의 내부 기본 해상도 (업스케일 전)

# 판넬이 16줄 모두 보기 좋게 더 높게 잡음
PAD_X, PAD_Y = 52, 8  # 패널 사방 여백
R = 6  # 모서리 라운드 반지름
BORDER_W = 3  # 테두리 두께
PX0, PY0 = PAD_X, PAD_Y  # 내부 패널의 시작점 좌표
PW = PX_W - 2 * PAD_X  # 내부 패널 가로길이
PH = PX_H - 2 * PAD_Y  # 내부 패널 세로길이

PANELS = [  # 여러 가지 패널 색상 테마 정의 (이름, 밝은색, 어두운색)
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
# 실행 시 적용할 패널 색 기본값 (문자열만 수정하면 색 바뀜)
# 사용가능: red orange gold lime green teal cyan blue pink slate / all
PANEL_COLOR = "green"  # 실행 시 기본 값: cyan
PREVIEW = True  # True면 이미지 생성 후 pygame으로 결과 미리보기
GRAYS = [  # 테두리/배경용 그레이 계열 그라데이션 컬러 모음
    (60, 62, 68),
    (110, 114, 122),
    (180, 184, 190),
    (230, 232, 236),
    (150, 154, 160),
    (90, 94, 100),
]
BALL_COLORS = [  # ball 애니메이션에 사용되는 볼 색상 쌍들 (밝은색, 어두운색)
    ((255, 70, 70), (190, 30, 30)),
    ((255, 150, 50), (200, 90, 20)),
    ((255, 220, 60), (200, 160, 25)),
    ((80, 210, 90), (40, 140, 50)),
    ((70, 140, 255), (35, 80, 190)),
    ((180, 80, 220), (120, 40, 160)),
    ((245, 248, 255), (175, 185, 205)),
]

RANK_COUNT = 16  # 랭킹 줄 수
# 랭킹별 번호 색깔 정의: 1–4등은 파랑, 5–13등은 흰색, 14–16등은 빨강
COLOR_BLUE = (50, 130, 255)  # 파란 색 (1~4등 랭킹 텍스트)
COLOR_WHITE = (255, 255, 255)  # 흰색 (5~13등)
COLOR_RED = (255, 55, 55)  # 빨강 (14~16등)
NUM_COL_W = 22  # 랭킹 번째 숫자 들어갈 좌측 칸 폭
DIGIT_W, DIGIT_H = 6, 7  # 랭킹 숫자 한 자릿수에 해당하는 픽셀 폭, 높이 (6x7)
DIGIT_GAP = 1  # 자리간 픽셀 간격
TEXT_NUDGE_X = 2  # 숫자들이 칸 내부에서 조금 왼쪽으로 이동할지 픽셀 단위 보정(+)면 왼쪽 이동

def rank_color(rank):
    """1부터 시작하는 랭킹 숫자를 받아 해당하는 텍스트 컬러 반환"""
    if rank <= 4:
        return COLOR_BLUE  # 상위 4개는 파란색
    if rank <= 13:
        return COLOR_WHITE  # 5~13등은 흰색
    return COLOR_RED  # 14~16등은 빨간색

# 6x7 크기의 각 숫자 bitmap (가독성을 위해 굵게 만듦)
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
    # v 값이 [a, b] 범위를 벗어나지 않도록 고정 (클램프)
    return max(a, min(b, v))

def mix(c1, c2, t):
    # 색상 c1과 c2를 t(0~1)만큼 선형 보간해서 반환
    t = clamp(t)
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def hsv(h, s, v):
    # HSV 색상값을 RGB 튜플로 변환
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
    return (int(r * 255), int(g * 255), int(b * 255))

def gray_at(u):
    # GRAYS 배열 기반에서 u 위치에 해당하는 색상 리턴 (그라데이션)
    n = len(GRAYS)
    x = (u % 1.0) * n
    i = int(x) % n
    f = x - int(x)
    return mix(GRAYS[i], GRAYS[(i + 1) % n], f)

def rounded_mask(w, h, r):
    # w*h 이미지에 반지름 r의 라운드 사각형 마스크(L) 반환
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=255)
    return m

OUTER_M = rounded_mask(PW, PH, R)  # 패널 바깥 마스크
INNER_M = Image.new("L", (PW, PH), 0)  # 패널 안쪽 마스크
ImageDraw.Draw(INNER_M).rounded_rectangle(
    [BORDER_W, BORDER_W, PW - 1 - BORDER_W, PH - 1 - BORDER_W],
    radius=max(1, R - BORDER_W),
    fill=255,
)
BORDER_M = ImageChops.subtract(OUTER_M, INNER_M)  # 테두리만 남긴 마스크
border_meta = []  # 테두리 픽셀 위치와 각도 목록
cx, cy = (PW - 1) / 2.0, (PH - 1) / 2.0  # 패널 중심좌표
bm = BORDER_M.load()
for ly in range(PH):
    for lx in range(PW):
        if bm[lx, ly] > 0:
            ang = (math.atan2(ly - cy, lx - cx) + math.pi) / (2 * math.pi)
            border_meta.append((lx, ly, ang))

def blit_digit(mask, ch, x, y):
    # 지정한 숫자 문자(ch)를 (x,y)에 6x7 배치 (mask는 PIL L모드 이미지)
    pattern = DIGIT[ch]
    pix = mask.load()
    for dy, row in enumerate(pattern):
        for dx, bit in enumerate(row):
            if bit == "1":
                xx, yy = x + dx, y + dy
                if 0 <= xx < PW and 0 <= yy < PH:
                    pix[xx, yy] = 255

def build_rank_text_layers(_frame_i=0):
    """
    랭킹 숫자(1~16)를 패널 위에 픽셀 단위로 렌더한 각종 마스크 레이어 생성.
    반환: (모든 숫자 RGB, 전체마스크, 파랑만, 흰색만, 빨강만)
    """
    mask = Image.new("L", (PW, PH), 0)  # 전체 숫자 마스크
    blue_m = Image.new("L", (PW, PH), 0)  # 파랑 랭킹 마스크
    white_m = Image.new("L", (PW, PH), 0)  # 흰색 랭킹 마스크
    red_m = Image.new("L", (PW, PH), 0)  # 빨강 랭킹 마스크
    left = BORDER_W + 3  # 왼쪽 시작점
    top = BORDER_W + 2  # 위쪽 시작점
    bottom = PH - BORDER_W - 2  # 아래쪽 끝점
    row_h = (bottom - top) / RANK_COUNT  # 한 줄 높이

    for i in range(RANK_COUNT):  # 1 ~ 16등 반복
        rank = i + 1  # 랭킹(1부터)
        label = str(rank)  # 숫자 문자열
        cy = top + (i + 0.5) * row_h  # 해당 랭킹 숫자 y여백 중앙값
        total_w = len(label) * DIGIT_W + (len(label) - 1) * DIGIT_GAP  # 전체 폭
        # 숫자 오른쪽 정렬(1~9, 10~16 너비 차이 맞춤)
        tx = left + NUM_COL_W - total_w - 1 - TEXT_NUDGE_X
        ty = int(cy - DIGIT_H / 2)
        ty = max(int(top + i * row_h) + 1, min(ty, int(top + (i + 1) * row_h) - DIGIT_H - 1))
        row_mask = Image.new("L", (PW, PH), 0)
        x = tx
        for ch in label:  # 숫자 문자열의 각 글자(1~2글자)
            blit_digit(row_mask, ch, x, ty)
            x += DIGIT_W + DIGIT_GAP
        row_mask = ImageChops.multiply(row_mask, INNER_M)  # 마스크로 내부 영역만 남김
        mask = ImageChops.lighter(mask, row_mask)  # 전체 마스크에 병합
        col = rank_color(rank)
        if col == COLOR_BLUE:
            blue_m = ImageChops.lighter(blue_m, row_mask)
        elif col == COLOR_RED:
            red_m = ImageChops.lighter(red_m, row_mask)
        else:
            white_m = ImageChops.lighter(white_m, row_mask)
    # 파랑/흰색/빨강 각각 마스크 영역에 해당 색으로 RGB 레이어 채움
    rgb = Image.new("RGB", (PW, PH), (0, 0, 0))
    rgb.paste(Image.new("RGB", (PW, PH), COLOR_BLUE), mask=blue_m)
    rgb.paste(Image.new("RGB", (PW, PH), COLOR_WHITE), mask=white_m)
    rgb.paste(Image.new("RGB", (PW, PH), COLOR_RED), mask=red_m)
    return rgb, mask, blue_m, white_m, red_m

def draw_rank_chrome(panel_img, frame_i):
    # 패널에 숫자 부분 뒤 어두운 밴드+랜더링 시 선택 표시+구분선 등 그리기
    draw = ImageDraw.Draw(panel_img)
    left = BORDER_W + 2  # 좌측 여백
    right = PW - BORDER_W - 3  # 우측 끝점
    top = BORDER_W + 2  # 상단 여백
    bottom = PH - BORDER_W - 2  # 하단 여백
    row_h = (bottom - top) / RANK_COUNT  # 한 줄 높이
    line_x0 = left + NUM_COL_W  # 숫자 열 끝 부분 (구분선 시작)

    # 숫자 열 뒤 어두운 밴드(명암 차 증가용)
    draw.rectangle([left, top, line_x0 - 1, bottom], fill=(0, 0, 0, 160))

    # 각 행별로 선택 영역(애니메이션) 표시
    sel = int(frame_i * RANK_COUNT / N) % RANK_COUNT  # 현재 선택 행
    y0 = int(top + sel * row_h)
    y1 = int(top + (sel + 1) * row_h)
    draw.rectangle([left, y0, right, max(y0 + 1, y1 - 1)], fill=(0, 0, 0, 255))

    # 숫자 우측 영역에만 구분선 그림(행별 탐색)
    for i in range(RANK_COUNT + 1):
        yy = int(top + i * row_h)
        draw.line([(line_x0, yy), (right, yy)], fill=(0, 0, 0, 255), width=1)

def gravity_bounce(u):
    # 0~1을 받아 포물선 바운스 모션 반환(가로축 균일)
    return (4.0 * u * (1.0 - u)) ** 0.92

def make_balls():
    # 볼(balls)들의 파라미터(위치/진폭/위상/크기 및 색상) 사전 리스트 생성
    ground = PX_H - 18  # 바닥 위치
    xs = [36, 80, 124, 168, 212, 256, 290]  # x축 중간값들
    amps = [46, 40, 52, 44, 48, 38, 50]  # y축 진폭
    cycles = [3, 3, 3, 4, 3, 3, 4]  # 애니메이션 내 반복 횟수(주기)
    phases = [0.00, 0.12, 0.28, 0.40, 0.55, 0.70, 0.85]  # 위상차
    rs = [1.9, 1.7, 2.0, 1.8, 1.85, 1.75, 1.95]  # 반지름
    x_amps = [4, 5, 3, 5, 4, 5, 3]  # x축 흔들림
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
    # 주어진 ball(b)과 프레임 인덱스에서 현재 위치와 상태 계산
    u = (frame_i / N * b["cycles"] + b["phase"]) % 1.0  # 주기 내 위치
    h = gravity_bounce(u)  # 높이(바운스)
    if h < 0.16:
        k = h / 0.16
        squash = 0.55 + 0.45 * k
        stretch = 1.35 - 0.35 * k
    elif h > 0.82:
        squash, stretch = 1.1, 0.9
    else:
        squash, stretch = 1.0, 1.0
    y = b["ground"] - b["r"] * squash - h * b["amp"]  # y좌표(점프/스쿼시)
    xu = (frame_i / N * b["x_cycles"] + b["phase"] * 0.5) % 1.0  # x축 진동 위치
    x = b["x"] + math.sin(xu * math.pi * 2) * b["x_amp"]  # x좌표
    return x, y, b["r"] * stretch, b["r"] * squash, h, b["light"], b["dark"]

def draw_ball(draw, x, y, rx, ry, h, light, dark):
    # 애니메이션 프레임 안에서 하나의 공을 그린다
    hi = mix(light, (255, 255, 255), 0.45)  # highlight 컬러(광택)
    draw.ellipse([x - rx, y - ry, x + rx, y + ry], fill=light)  # 메인 라이트 색 타원
    draw.ellipse([x - rx + 0.5, y, x + rx - 0.5, y + ry], fill=dark)  # 그림자 타원
    draw.ellipse(
        [x - rx * 0.55, y - ry * 0.7, x - rx * 0.05, y - ry * 0.12], fill=hi  # 광택 타원
    )
    sw = rx * (1.25 - 0.6 * h)  # 하이라이트 길이(높이 비례)
    sh = max(1, 1.2 + (1 - h))  # 하이라이트 두께
    gy = y + ry + 2  # 바닥 하이라이트 y 위치
    draw.ellipse([x - sw, gy - sh * 0.3, x + sw, gy + sh], fill=(18, 18, 22))  # 바닥 그림자

def shader_bg(t):
    # 패널용 동적 배경 그리기 함수 (시간 t)
    img = Image.new("RGB", (PX_W, PX_H), (0, 0, 0))
    pix = img.load()
    for y in range(PX_H):
        for x in range(PX_W):
            nx, ny = x / PX_W, y / PX_H  # 정규화 좌표
            cell = 10  # 배경 그리드 기본 셀 크기
            cx = (x % cell) / cell
            cy = (y % cell) / cell
            diag = abs(cx - 0.5) + abs(cy - 0.5)  # 중심과 거리
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
            vig = 1 - clamp((dx * dx + dy * dy) * 1.2, 0, 0.5)  # 비네팅(코너 어둡게)
            val *= vig
            pix[x, y] = (0, 0, 0) if val < 0.02 else hsv(hh % 1.0, 0.7, clamp(val))
    return img

def draw_panel_without_text(base_img, t, light, dark, frame_i):
    # 텍스트 없이 순수 배경/크롬/테두리만 합성해서 반환
    layer = Image.new("RGBA", (PX_W, PX_H), (0, 0, 0, 0))
    sh = Image.new("RGBA", (PW, PH), (0, 0, 0, 70))  # 그림자(투명)
    sh.putalpha(OUTER_M)
    layer.paste(sh, (PX0 + 2, PY0 + 3), sh)  # 그림자 위치
    im = INNER_M.load()
    shaded = Image.new("RGBA", (PW, PH), (0, 0, 0, 0))  # 패널 컬러 음영
    sp = shaded.load()
    for ly in range(PH):
        ny = ly / max(1, PH - 1)
        col = mix(light, dark, ny * 0.7)
        for lx in range(PW):
            if im[lx, ly] > 0:
                sp[lx, ly] = (*col, 235)
    layer.paste(shaded, (PX0, PY0), shaded)  # 내부 패널 붙이기
    panel_local = Image.new("RGBA", (PW, PH), (0, 0, 0, 0))
    draw_rank_chrome(panel_local, frame_i)
    r, g, b, a = panel_local.split()
    a = ImageChops.multiply(a, INNER_M)
    panel_local = Image.merge("RGBA", (r, g, b, a))
    layer.paste(panel_local, (PX0, PY0), panel_local)  # 크롬/밴드 오버레이
    border = Image.new("RGBA", (PW, PH), (0, 0, 0, 0))  # 테두리 성공/실패 효과
    bp = border.load()
    for lx, ly, ang in border_meta:
        gcol = gray_at(ang + t)
        shimmer = 0.85 + 0.15 * math.sin((ang * 4 + t * 2) * math.pi * 2)
        gcol = tuple(int(clamp(c * shimmer, 0, 255)) for c in gcol)
        bp[lx, ly] = (*gcol, 255)
    layer.paste(border, (PX0, PY0), border)
    return Image.alpha_composite(base_img.convert("RGBA"), layer).convert("RGB")

def make_base_frame(i, ball_hist):
    # ball 애니메이션까지 포함된 패널용 배경 1프레임 생성
    t = i / N  # 0~1 구간 내 현재 프레임
    img = shader_bg(t)
    pix = img.load()
    draw = ImageDraw.Draw(img)
    bar = 8  # 상하단 강조 라인 두께
    accent = (40, 70, 90)  # 강조 색상
    for y in range(bar):  # 상단/하단 그라데이션
        a = 0.22 * (1 - y / bar)
        for x in range(PX_W):
            pix[x, y] = mix(pix[x, y], accent, a)
            pix[x, PX_H - 1 - y] = mix(pix[x, PX_H - 1 - y], accent, a)
    blink = 0.55 + 0.35 * (0.5 + 0.5 * math.sin(t * math.pi * 2))
    line = mix((50, 120, 140), (120, 230, 245), blink)
    for x in range(PX_W):  # 상하단 라인 강조
        pix[x, bar] = mix(pix[x, bar], line, 0.5)
        pix[x, PX_H - 1 - bar] = mix(pix[x, PX_H - 1 - bar], line, 0.5)
    WHITE = (220, 230, 240)  # 모서리 강조 흰색
    m, L = 5, 14  # 모서리 크기 및 선 길이
    for ox, oy, sx, sy in [  # 네 모서리 라인 출력
        (m, m, 1, 1),
        (PX_W - m - 1, m, -1, 1),
        (m, PX_H - m - 1, 1, -1),
        (PX_W - m - 1, PX_H - m - 1, -1, -1),
    ]:
        for j in range(L):
            pix[ox + j * sx, oy] = WHITE
            pix[ox, oy + j * sy] = WHITE
    for x, y, rx, ry, h, light, dark in sorted(ball_hist[i], key=lambda p: p[1]):
        draw_ball(draw, x, y, rx, ry, h, light, dark)  # 볼 여러 개 프레임마다 그림
    return img, t

def stamp_rank_text(rgb_img, text_rgb_full, mask_full):
    # rgb_img에 text_rgb_full을 마스크(mask_full) 기준으로 붙임
    rgb_img.paste(text_rgb_full, mask=mask_full)
    return rgb_img

def force_palette_rank_colors(img_p):
    """팔레트 컬러 253/254/255번을 각각 blue, red, white로 고정."""
    pal = img_p.getpalette()
    if not pal:
        return img_p
    pal = list(pal) + [0] * (768 - len(pal))
    for i in range(256):
        r, g, b = pal[i * 3], pal[i * 3 + 1], pal[i * 3 + 2]
        if r >= 250 and g >= 250 and b >= 250:
            pal[i * 3 : i * 3 + 3] = list(COLOR_WHITE)
    # 고정 슬롯에 강제 셋팅
    pal[253 * 3 : 253 * 3 + 3] = list(COLOR_BLUE)
    pal[254 * 3 : 254 * 3 + 3] = list(COLOR_RED)
    pal[255 * 3 : 255 * 3 + 3] = list(COLOR_WHITE)
    img_p.putpalette(pal[:768])
    return img_p

def stamp_rank_colors_on_p(img_p, blue_big, white_big, red_big):
    # 랭킹 테마의 파랑/빨강/흰색 마스크를 팔레트 인덱스에 할당 (P모드)
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
    """주어진 color_key(str)로부터 렌더링 패널들(name, light, dark) 리스트 반환."""
    key = (color_key or "").strip().lower()
    if key in ("", "all", "*"):
        return list(PANELS)
    matched = [p for p in PANELS if p[0].split("_", 1)[-1] == key or p[0] == key]
    if not matched:
        names = ", ".join(p[0].split("_", 1)[-1] for p in PANELS)
        raise SystemExit(f"Unknown PANEL_COLOR={color_key!r}. Use one of: {names}, all")
    return matched

def preview_gif(path):
    """pygame으로 GIF 루프 재생, ESC/창닫기로 종료."""
    import pygame

    path = os.fspath(path)
    frames = []  # 프레임 리스트
    durations = []  # 각 프레임별 지속 시간 리스트
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
    scale = min(1.0, info.current_w / w * 0.9, info.current_h / h * 0.9)  # 화면 해상도 비율 조정
    vw, vh = max(1, int(w * scale)), max(1, int(h * scale))  # 실제 출력할 윈도우 크기
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
    # 아웃풋 디렉토리 생성(바탕화면에 저장됨)
    out_dir = os.path.join(os.path.expanduser("~"), "Desktop", "ui_panel_gifs")
    os.makedirs(out_dir, exist_ok=True)
    panels = resolve_panels(PANEL_COLOR)  # 선택 패널 리스트
    print(f"Panel {PW}x{PH} | PANEL_COLOR={PANEL_COLOR!r} → {[p[0] for p in panels]}")
    balls = make_balls()  # 볼 파라미터 만들기
    ball_hist = [[ball_at(b, i) for b in balls] for i in range(N)]  # 프레임별 볼의 정보 미리 계산
    bases = [make_base_frame(i, ball_hist) for i in range(N)]  # 프레임마다 배경+볼 애니 준비
    print("Building digit layers...")
    text_rgb, mask, blue_m, white_m, red_m = build_rank_text_layers(0)  # 랭킹 숫자 마스크 미리 생성

    def to_full(layer):
        # 320x180 안쪽패널을 전체 패널(PX_W, PX_H) 안에 위치시키기 위함
        full = Image.new(layer.mode, (PX_W, PX_H), 0 if layer.mode == "L" else (0, 0, 0))
        full.paste(layer, (PX0, PY0))
        return full

    text_full = to_full(text_rgb)
    mask_full = to_full(mask)
    blue_full = to_full(blue_m)
    white_full = to_full(white_m)
    red_full = to_full(red_m)

    text_big = text_full.resize((W, H), Image.Resampling.NEAREST)  # 최종 해상도용 렌더
    mask_big = mask_full.resize((W, H), Image.Resampling.NEAREST)
    blue_big = blue_full.resize((W, H), Image.Resampling.NEAREST)
    white_big = white_full.resize((W, H), Image.Resampling.NEAREST)
    red_big = red_full.resize((W, H), Image.Resampling.NEAREST)

    saved = []  # 만들어진 GIF 경로 저장
    for name, light, dark in panels:  # 패널/투명도별로 반복
        print(f"Rendering {name}...")
        frames_p = []
        for i, (img, t) in enumerate(bases):
            framed = stamp_rank_text(
                draw_panel_without_text(img.copy(), t, light, dark, i),
                text_full,
                mask_full,
            )
            big = framed.resize((W, H), Image.Resampling.NEAREST)  # 최종 스케일업
            big.paste(text_big, mask=mask_big)  # 랭킹 숫자 업스케일 후 직접 스탬프
            q = big.convert(
                "P", palette=Image.Palette.ADAPTIVE, colors=192, dither=Image.Dither.NONE
            )  # 팔레트 모드 변환(최대 192색)
            q = force_palette_rank_colors(q)  # 랭킹색 팔레트 강제
            q = stamp_rank_colors_on_p(q, blue_big, white_big, red_big)  # 랭킹색 영역 팔레트 인덱스 강제
            frames_p.append(q)
        path = os.path.join(out_dir, f"game_ui_panel_{name}.gif")
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
        # 한 색만 지정했을 때 그 화면을 바로 보여줌 (all이면 첫 번째)
        preview_gif(saved[0])

if __name__ == "__main__":
    main()  # main()만 실행 (직접 실행시)
