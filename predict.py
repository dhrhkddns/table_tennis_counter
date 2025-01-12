import cv2
import numpy as np
import time
import ctypes
from ultralytics import YOLO
import pygame
from PIL import Image, ImageDraw, ImageFont

user32 = ctypes.windll.user32

# ----------------------------------------------------------------------------------------
# 1) pygame 오디오 초기화 및 사운드 로드
# ----------------------------------------------------------------------------------------
pygame.mixer.init()
sound = pygame.mixer.Sound(r"C:\Users\omyra\Desktop\coding\ping_pong\retro-coin-4-236671.mp3")

# ----------------------------------------------------------------------------------------
# 2) YOLO 모델 로드
# ----------------------------------------------------------------------------------------
model = YOLO(r"C:\Users\omyra\Desktop\coding\ping_pong\Ping-Pong-Detection-3\Results\weights\best.pt")
model.to("cuda")

# ----------------------------------------------------------------------------------------
# 3) 카메라 디바이스 연결
# ----------------------------------------------------------------------------------------
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 35)

# ----------------------------------------------------------------------------------------
# 4) 그래프, 바운스 관련 전역 변수 정의
# ----------------------------------------------------------------------------------------
x_values = []                    # 공의 x좌표 기록을 저장하는 리스트 (예: [100, 105, 110, ...])
y_values = []                    # 공의 y좌표 기록을 저장하는 리스트 (예: [200, 195, 190, ...]) 
orange_pixel_values = []         # 프레임별 오렌지색 픽셀 수를 저장하는 리스트 (예: [150, 148, 152, ...])
frame_count = 0                  # 현재까지 처리된 프레임 수를 카운트 (예: 1, 2, 3, ...)
MAX_POINTS = 100                 # 그래프에 표시할 최대 데이터 포인트 수 (예: 최근 100개 프레임만 표시)

bounce_count = 0                 # 공이 바운스한 총 횟수 (예: 0에서 시작해서 바운스할 때마다 1씩 증가)

consecutiveDownCount = 0         # 연속으로 아래로 움직인 프레임 수 (예: 3프레임 연속 하강 시 3)
consecutiveUpCount = 0           # 연속으로 위로 움직인 프레임 수 (예: 2프레임 연속 상승 시 2)
state = None                     # 현재 공의 이동 상태 ('up' 또는 'down' 또는 None)
DOWN_THRESHOLD = 2               # 바운스 감지를 위한 최소 하강 프레임 수 (예: 2프레임 이상 연속 하강)
UP_THRESHOLD = 1                 # 바운스 감지를 위한 최소 상승 프레임 수 (예: 1프레임 이상 연속 상승)
PIXEL_THRESHOLD = 3.0           # 움직임 감지를 위한 최소 픽셀 변화량 (예: y좌표가 3픽셀 이상 변할 때)
last_y = None                    # 이전 프레임의 y좌표 값 (예: 200)

bounce_points = []              # 바운스가 발생한 지점의 좌표 리스트 (예: [(100,200), (150,200), ...])
bounce_times = []               # 각 바운스가 발생한 시간 리스트 (예: [1.23, 2.45, 3.67, ...])

CONTINUOUS_TIMEOUT = 1.0        # 연속된 바운스 간의 최소 시간 간격(초) (바운스가 너무 멀어지면 떨어진걸로 인식)
last_bounce_time = None         # 마지막 바운스가 감지된 시간 (예: 1234567.89)

sound_enabled = False           # 바운스 시 소리 재생 여부 (True: 소리 켬, False: 소리 끔)
ignore_zero_orange = False      # 오렌지색 픽셀이 0일 때 무시할지 여부 (True: 무시, False: 처리)

button_rect = [500, 20, 120, 40]         # 소리 켜기/끄기 버튼의 위치와 크기 [x, y, width, height]
button_rect_ignore = [500, 70, 120, 40]  # 오렌지픽셀 무시 설정 버튼의 위치와 크기 [x, y, width, height]

FONT_PATH = r"C:\Users\omyra\Desktop\coding\ping_pong\Digital Display.ttf"  # 디지털 폰트 파일 경로
FONT_SIZE = 400                 # 폰트 크기 (픽셀 단위)
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)  # PIL 폰트 객체 생성 (바운스 카운트 표시용)

# BGR 형식의 색상 시퀀스
color_sequence = [
    (255, 255, 255),  # 흰색
    (0, 0, 255),      # 빨간색 
    (0, 165, 255),    # 주황색
    (0, 255, 255),    # 노란색
    (144, 238, 144),  # 연한 초록색
    (0, 255, 0),      # 초록색
    (255, 255, 0),    # 하늘색
    (255, 0, 0),      # 파란색
    (128, 0, 255),    # 분홍색
    (255, 0, 255)     # 보라색
]
intensity_levels = [0.5 + 0.05 * i for i in range(10)]  # 각 색상의 밝기 레벨 (예: [0.5, 0.55, 0.6, ..., 0.95])

def get_color(count):
    if count >= 1000:  # 바운스 카운트가 1000 이상이면 보라색 반환 (예: count=1234 -> (255,0,255))
        return (255, 0, 255)

    color_index = count // 100  # 100단위로 기본 색상 결정 (예: count=234 -> index=2)
    if color_index >= len(color_sequence):  # 색상 시퀀스 범위 초과 시 마지막 색상 사용
        color_index = len(color_sequence) - 1  # (예: color_index=11 -> 9로 조정)

    step_in_block = (count % 100) // 10  # 각 색상 내에서 10단위로 "밝기 단계" 결정 (예: count=234 -> step=3 밝기 단계 step 0...9)
    intensity = intensity_levels[step_in_block] if step_in_block < len(intensity_levels) else 1.0  # 밝기 레벨 선택 (예: step=3 -> 0.65)
    base_color = color_sequence[color_index]  # 기본 색상 선택 (예: (0,255,0))

    color_bgr = np.uint8([[base_color]])  # BGR 색상을 numpy 배열로 변환 (예: [[[0,255,0]]])
    color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]  # BGR을 HSV로 변환 (예: [60,255,255])

    color_hsv = color_hsv.astype(float)  # HSV 값을 실수형으로 변환하여 연산 가능하게 함
    color_hsv[2] = min(color_hsv[2] * intensity, 255)  # Value(밝기) 값 조정 (예: 255 * 0.65 = 165.75)
    color_hsv = color_hsv.astype(np.uint8)  # 다시 정수형으로 변환

    intense_color = cv2.cvtColor(np.uint8([[color_hsv]]), cv2.COLOR_HSV2BGR)[0][0]  # HSV를 BGR로 다시 변환
    intense_color_rgb = (intense_color[2], intense_color[1], intense_color[0])  # BGR을 RGB로 변환 (예: (0,165,0))
    return intense_color_rgb  # 최종 RGB 색상 반환

# =============================================================================
# 드래그/리사이즈 가능한 빨간 사각형 관련 전역 변수
# =============================================================================
drag_rect_x, drag_rect_y = 100, 100  # 사각형 왼상단 초기 위치
drag_rect_w, drag_rect_h = 150, 150  # 사각형 폭, 높이
dragging = False                     # 현재 드래그(이동) 중인지 여부
resizing_corner = None               # 현재 리사이즈 중인 corner (None, 'tl', 'tr', 'bl', 'br')
drag_offset_x, drag_offset_y = 0, 0  # (이동용) 드래그 시작점 대비 사각형 내부 오프셋
corner_size = 10                     # 각 모서리 핸들의 반지름(또는 반폭)

# =============================================================================
# 사각형 내부에서 공이 감지된 시간을 실시간으로 표시하기 위한 변수
# =============================================================================
ball_in_rect_start = None   # 사각형 안에 공이 들어온 시점(초)
in_rect_time = 0.0          # 사각형 안에 있는 동안의 시간(실시간 업데이트)

# ----------------------------------------------------------------------------------------
# (A) 우클릭 확대/복귀 기능 관련 전역 변수
# ----------------------------------------------------------------------------------------
enlarged_view = None  # 'tl', 'tr', 'bl', 'br' or None (기본값: None=4분할)

# ----------------------------------------------------------------------------------------
# 9) mouse_callback 함수
# ----------------------------------------------------------------------------------------
last_mouse_move_time = time.time()
mouse_visible = True

def mouse_callback(event, x, y, flags, param):
    global sound_enabled, ignore_zero_orange
    global last_mouse_move_time, mouse_visible
    global dragging, drag_offset_x, drag_offset_y
    global drag_rect_x, drag_rect_y, drag_rect_w, drag_rect_h
    global resizing_corner
    global enlarged_view  # (A) 우클릭 확대/복귀

    if event == cv2.EVENT_MOUSEMOVE:
        last_mouse_move_time = time.time()
        if not mouse_visible:
            user32.ShowCursor(True)
            mouse_visible = True

        # 리사이즈 중이면 각 코너별로 크기 갱신
        if resizing_corner is not None:
            if resizing_corner == 'tl':
                new_w = drag_rect_w + (drag_rect_x - x)
                new_h = drag_rect_h + (drag_rect_y - y)
                new_x = x
                new_y = y
                if new_w < 10:
                    new_w = 10
                    new_x = drag_rect_x + drag_rect_w - 10
                if new_h < 10:
                    new_h = 10
                    new_y = drag_rect_y + drag_rect_h - 10
                new_x = max(0, min(new_x, drag_rect_x + drag_rect_w))
                new_y = max(0, min(new_y, drag_rect_y + drag_rect_h))

                if new_x < 0: new_x = 0
                if new_y < 0: new_y = 0
                if new_x > 640: new_x = 640
                if new_y > 480: new_y = 480

                drag_rect_w = new_w
                drag_rect_h = new_h
                drag_rect_x = new_x
                drag_rect_y = new_y

            elif resizing_corner == 'tr':
                new_w = x - drag_rect_x
                new_h = drag_rect_h + (drag_rect_y - y)
                new_y = y
                if new_w < 10:
                    new_w = 10
                if new_h < 10:
                    new_h = 10
                    new_y = drag_rect_y + drag_rect_h - 10
                if new_w > 640 - drag_rect_x:
                    new_w = 640 - drag_rect_x
                if new_y < 0:
                    new_y = 0

                drag_rect_w = new_w
                drag_rect_h = new_h
                drag_rect_y = new_y

            elif resizing_corner == 'bl':
                new_w = drag_rect_w + (drag_rect_x - x)
                new_h = y - drag_rect_y
                new_x = x
                if new_w < 10:
                    new_w = 10
                    new_x = drag_rect_x + drag_rect_w - 10
                if new_h < 10:
                    new_h = 10
                if new_x < 0:
                    new_x = 0
                if new_h > 480 - drag_rect_y:
                    new_h = 480 - drag_rect_y

                drag_rect_w = new_w
                drag_rect_h = new_h
                drag_rect_x = new_x

            elif resizing_corner == 'br':
                new_w = x - drag_rect_x
                new_h = y - drag_rect_y
                if new_w < 10:
                    new_w = 10
                if new_h < 10:
                    new_h = 10
                if new_w > 640 - drag_rect_x:
                    new_w = 640 - drag_rect_x
                if new_h > 480 - drag_rect_y:
                    new_h = 480 - drag_rect_y

                drag_rect_w = new_w
                drag_rect_h = new_h

        elif dragging:
            new_x = x - drag_offset_x
            new_y = y - drag_offset_y
            new_x = max(0, min(new_x, 640 - drag_rect_w))
            new_y = max(0, min(new_y, 480 - drag_rect_h))
            drag_rect_x, drag_rect_y = new_x, new_y

    elif event == cv2.EVENT_LBUTTONDOWN:
        # 사운드 ON/OFF 버튼
        if (button_rect[0] <= x - 640 <= button_rect[0] + button_rect[2] and
            button_rect[1] <= y <= button_rect[1] + button_rect[3]):
            sound_enabled = not sound_enabled
            print(f"Sound Enabled: {sound_enabled}")
        
        # Ignore Zero Orange 버튼
        elif (button_rect_ignore[0] <= x - 640 <= button_rect_ignore[0] + button_rect_ignore[2] and
              button_rect_ignore[1] <= y <= button_rect_ignore[1] + button_rect_ignore[3]):
            ignore_zero_orange = not ignore_zero_orange
            print(f"Ignore Zero Orange Pixels: {ignore_zero_orange}")
        
        else:
            corners = {
                'tl': (drag_rect_x, drag_rect_y),
                'tr': (drag_rect_x + drag_rect_w, drag_rect_y),
                'bl': (drag_rect_x, drag_rect_y + drag_rect_h),
                'br': (drag_rect_x + drag_rect_w, drag_rect_y + drag_rect_h)
            }
            corner_clicked = None
            for ckey, cpos in corners.items():
                cx, cy = cpos
                if (cx - corner_size <= x <= cx + corner_size and 
                    cy - corner_size <= y <= cy + corner_size):
                    corner_clicked = ckey
                    break

            if corner_clicked:
                resizing_corner = corner_clicked
            else:
                # 사각형 내부라면 드래그(이동) 시작
                if (drag_rect_x <= x < drag_rect_x + drag_rect_w and
                    drag_rect_y <= y < drag_rect_y + drag_rect_h):
                    dragging = True
                    drag_offset_x = x - drag_rect_x
                    drag_offset_y = y - drag_rect_y

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        resizing_corner = None

    # (A) 우클릭 시 해당 쿼드런트만 확대 or 복귀
    elif event == cv2.EVENT_RBUTTONDOWN:
        if enlarged_view is None:
            # 4개 쿼드런트 범위:
            # top-left:    y in [0,480), x in [0,640)
            # top-right:   y in [0,480), x in [640,1280)
            # bottom-left: y in [480,960), x in [0,640)
            # bottom-right:y in [480,960), x in [640,1280)

            if 0 <= y < 480 and 0 <= x < 640:
                enlarged_view = 'tl'
            elif 0 <= y < 480 and 640 <= x < 1280:
                enlarged_view = 'tr'
            elif 480 <= y < 960 and 0 <= x < 640:
                enlarged_view = 'bl'
            elif 480 <= y < 960 and 640 <= x < 1280:
                enlarged_view = 'br'

            if enlarged_view is not None:
                print(f"Enlarged => {enlarged_view}")

        else:
            # 이미 확대된 상태라면 다시 None으로 (4분할)
            print(f"Return to 4-split from: {enlarged_view}")
            enlarged_view = None


# ----------------------------------------------------------------------------------------
# 10) render_text_with_ttf()
# ----------------------------------------------------------------------------------------
def render_text_with_ttf(
    text,
    font=font,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
    width=960,
    height=540
):
    img_pil = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img_pil)

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    text_x = (width - text_w) // 2
    text_y = (height - text_h) // 2
    draw.text((text_x, text_y), text, font=font, fill=text_color)

    img_np = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_bgr

# ----------------------------------------------------------------------------------------
# 11) y좌표 그래프 그리기 함수
# ----------------------------------------------------------------------------------------
def draw_y_graph(x_data, y_data, width=640, height=480, max_y=480, bounce_pts=None):
    if bounce_pts is None:
        bounce_pts = []

    graph_img = np.zeros((height, width, 3), dtype=np.uint8)
    if len(x_data) < 2:
        return graph_img

    max_x = x_data[-1] if x_data[-1] != 0 else 1

    for i in range(len(x_data) - 1):
        if y_data[i] is None or y_data[i+1] is None:
            continue
        x1_ori, y1_ori = x_data[i], y_data[i]
        x2_ori, y2_ori = x_data[i+1], y_data[i+1]

        x1 = int((x1_ori - x_data[0]) / (max_x - x_data[0] + 1e-6) * (width - 1))
        x2 = int((x2_ori - x_data[0]) / (max_x - x_data[0] + 1e-6) * (width - 1))

        y1 = int(y1_ori / max_y * (height - 1))
        y2 = int(y2_ori / max_y * (height - 1))

        cv2.line(graph_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for i in range(len(x_data)):
        if y_data[i] is None:
            continue
        x_ori, y_ori = x_data[i], y_data[i]
        x_pt = int((x_ori - x_data[0]) / (max_x - x_data[0] + 1e-6) * (width - 1))
        y_pt = int(y_ori / max_y * (height - 1))
        cv2.circle(graph_img, (x_pt, y_pt), 4, (255, 0, 0), -1)

    for (bx_ori, by_ori) in bounce_pts:
        if bx_ori < x_data[0]:
            continue
        bx = int((bx_ori - x_data[0]) / (max_x - x_data[0] + 1e-6) * (width - 1))
        by = int(by_ori / max_y * (height - 1))
        cv2.circle(graph_img, (bx, by), 5, (0, 0, 255), -1)

    # 사운드 ON/OFF 버튼
    cv2.rectangle(
        graph_img,
        (button_rect[0], button_rect[1]),
        (button_rect[0] + button_rect[2], button_rect[1] + button_rect[3]),
        (120, 120, 120),
        -1
    )
    text_sound = "Sound: ON" if sound_enabled else "Sound: OFF"
    cv2.putText(
        graph_img,
        text_sound,
        (button_rect[0] + 10, button_rect[1] + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Ignore0 버튼
    cv2.rectangle(
        graph_img,
        (button_rect_ignore[0], button_rect_ignore[1]),
        (button_rect_ignore[0] + button_rect_ignore[2], button_rect_ignore[1] + button_rect_ignore[3]),
        (120, 120, 120),
        -1
    )
    text_ignore = "Ignore0: ON" if ignore_zero_orange else "Ignore0: OFF"
    cv2.putText(
        graph_img,
        text_ignore,
        (button_rect_ignore[0] + 5, button_rect_ignore[1] + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    return graph_img

# ----------------------------------------------------------------------------------------
# 12) 오렌지 픽셀 그래프 그리기 함수
# ----------------------------------------------------------------------------------------
def draw_orange_graph(x_data, orange_data, width=640, height=480, max_y=None):
    if max_y is None:
        valid_orange_data = [v for v in orange_data if v is not None]
        max_y = max(valid_orange_data) if valid_orange_data else 1

    graph_img = np.zeros((height, width, 3), dtype=np.uint8)
    if len(x_data) < 2:
        return graph_img

    max_x = x_data[-1] if x_data[-1] != 0 else 1

    for i in range(len(x_data) - 1):
        if orange_data[i] is None or orange_data[i+1] is None:
            continue
        x1_ori, y1_ori = x_data[i], orange_data[i]
        x2_ori, y2_ori = x_data[i+1], orange_data[i+1]

        x1 = int((x1_ori - x_data[0]) / (max_x - x_data[0] + 1e-6) * (width - 1))
        x2 = int((x2_ori - x_data[0]) / (max_x - x_data[0] + 1e-6) * (width - 1))

        y1 = int(y1_ori / max_y * (height - 1)) if max_y > 0 else 0
        y2 = int(y2_ori / max_y * (height - 1)) if max_y > 0 else 0

        cv2.line(graph_img, (x1, height - y1), (x2, height - y2), (0, 165, 255), 2)

    for i in range(len(x_data)):
        if orange_data[i] is None:
            continue
        x_ori, y_ori = x_data[i], orange_data[i]
        x_pt = int((x_ori - x_data[0]) / (max_x - x_data[0] + 1e-6) * (width - 1))
        y_pt = int(y_ori / max_y * (height - 1)) if max_y > 0 else 0
        cv2.circle(graph_img, (x_pt, height - y_pt), 4, (0, 165, 255), -1)
        cv2.putText(
            graph_img,
            f"{y_ori}",
            (x_pt + 5, height - y_pt - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            1,
            cv2.LINE_AA
        )

    cv2.line(graph_img, (0, height - 1), (width - 1, height - 1), (255, 255, 255), 1)
    cv2.line(graph_img, (0, 0), (0, height - 1), (255, 255, 255), 1)

    cv2.putText(
        graph_img,
        "Orange Pixel Count",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return graph_img

# ----------------------------------------------------------------------------------------
# 13) Combined, Bounce Count 창을 생성 & Combined 창을 전체화면으로 시작
# ----------------------------------------------------------------------------------------
cv2.namedWindow("Combined", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Combined", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
is_fullscreen_combined = True

cv2.namedWindow("Bounce Count Window", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Bounce Count Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
is_fullscreen_bounce = True

cv2.setMouseCallback("Combined", mouse_callback)

prev_bounce_count = None
bounce_img = None
is_fullscreen = False

# ----------------------------------------------------------------------------------------
# 14) 추가된 전역 변수: 상태 관리
# ----------------------------------------------------------------------------------------
current_state = "waiting"
state_display_text = "Waiting"
state_font = cv2.FONT_HERSHEY_SIMPLEX
state_font_scale = 1.0
state_font_color = (255, 255, 255)
state_font_thickness = 2
state_change_time = None

stationary_start_time = None
stationary_threshold = 2.0
movement_threshold = 5
last_position = None

previous_bounce_time = None

# ----------------------------------------------------------------------------------------
# 14-1) 추가: 마지막 검출 시각(공이 마지막으로 발견된 시간)을 저장할 변수
# ----------------------------------------------------------------------------------------
last_detection_time = None

# ========================================================================================
# 바운스 간 시간차 표시를 위한 전역 변수
# ========================================================================================
bounce_time_diff = None

# ========================================================================================
# (추가) 최근 TRACKING 종료 시점의 bounce_count를 시각화하기 위한 이력
# ========================================================================================
bounce_history = []   # 최대 8개 정도만 저장해서 사각형에 표시

# ----------------------------------------------------------------------------------------
# 15) 메인 루프
# ----------------------------------------------------------------------------------------
prev_time = time.time()
fps = 0.0

while True:
    now = time.time()
    if now - last_mouse_move_time > 3.0:
        if mouse_visible:
            user32.ShowCursor(False)
            mouse_visible = False

    ret, frame = cap.read()
    if not ret:
        print("No more frames or camera error.")
        break

    current_time = time.time()
    time_diff = current_time - prev_time
    if time_diff > 1e-9:
        fps = 1.0 / time_diff
    prev_time = current_time

    results = model.predict(frame, imgsz=640, conf=0.5, max_det=1, show=False, device=0)
    boxes = results[0].boxes

    x_values.append(frame_count)
    frame_count += 1

    detected = False
    orange_pixels = 0

    # ------------------------------------------------------------------------------------
    # 공 검출 여부 확인
    # ------------------------------------------------------------------------------------
    if len(boxes) > 0:
        x1, y1, x2, y2 = boxes[0].xyxy[0].cpu().numpy()
        y_center = (y1 + y2) / 2.0
        x_center = (x1 + x2) / 2.0  # 공의 x중심좌표

        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
        x1i = max(0, x1i)
        y1i = max(0, y1i)
        x2i = min(frame.shape[1], x2i)
        y2i = min(frame.shape[0], y2i)

        roi = frame[y1i:y2i, x1i:x2i]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([10, 100, 100], dtype=np.uint8)
        upper_orange = np.array([25, 255, 255], dtype=np.uint8)
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        orange_pixels = cv2.countNonZero(mask_orange)

        if ignore_zero_orange:
            if orange_pixels >= 5:
                detected = True
        else:
            detected = True

        if detected:
            last_detection_time = time.time()

            y_values.append(y_center)
            orange_pixel_values.append(orange_pixels)

            # --------------------------------------------------------------------------
            # 상태 전환(ready / tracking) 확인
            # --------------------------------------------------------------------------
            if last_position is not None:
                dy = y_center - last_position
                movement = abs(dy)
            else:
                movement = 0

            if current_state == "ready":
                if movement > movement_threshold:
                    current_state = "tracking"
                    bounce_count = 0
                    bounce_points = []
                    bounce_times = []
                    previous_bounce_time = None
                    print("State changed to TRACKING")

            if movement > movement_threshold:
                if stationary_start_time is not None:
                    stationary_start_time = None
            else:
                if stationary_start_time is None:
                    stationary_start_time = time.time()
                elif (time.time() - stationary_start_time) >= stationary_threshold:
                    # 공이 2초 이상 멈췄을 때
                    if in_rect_time >= 2.0 and current_state != "ready":
                        # 빨간 사각형 안에 공이 2초 이상 => ready
                        current_state = "ready"
                        state_change_time = time.time()
                        print("State changed to READY")

            last_position = y_center

            # --------------------------------------------------------------------------
            # 바운스 카운트 로직
            # --------------------------------------------------------------------------
            if current_state == "tracking":
                if last_y is not None:
                    dy_tracking = y_center - last_y
                    if abs(dy_tracking) > PIXEL_THRESHOLD:
                        if dy_tracking > 0:
                            consecutiveDownCount += 1
                            consecutiveUpCount = 0
                        else:
                            consecutiveUpCount += 1
                            consecutiveDownCount = 0

                        if state is None:
                            if consecutiveDownCount >= DOWN_THRESHOLD:
                                state = "down"
                        elif state == "down":
                            if consecutiveUpCount >= UP_THRESHOLD:
                                bounce_count += 1
                                print("Bounce detected!")
                                if sound_enabled:
                                    sound.play()

                                bounce_points.append((x_values[-1], y_values[-1]))
                                current_bounce_time = time.time()
                                bounce_times.append(current_bounce_time)

                                if previous_bounce_time is not None:
                                    td = current_bounce_time - previous_bounce_time
                                    print(f"Time diff between last two bounces: {td:.2f} s")
                                    bounce_time_diff = td
                                    # (추가) 만약 td > 1.0 이면 상태를 waiting으로 돌리고 싶다면 여기서 처리
                                    if td > 1.0:
                                        # waiting으로 전환하기 전에 bounce_history에 기록
                                        bounce_history.append(bounce_count)
                                        if len(bounce_history) > 8:
                                            bounce_history.pop(0)

                                        current_state = "waiting"
                                        # bounce_count도 0으로 리셋
                                        bounce_count = 0
                                        bounce_points = []
                                        bounce_times = []
                                        previous_bounce_time = None
                                        print("State changed to WAITING (timeout)")
                                previous_bounce_time = current_bounce_time

                                state = "up"
                                consecutiveDownCount = 0
                                consecutiveUpCount = 0
                        elif state == "up":
                            if consecutiveDownCount >= DOWN_THRESHOLD:
                                state = "down"
                                consecutiveUpCount = 0
                                consecutiveDownCount = 0

                last_y = y_center

            # --------------------------------------------------------------------------
            # 디버그용 사각형 & 텍스트
            # --------------------------------------------------------------------------
            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"y_center={int(y_center)}",
                (x1i, y1i - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                frame,
                f"Orange px: {orange_pixels}",
                (x1i, y2i + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
                cv2.LINE_AA
            )
        else:
            y_values.append(None)
            orange_pixel_values.append(None)
    else:
        y_values.append(None)
        orange_pixel_values.append(None)

    # -------------------------------------------------------------------------
    # (1) 공이 사각형 안에 있는 동안 => in_rect_time = 현재시간 - 진입시점
    # (2) 공이 나가면 => in_rect_time = 0
    # -------------------------------------------------------------------------
    if len(boxes) > 0 and detected:
        # 공 중심좌표 (x_center, y_center)가 사각형 내부인지 확인
        if (drag_rect_x <= x_center < drag_rect_x + drag_rect_w and
            drag_rect_y <= y_center < drag_rect_y + drag_rect_h):
            if ball_in_rect_start is None:
                ball_in_rect_start = time.time()
            in_rect_time = time.time() - ball_in_rect_start
        else:
            in_rect_time = 0.0
            ball_in_rect_start = None
    else:
        in_rect_time = 0.0
        ball_in_rect_start = None

    # -------------------------------------------------------------------------
    # state==ready 인 상태에서 공이 안보이는(=detected=False) 1초 경과 시 waiting으로
    # -------------------------------------------------------------------------
    if current_state == "ready":
        if last_detection_time is not None and (time.time() - last_detection_time) > 1.0:
            current_state = "waiting"
            print("State changed to WAITING (no detection for 1s in READY)")

    # -------------------------------------------------------------------------
    # "TRACKING" → "WAITING" 조건(1): 마지막 검출 이후 1초 이상 감지 X
    # -------------------------------------------------------------------------
    if current_state == "tracking":
        if last_detection_time is not None and (time.time() - last_detection_time) >= 1.0:
            bounce_history.append(bounce_count)
            if len(bounce_history) > 8:
                bounce_history.pop(0)

            bounce_count = 0
            consecutiveDownCount = 0
            consecutiveUpCount = 0
            state = None
            current_state = "waiting"
            print("No detection for 1 second in TRACKING => bounce_count reset to 0, state changed to WAITING")

    # 그래프 데이터 길이 제한
    if len(x_values) > MAX_POINTS:
        x_values.pop(0)
        y_values.pop(0)
        orange_pixel_values.pop(0)

    # 바운스 연속 감지 제한 (Optional)
    if last_bounce_time is not None:
        if time.time() - last_bounce_time > CONTINUOUS_TIMEOUT:
            bounce_count = 0
            last_bounce_time = None
            print("No bounce for a while -> reset bounce_count to 0")

    # ------------------------------------------------------------------------------------
    # (B) Combined 화면 만들기
    # ------------------------------------------------------------------------------------
    combined_img = np.zeros((960, 1280, 3), dtype=np.uint8)

    # 먼저 y_graph_img, orange_graph_img, frame_resized 등 생성
    frame_resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

    # ### (추가/수정 부분) : 여기서 frame_resized에 State, FPS, Bounce Dt를 표시
    cv2.putText(
        frame_resized,
        f"State: {current_state.upper()}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        frame_resized,
        f"FPS: {fps:.2f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    if bounce_time_diff is not None:
        cv2.putText(
            frame_resized,
            f"Bounce Dt: {bounce_time_diff:.2f}s",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # 드래그/리사이즈 사각형
    cv2.rectangle(
        frame_resized,
        (drag_rect_x, drag_rect_y),
        (drag_rect_x + drag_rect_w, drag_rect_y + drag_rect_h),
        (0, 0, 255),
        2
    )
    # 각 코너 표시
    corners = [
        (drag_rect_x, drag_rect_y),
        (drag_rect_x + drag_rect_w, drag_rect_y),
        (drag_rect_x, drag_rect_y + drag_rect_h),
        (drag_rect_x + drag_rect_w, drag_rect_y + drag_rect_h)
    ]
    for (cx, cy) in corners:
        cv2.rectangle(
            frame_resized,
            (cx - corner_size, cy - corner_size),
            (cx + corner_size, cy + corner_size),
            (0, 0, 255),
            -1
        )

    # 사각형 내부 실시간 시간 표시
    cv2.putText(
        frame_resized,
        f"In-Rect Time: {in_rect_time:.2f}s",
        (drag_rect_x + 5, drag_rect_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    # y_graph / orange_graph
    y_graph_img = draw_y_graph(
        x_values,
        y_values,
        width=640,
        height=480,
        max_y=480,
        bounce_pts=bounce_points
    )
    valid_orange = [v for v in orange_pixel_values if v is not None]
    max_orange = max(valid_orange) if valid_orange else 1
    orange_graph_img = draw_orange_graph(
        x_values,
        orange_pixel_values,
        width=640,
        height=480,
        max_y=max_orange
    )

    # enlarged_view 여부에 따라 화면 배치
    if enlarged_view is None:
        # 4분할 표시
        combined_img[0:480, 0:640] = frame_resized       # top-left
        combined_img[0:480, 640:1280] = y_graph_img      # top-right
        combined_img[480:960, 0:640] = orange_graph_img  # bottom-left

        # bottom-right (480:960, 640:1280) => bounce_history 표시
        square_width = 55
        margin = 20
        num_squares = 8
        total_width = num_squares * square_width + (num_squares - 1) * margin
        offset_x = 1280 - total_width - margin
        offset_y = 960 - square_width - margin

        # 각 사각형에 이름과 숫자를 표시하기 위한 리스트 (예시)
        names = [f"Name{i+1}" for i in range(num_squares)]
        numbers = bounce_history[-num_squares:]  # 마지막 8개 값 사용

        for i in range(num_squares):
            x1 = offset_x + i * (square_width + margin)
            y1 = offset_y
            x2 = x1 + square_width
            y2 = y1 + square_width

            # 사각형 그리기
            cv2.rectangle(combined_img, (x1, y1), (x2, y2), (255, 255, 255), 2)

            if i < len(numbers):
                # 이름 그리기 (사각형 위쪽)
                name = names[i]
                (text_w, text_h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                text_x = x1 + (square_width - text_w) // 2
                text_y = y1 + text_h + 2  # 사각형 위쪽 여백
                cv2.putText(
                    combined_img,
                    name,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

                # 숫자 그리기 (사각형 아래쪽)
                number = str(numbers[i])
                (num_w, num_h), _ = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                num_x = x1 + (square_width - num_w) // 2
                num_y = y2 - 5  # 사각형 아래쪽 여백
                cv2.putText(
                    combined_img,
                    number,
                    (x1, num_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

    else:
        # (A) 'tl', 'tr', 'bl', 'br' 중 하나만 크게
        if enlarged_view == 'tl':
            big_view = cv2.resize(frame_resized, (1280, 960), interpolation=cv2.INTER_AREA)
            combined_img = big_view
        elif enlarged_view == 'tr':
            big_view = cv2.resize(y_graph_img, (1280, 960), interpolation=cv2.INTER_AREA)
            combined_img = big_view
        elif enlarged_view == 'bl':
            big_view = cv2.resize(orange_graph_img, (1280, 960), interpolation=cv2.INTER_AREA)
            combined_img = big_view
        elif enlarged_view == 'br':
            # (A) 'tl', 'tr', 'bl', 'br' 중 하나만 크게
            # 1) bottom-right 확대를 위해 640x480 캔버스(sub_img) 만들기
            sub_img = np.zeros((480, 640, 3), dtype=np.uint8)

            # 사각형 세로 길이를 조정할 변수 도입
            rectangle_height = 80  # 원하는 세로 길이로 조정

            # 2) bounce history 사각형을 그리는 로직 수행
            #    (4분할에서 bottom-right에 그리던 부분을 수정)
            square_width = 55
            margin = 20
            num_squares = 8
            total_width = num_squares * square_width + (num_squares - 1) * margin
            offset_x = 640 - total_width - margin
            offset_y = 480 - rectangle_height - margin  # 세로 위치 조정

            # 각 사각형에 이름과 숫자를 표시하기 위한 리스트 (예시)
            names = [f"Name{i+1}" for i in range(num_squares)]
            # 예시로 bounce_history를 이름과 매칭 (실제 데이터에 맞게 수정 필요)
            numbers = bounce_history[-num_squares:]  # 마지막 8개 값 사용

            for i in range(num_squares):
                x1 = offset_x + i * (square_width + margin)
                y1 = offset_y
                x2 = x1 + square_width
                y2 = y1 + rectangle_height

                # 사각형 그리기
                cv2.rectangle(sub_img, (x1, y1), (x2, y2), (255, 255, 255), 2)

                if i < len(numbers):
                    # 이름 그리기 (사각형 위쪽)
                    name = names[i]
                    (text_w, text_h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    text_x = x1 + (square_width - text_w) // 2
                    text_y = y1 + text_h + 5  # 사각형 위쪽 여백
                    cv2.putText(
                        sub_img,
                        name,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA
                    )

                    # 숫자 그리기 (사각형 아래쪽)
                    number = str(numbers[i])
                    (num_w, num_h), _ = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                    num_x = x1 + (square_width - num_w) // 2
                    num_y = y2 - 10  # 사각형 아래쪽 여백
                    cv2.putText(
                        sub_img,
                        number,
                        (num_x, num_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

            # 3) 완성된 sub_img를 1280x960으로 확대 후 combined_img에 넣기
            big_view = cv2.resize(sub_img, (1280, 960), interpolation=cv2.INTER_AREA)
            combined_img = big_view

    # 최종 표시
    cv2.imshow("Combined", combined_img)

    # 바운스 카운트 전용 윈도우
    if bounce_count != prev_bounce_count:
        color = get_color(bounce_count)
        bounce_img = render_text_with_ttf(
            text=str(bounce_count),
            font=font,
            text_color=color,
            bg_color=(0, 0, 0),
            width=960,
            height=540
        )
        prev_bounce_count = bounce_count

    if bounce_img is not None:
        cv2.imshow("Bounce Count Window", bounce_img)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key in [ord('f'), ord('F')]:
        if is_fullscreen_combined:
            cv2.setWindowProperty("Combined", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        else:
            cv2.setWindowProperty("Combined", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
        if is_fullscreen_bounce:
            cv2.setWindowProperty("Bounce Count Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        else:
            cv2.setWindowProperty("Bounce Count Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        is_fullscreen_combined = not is_fullscreen_combined
        is_fullscreen_bounce = not is_fullscreen_bounce

# ----------------------------------------------------------------------------------------
# 16) 종료 처리
# ----------------------------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
