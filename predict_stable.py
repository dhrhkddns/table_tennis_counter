import cv2
import numpy as np
import time
from ultralytics import YOLO
import pygame
from PIL import Image, ImageDraw, ImageFont
import threading
from playsound import playsound
import ctypes  # 시스템 커서 제어용 (Windows 전용)
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
model.to("cuda")  # GPU 사용, 필요에 따라 "cpu"로 변경 가능

# ----------------------------------------------------------------------------------------
# 3) 카메라 디바이스 연결
# ----------------------------------------------------------------------------------------
cap = cv2.VideoCapture(2)  # 내장 웹캠은 0, 외장캠은 1 등으로 조정하세요

# ----------------------------------------------------------------------------------------
# 4) 그래프, 바운스 관련 전역 변수 정의
# ----------------------------------------------------------------------------------------
x_values = []
y_values = []
orange_pixel_values = []
frame_count = 0
MAX_POINTS = 100  # 그래프에 남길 최대 포인트 수

bounce_count = 0

consecutiveDownCount = 0
consecutiveUpCount = 0
state = None  # "down" 혹은 "up"
DOWN_THRESHOLD = 2
UP_THRESHOLD = 1
PIXEL_THRESHOLD = 3.0
last_y = None

bounce_points = []
bounce_times = []

CONTINUOUS_TIMEOUT = 1.0
last_bounce_time = None

# ----------------------------------------------------------------------------------------
# 5) 사운드 켜짐/꺼짐 & 오렌지 픽셀 0 무시 여부 토글을 위한 전역 변수
# ----------------------------------------------------------------------------------------
sound_enabled = False
ignore_zero_orange = False

# ----------------------------------------------------------------------------------------
# 6) 버튼 위치 정의 (버튼 Rect: [x, y, width, height])
#    - 첫 번째 버튼: 사운드 Toggle
#    - 두 번째 버튼: 오렌지 픽셀 0 무시 Toggle
# ----------------------------------------------------------------------------------------
button_rect = [500, 20, 120, 40]          # 사운드 토글 버튼
button_rect_ignore = [500, 70, 120, 40]  # 오렌지 0 무시 토글 버튼

# ----------------------------------------------------------------------------------------
# 7) 폰트 설정 (TTF)
# ----------------------------------------------------------------------------------------
FONT_PATH = r"C:\Users\omyra\Desktop\coding\ping_pong\Digital Display.ttf"
FONT_SIZE = 400
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

# ----------------------------------------------------------------------------------------
# 8) 색상 순서 및 강도 정의
# ----------------------------------------------------------------------------------------
# 10가지 색상 순서 (BGR)
color_sequence = [
    (255, 255, 255),  # 0-99: 흰색 (White)
    (0, 0, 255),      # 100-199: 빨강 (Red)
    (0, 165, 255),    # 200-299: 주황 (Orange)
    (0, 255, 255),    # 300-399: 노랑 (Yellow)
    (144, 238, 144),  # 400-499: 연두 (Light Green)
    (0, 255, 0),      # 500-599: 초록 (Green)
    (255, 255, 0),    # 600-699: 청록 (Cyan)
    (255, 0, 0),      # 700-799: 파랑 (Blue)
    (128, 0, 255),    # 800-899: 보라 (Purple) 
    (255, 0, 255)     # 900-999: 자주 (Magenta)
]

# 강도 단계 (0.5 ~ 1.0, 10단계)
intensity_levels = [0.5 + 0.05 * i for i in range(10)]  # 0.5, 0.55, ..., 0.95

def get_color(count):
    """
    카운트에 따라 색상을 반환합니다.
    - 100마다 색상 블록 변경 (0-99, 100-199, ..., 900-999)
    - 각 블록 내에서 10 단위로 강도 증가
    """
    if count >= 1000:
        return (255, 0, 255)  # 1000 이상: 자주 (Magenta)

    color_index = count // 100  # 0-9
    if color_index >= len(color_sequence):
        color_index = len(color_sequence) - 1

    step_in_block = (count % 100) // 10  # 0-9

    # 강도 적용
    intensity = intensity_levels[step_in_block] if step_in_block < len(intensity_levels) else 1.0
    base_color = color_sequence[color_index]

    # BGR to HSV 변환
    color_bgr = np.uint8([[base_color]])
    color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]

    # Value 조절 (밝기 증가)
    color_hsv = color_hsv.astype(float)
    color_hsv[2] = min(color_hsv[2] * intensity, 255)
    color_hsv = color_hsv.astype(np.uint8)

    # HSV to BGR 변환
    intense_color = cv2.cvtColor(np.uint8([[color_hsv]]), cv2.COLOR_HSV2BGR)[0][0]

    # BGR to RGB 변환
    intense_color_rgb = (intense_color[2], intense_color[1], intense_color[0])

    return intense_color_rgb

# ----------------------------------------------------------------------------------------
# 9) mouse_callback 함수
#    - "Combined" 창에 대한 마우스 클릭 콜백
#    - 각 버튼 영역을 클릭하면 토글
# ----------------------------------------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global sound_enabled, ignore_zero_orange
    global last_mouse_move_time, mouse_visible  # 전역 변수 추가
    
    # 마우스가 움직이면 시간 갱신 및 커서 표시
    if event == cv2.EVENT_MOUSEMOVE:
        last_mouse_move_time = time.time()
        if not mouse_visible:
            user32.ShowCursor(True)
            mouse_visible = True
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # 사운드 토글 버튼
        if (button_rect[0] <= x - 640 <= button_rect[0] + button_rect[2] and
            button_rect[1] <= y <= button_rect[1] + button_rect[3]):
            sound_enabled = not sound_enabled
            print(f"Sound Enabled: {sound_enabled}")
        
        # 오렌지 픽셀 0 무시 토글 버튼
        elif (button_rect_ignore[0] <= x - 640 <= button_rect_ignore[0] + button_rect_ignore[2] and
              button_rect_ignore[1] <= y <= button_rect_ignore[1] + button_rect_ignore[3]):
            ignore_zero_orange = not ignore_zero_orange
            print(f"Ignore Zero Orange Pixels: {ignore_zero_orange}")

# ----------------------------------------------------------------------------------------
# 10) render_text_with_ttf(): Pillow TTF 폰트를 이용해 텍스트를 그린 뒤 OpenCV 이미지로 반환
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
        # 데이터가 부족하면 그냥 검은 배경
        return graph_img

    max_x = x_data[-1] if x_data[-1] != 0 else 1

    # y 그래프 선
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

    # 데이터 포인트(원)
    for i in range(len(x_data)):
        if y_data[i] is None:
            continue
        x_ori, y_ori = x_data[i], y_data[i]
        x_pt = int((x_ori - x_data[0]) / (max_x - x_data[0] + 1e-6) * (width - 1))
        y_pt = int(y_ori / max_y * (height - 1))
        cv2.circle(graph_img, (x_pt, y_pt), 4, (255, 0, 0), -1)

    # 바운스 지점 표시
    for (bx_ori, by_ori) in bounce_pts:
        if bx_ori < x_data[0]:
            continue
        bx = int((bx_ori - x_data[0]) / (max_x - x_data[0] + 1e-6) * (width - 1))
        by = int(by_ori / max_y * (height - 1))
        cv2.circle(graph_img, (bx, by), 5, (0, 0, 255), -1)

    # 사운드 토글 버튼
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

    # 오렌지 픽셀 0 무시 토글 버튼
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

    # 축
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

# 마우스 콜백 설정
cv2.setMouseCallback("Combined", mouse_callback)

# 바운스 카운트 창 관련
prev_bounce_count = None
bounce_img = None
is_fullscreen = False  # Bounce Count Window 전체화면 토글이 필요하면 사용

# 마우스 커서 제어를 위한 전역 변수 추가 (main 루프 전에 추가)
last_mouse_move_time = time.time()
mouse_visible = True

# ----------------------------------------------------------------------------------------
# 14) 메인 루프
# ----------------------------------------------------------------------------------------
while True:
    # 마우스 3초 이상 움직이지 않으면 커서 숨김
    now = time.time()
    if now - last_mouse_move_time > 3.0:
        if mouse_visible:
            user32.ShowCursor(False)
            mouse_visible = False
            
    ret, frame = cap.read()
    if not ret:
        print("No more frames or camera error.")
        break

    # YOLO 추론
    results = model.predict(frame, imgsz=640, conf=0.5, max_det=1, show=False, device=0)
    boxes = results[0].boxes

    # 프레임 인덱스 누적
    x_values.append(frame_count)
    frame_count += 1

    detected = False  # 검출 여부 초기화
    orange_pixels = 0  # 주황색 픽셀 수 초기화

    if len(boxes) > 0:
        x1, y1, x2, y2 = boxes[0].xyxy[0].cpu().numpy()
        y_center = (y1 + y2) / 2.0

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

        # ignore_zero_orange가 True면 orange_pixels가 5 이상일 때만 공으로 인식
        if ignore_zero_orange:
            if orange_pixels >= 5:
                detected = True
        else:
            detected = True  # 토글이 꺼져있으면 오렌지 픽셀 수와 관계없이 검출됨

        if detected:
            y_values.append(y_center)
            orange_pixel_values.append(orange_pixels)

            # 바운스 로직
            if last_y is not None:
                dy = y_center - last_y
                if abs(dy) > PIXEL_THRESHOLD:
                    if dy > 0:
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
                            last_bounce_time = time.time()
                            bounce_times.append(last_bounce_time)
                            if len(bounce_times) >= 2:
                                time_diff = bounce_times[-1] - bounce_times[-2]
                                print(f"Time diff between last two bounces: {time_diff:.2f} s")

                            state = "up"
                            consecutiveDownCount = 0
                            consecutiveUpCount = 0
                    elif state == "up":
                        if consecutiveDownCount >= DOWN_THRESHOLD:
                            state = "down"
                            consecutiveUpCount = 0
                            consecutiveDownCount = 0

            last_y = y_center

            # 박스 표시
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
            # orange_pixels가 0 이하이면 공 미검출로 처리
            y_values.append(None)
            orange_pixel_values.append(None)
    else:
        # 박스 미검출 시
        y_values.append(None)
        orange_pixel_values.append(None)

    # 오래된 데이터 제거
    if len(x_values) > MAX_POINTS:
        x_values.pop(0)
        y_values.pop(0)
        orange_pixel_values.pop(0)

    # 연속 모드(바운스) 타임아웃
    if last_bounce_time is not None:
        if time.time() - last_bounce_time > CONTINUOUS_TIMEOUT:
            bounce_count = 0
            last_bounce_time = None
            print("No bounce for a while -> reset bounce_count to 0")

    # 640×480 사이즈로 리사이즈
    frame_resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

    # 그래프 이미지들
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

    # 1280×960 크기의 빈 캔버스 생성 후 위치별로 배치
    combined_img = np.zeros((960, 1280, 3), dtype=np.uint8)
    combined_img[0:480, 0:640]      = frame_resized
    combined_img[0:480, 640:1280]   = y_graph_img
    combined_img[480:960, 0:640]    = orange_graph_img
    # 오른쪽 아래 영역(480:960, 640:1280)은 사용 안 함(검은색)

    cv2.imshow("Combined", combined_img)

    # 바운스 카운트 창 갱신
    if bounce_count != prev_bounce_count:
        color = get_color(bounce_count)
        bounce_img = render_text_with_ttf(
            text=str(bounce_count),
            font=font,
            text_color=color,  # 이미 RGB로 변환된 색상
            bg_color=(0, 0, 0),
            width=960,
            height=540
        )
        prev_bounce_count = bounce_count

    if bounce_img is not None:
        cv2.imshow("Bounce Count Window", bounce_img)

    # 키보드 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC -> 종료
        break
    elif key in [ord('f'), ord('F')]:
        # Combined 창 전체화면 ↔ 창 모드 토글
        if is_fullscreen_combined:
            cv2.setWindowProperty("Combined", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        else:
            cv2.setWindowProperty("Combined", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        is_fullscreen_combined = not is_fullscreen_combined

    # 프레임 처리 시간 측정 및 로그 (필요시 활성화)
    # end_time = time.time()
    # elapsed = end_time - start_time
    # print(f"Frame processing time: {elapsed:.4f} seconds")

# ----------------------------------------------------------------------------------------
# 15) 종료 처리
# ----------------------------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
