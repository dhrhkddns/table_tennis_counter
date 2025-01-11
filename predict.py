# 필요한 라이브러리 임포트
import cv2  # OpenCV 라이브러리 - 이미지/비디오 처리용 (예: frame = cv2.imread("image.jpg"))
import numpy as np  # 수치 연산 라이브러리 - 배열 처리용 (예: arr = np.zeros((10,10)))
import time  # 시간 측정용 라이브러리 (예: start_time = time.time())
from ultralytics import YOLO  # YOLO 객체 탐지 모델 (예: model = YOLO("yolov8n.pt"))
import pygame  # 오디오 재생용 라이브러리 (예: pygame.mixer.Sound("beep.wav").play())
from PIL import Image, ImageDraw, ImageFont  # 이미지 처리/텍스트 렌더링용 (예: img = Image.new("RGB", (100,100)))
import ctypes  # Windows API 접근용 (예: user32.ShowCursor(False))
user32 = ctypes.windll.user32  # Windows 사용자 인터페이스 제어용

# ----------------------------------------------------------------------------------------
# 1) pygame 오디오 초기화 및 사운드 로드
# ----------------------------------------------------------------------------------------
pygame.mixer.init()  # pygame 오디오 시스템 초기화 (예: 44100Hz, 16bit, stereo)
sound = pygame.mixer.Sound(r"C:\Users\omyra\Desktop\coding\ping_pong\retro-coin-4-236671.mp3")  # 효과음 파일 로드

# ----------------------------------------------------------------------------------------
# 2) YOLO 모델 로드
# ----------------------------------------------------------------------------------------
model = YOLO(r"C:\Users\omyra\Desktop\coding\ping_pong\Ping-Pong-Detection-3\Results\weights\best.pt")  # 학습된 YOLO 모델 로드
model.to("cuda")  # GPU 메모리로 모델 이동 (예: NVIDIA RTX 3080에서 실행)

# ----------------------------------------------------------------------------------------
# 3) 카메라 디바이스 연결
# ----------------------------------------------------------------------------------------
cap = cv2.VideoCapture(1)  # 카메라 연결 (예: 외장 웹캠은 보통 1번 인덱스)
# 카메라 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 가로 해상도 설정 (예: 640픽셀)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 세로 해상도 설정 (예: 480픽셀)
cap.set(cv2.CAP_PROP_FPS, 35)  # FPS 설정 (예: 초당 30프레임)

# ----------------------------------------------------------------------------------------
# 4) 그래프, 바운스 관련 전역 변수 정의
# ----------------------------------------------------------------------------------------
x_values = []
y_values = []
orange_pixel_values = []
frame_count = 0
MAX_POINTS = 100

bounce_count = 0

consecutiveDownCount = 0
consecutiveUpCount = 0
state = None
DOWN_THRESHOLD = 2
UP_THRESHOLD = 1
PIXEL_THRESHOLD = 3.0
last_y = None

bounce_points = []
bounce_times = []

CONTINUOUS_TIMEOUT = 1.0
last_bounce_time = None

sound_enabled = False
ignore_zero_orange = False

button_rect = [500, 20, 120, 40]
button_rect_ignore = [500, 70, 120, 40]

FONT_PATH = r"C:\Users\omyra\Desktop\coding\ping_pong\Digital Display.ttf"
FONT_SIZE = 400
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

color_sequence = [
    (255, 255, 255),
    (0, 0, 255),
    (0, 165, 255),
    (0, 255, 255),
    (144, 238, 144),
    (0, 255, 0),
    (255, 255, 0),
    (255, 0, 0),
    (128, 0, 255),
    (255, 0, 255)
]
intensity_levels = [0.5 + 0.05 * i for i in range(10)]

def get_color(count):
    if count >= 1000:
        return (255, 0, 255)

    color_index = count // 100
    if color_index >= len(color_sequence):
        color_index = len(color_sequence) - 1

    step_in_block = (count % 100) // 10
    intensity = intensity_levels[step_in_block] if step_in_block < len(intensity_levels) else 1.0
    base_color = color_sequence[color_index]

    color_bgr = np.uint8([[base_color]])
    color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]

    color_hsv = color_hsv.astype(float)
    color_hsv[2] = min(color_hsv[2] * intensity, 255)
    color_hsv = color_hsv.astype(np.uint8)

    intense_color = cv2.cvtColor(np.uint8([[color_hsv]]), cv2.COLOR_HSV2BGR)[0][0]
    intense_color_rgb = (intense_color[2], intense_color[1], intense_color[0])
    return intense_color_rgb

# ----------------------------------------------------------------------------------------
# 9) mouse_callback 함수
# ----------------------------------------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global sound_enabled, ignore_zero_orange
    global last_mouse_move_time, mouse_visible

    if event == cv2.EVENT_MOUSEMOVE:
        last_mouse_move_time = time.time()
        if not mouse_visible:
            user32.ShowCursor(True)
            mouse_visible = True

    if event == cv2.EVENT_LBUTTONDOWN:
        if (button_rect[0] <= x - 640 <= button_rect[0] + button_rect[2] and
            button_rect[1] <= y <= button_rect[1] + button_rect[3]):
            sound_enabled = not sound_enabled
            print(f"Sound Enabled: {sound_enabled}")
        
        elif (button_rect_ignore[0] <= x - 640 <= button_rect_ignore[0] + button_rect_ignore[2] and
              button_rect_ignore[1] <= y <= button_rect_ignore[1] + button_rect_ignore[3]):
            ignore_zero_orange = not ignore_zero_orange
            print(f"Ignore Zero Orange Pixels: {ignore_zero_orange}")

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

last_mouse_move_time = time.time()
mouse_visible = True

# ----------------------------------------------------------------------------------------
# 14) 추가된 전역 변수: 상태 관리
# ----------------------------------------------------------------------------------------
current_state = "waiting"                                # 현재 상태 (예: "waiting", "moving", "bouncing" 등)
state_display_text = "Waiting"                          # 화면에 표시될 상태 텍스트 (예: "Waiting", "Ball Moving" 등)
state_font = cv2.FONT_HERSHEY_SIMPLEX                  # 상태 텍스트에 사용될 폰트 종류
state_font_scale = 1.0                                 # 상태 텍스트의 크기 (1.0 = 기본 크기)
state_font_color = (255, 255, 255)                     # 상태 텍스트의 색상 (흰색 = (255, 255, 255))
state_font_thickness = 2                               # 상태 텍스트의 두께 (2 픽셀)
state_change_time = None                               # 상태가 마지막으로 변경된 시간 (예: 1234567890.123)

stationary_start_time = None                           # 공이 정지 상태로 진입한 시작 시간 (예: 1234567890.123)
stationary_threshold = 2.0                             # 공이 정지했다고 판단하는 시간 기준값 (2.0초)
movement_threshold = 5                                 # 공의 움직임을 감지하는 픽셀 단위 임계값 (5픽셀)
last_position = None                                   # 이전 프레임에서의 공의 위치 (예: (100, 200))

previous_bounce_time = None                            # 이전 바운스가 발생한 시간 (예: 1234567890.123)

# ----------------------------------------------------------------------------------------
# 14-1) 추가: 마지막 검출 시각(공이 마지막으로 발견된 시간)을 저장할 변수
# ----------------------------------------------------------------------------------------
last_detection_time = None

# ========================================================================================
# ## (추가된 부분) 바운스 간 시간차를 표시하기 위한 전역 변수
# ========================================================================================
bounce_time_diff = None  # 가장 최근 두 바운스 사이의 시간차를 저장할 변수

# ----------------------------------------------------------------------------------------
# 15) 메인 루프 (여기서 FPS 계산 & 원하는 FPS 맞추기)
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

        if ignore_zero_orange:
            if orange_pixels >= 5:
                detected = True
        else:
            detected = True

        if detected:
            last_detection_time = time.time()

            y_values.append(y_center)
            orange_pixel_values.append(orange_pixels)

            if last_position is not None:
                dy = y_center - last_position
                movement = abs(dy)
            else:
                dy = 0
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
                elif (time.time() - stationary_start_time) >= stationary_threshold:  # 예: 공이 2초 이상 정지해있는지 확인 (stationary_threshold가 2초일 경우)
                    if current_state != "ready":
                        current_state = "ready"
                        state_change_time = time.time()
                        print("State changed to READY")

            last_position = y_center

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
                                    # =================================================================================
                                    # ## (추가된 부분) 바운스 간 시간차를 bounce_time_diff에 저장
                                    # =================================================================================
                                    bounce_time_diff = td
                                    # ---------------------------------------------------------------------------------

                                    if td > 1.0:
                                        current_state = "waiting"
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

    if current_state == "tracking":
        if last_detection_time is not None and (time.time() - last_detection_time) >= 1.0:
            bounce_count = 0
            consecutiveDownCount = 0
            consecutiveUpCount = 0
            state = None
            current_state = "waiting"
            print("No detection for 1 second in TRACKING => bounce_count reset to 0, state changed to WAITING")

    if len(x_values) > MAX_POINTS:
        x_values.pop(0)
        y_values.pop(0)
        orange_pixel_values.pop(0)

    if last_bounce_time is not None:
        if time.time() - last_bounce_time > CONTINUOUS_TIMEOUT:
            bounce_count = 0
            last_bounce_time = None
            print("No bounce for a while -> reset bounce_count to 0")

    frame_resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

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

    combined_img = np.zeros((960, 1280, 3), dtype=np.uint8)
    combined_img[0:480, 0:640] = frame_resized
    combined_img[0:480, 640:1280] = y_graph_img
    combined_img[480:960, 0:640] = orange_graph_img

    cv2.putText(
        combined_img,
        f"State: {current_state.upper()}",
        (10, 30),
        state_font,
        1.0,
        state_font_color,
        state_font_thickness,
        cv2.LINE_AA
    )

    # FPS 표시
    cv2.putText(
        combined_img,
        f"FPS: {fps:.2f}",
        (10, 60),
        state_font,
        1.0,
        (0, 0, 0),
        state_font_thickness,
        cv2.LINE_AA
    )

    # =========================================================================
    # ## (추가된 부분) bounce_time_diff를 FPS 표시 바로 아래(예: y=90)에 표시
    # =========================================================================
    if bounce_time_diff is not None:
        cv2.putText(
            combined_img,
            f"Bounce Dt: {bounce_time_diff:.2f}s",
            (10, 90),  # FPS표시가 (10,60)이므로 그 바로 아래 90
            state_font,
            1.0,
            (0, 0, 0),
            state_font_thickness,
            cv2.LINE_AA
        )
    # -------------------------------------------------------------------------

    cv2.imshow("Combined", combined_img)

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
    if key == 27:
        break
    elif key in [ord('f'), ord('F')]:
        if is_fullscreen_combined:
            cv2.setWindowProperty("Combined", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        else:
            cv2.setWindowProperty("Combined", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        is_fullscreen_combined = not is_fullscreen_combined

# ----------------------------------------------------------------------------------------
# 16) 종료 처리
# ----------------------------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
