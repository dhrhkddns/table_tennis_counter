import cv2
import numpy as np
import time
import ctypes
import threading
from ultralytics import YOLO
import pygame
from PIL import Image, ImageFont, ImageDraw
from gtts import gTTS
import playsound

user32 = ctypes.windll.user32

# ----------------------------------------------------------------------------------------
# 1) pygame 오디오 초기화 및 사운드 로드
# ----------------------------------------------------------------------------------------
pygame.mixer.init()
#공이 준비사각형에 일정시간 있었을때 효과음
ready_sound = pygame.mixer.Sound(r"C:\Users\omyra\Desktop\coding\ping_pong\jihun_준비완료.mp3")
#1단위 바운스 효과음
bounce_count_sound = pygame.mixer.Sound(r"C:\Users\omyra\Desktop\coding\ping_pong\retro-coin-4-236671.mp3")
#10단위 바운스 효과음
collect_points_sound = pygame.mixer.Sound(r"C:\Users\omyra\Desktop\coding\ping_pong\collect-points-190037.mp3")  
#100단위 바운스 효과음
score_sound = pygame.mixer.Sound(r"C:\Users\omyra\Desktop\coding\ping_pong\score_sound.mp3")  
#공이 처음 리사이즈 사각형 안에 들어왔을때 효과음
tap_notification_sound = pygame.mixer.Sound(r"C:\Users\omyra\Desktop\coding\ping_pong\tap-notification-180637.mp3")
#공이 tracking 상태에서 나가거나 사라졌을때 (한 턴 종료)
alert_sound = pygame.mixer.Sound(r"C:\Users\omyra\Desktop\coding\ping_pong\alert-234711.mp3")
#결승 승리 효과음
final_win_sound = pygame.mixer.Sound(r"C:\Users\omyra\Desktop\coding\ping_pong\game-level-complete-143022.mp3")

#100,200,300...1000 단위로 효과음 재생----------------------------------------------------------------------------------------
hundred_unit_sounds = {
    100: pygame.mixer.Sound(r"터르난도-100.mp3"),
    200: pygame.mixer.Sound(r"터르난도-200.mp3"),
    300: pygame.mixer.Sound(r"터르난도-300.mp3"),
    400: pygame.mixer.Sound(r"터르난도-400.mp3"),
    500: pygame.mixer.Sound(r"터르난도-500.mp3"),
    600: pygame.mixer.Sound(r"터르난도-600.mp3"),
    700: pygame.mixer.Sound(r"터르난도-700.mp3"),
    800: pygame.mixer.Sound(r"터르난도-800.mp3"),
    900: pygame.mixer.Sound(r"터르난도-900.mp3"),
    1000: pygame.mixer.Sound(r"터르난도-1000.mp3"),
}

# [2] 전역 변수: 마스터 볼륨 & 효과음별 '상대 볼륨' (모두 1.0으로 초기화)
master_volume = 1.0
rel_bounce1 = 1.0
rel_bounce10 = 1.0
rel_bounce100 = 1.0
rel_tap = 1.0
rel_ready = 1.0
rel_alert = 1.0
rel_final_win = 1.0
rel_hundred_units = 1.0 #100,200.. 성우 목소리

def update_final_volumes():
    """
    마스터볼륨 * 상대볼륨 = 실제볼륨 으로 각 사운드에 set_volume()을 적용
    """
    bounce_count_sound.set_volume(master_volume * rel_bounce1)
    collect_points_sound.set_volume(master_volume * rel_bounce10)
    score_sound.set_volume(master_volume * rel_bounce100)
    tap_notification_sound.set_volume(master_volume * rel_tap)
    ready_sound.set_volume(master_volume * rel_ready)
    alert_sound.set_volume(master_volume * rel_alert)
    final_win_sound.set_volume(master_volume * rel_final_win)

    for sound in hundred_unit_sounds.values():
        sound.set_volume(master_volume * rel_hundred_units)
# 'Volume Control' 창과 Trackbar 콜백 함수 정의
# [3] 트랙바 콜백 함수들
# ------------------------------
def on_trackbar_master(val):
    global master_volume
    master_volume = val / 100.0
    update_final_volumes()

def on_trackbar_bounce1(val):
    global rel_bounce1
    rel_bounce1 = val / 100.0
    update_final_volumes()

def on_trackbar_bounce10(val):
    global rel_bounce10
    rel_bounce10 = val / 100.0
    update_final_volumes()

def on_trackbar_bounce100(val):
    global rel_bounce100
    rel_bounce100 = val / 100.0
    update_final_volumes()

def on_trackbar_tap(val):
    global rel_tap
    rel_tap = val / 100.0
    update_final_volumes()

def on_trackbar_ready(val):
    global rel_ready
    rel_ready = val / 100.0
    update_final_volumes()

def on_trackbar_alert(val):
    global rel_alert
    rel_alert = val / 100.0
    update_final_volumes()

def on_trackbar_final_win(val):
    global rel_final_win
    rel_final_win = val / 100.0
    update_final_volumes()

def on_trackbar_hundred_units(val):
    global rel_hundred_units
    rel_hundred_units = val / 100.0
    update_final_volumes()

# 새 창 생성
cv2.namedWindow("Volume Control")

# 0~100 범위, 현재값 100(=1.0)으로 설정
cv2.createTrackbar("MasterVolume", "Volume Control", 70, 100, on_trackbar_master)
cv2.createTrackbar("Bounce1", "Volume Control", 20, 100, on_trackbar_bounce1)
cv2.createTrackbar("Bounce10", "Volume Control", 35, 100, on_trackbar_bounce10)
cv2.createTrackbar("Bounce100", "Volume Control", 40, 100, on_trackbar_bounce100)
cv2.createTrackbar("Tap", "Volume Control", 80, 100, on_trackbar_tap)
cv2.createTrackbar("Ready", "Volume Control", 65, 100, on_trackbar_ready)
cv2.createTrackbar("Alert", "Volume Control", 40, 100, on_trackbar_alert)
cv2.createTrackbar("FinalWin", "Volume Control", 40, 100, on_trackbar_final_win)
cv2.createTrackbar("HundredUnits", "Volume Control", 50, 100, on_trackbar_hundred_units)

def speak_winner(name):
    """
    우승자 이름을 gTTS로 발음해주는 간단한 함수 예시
    여기서는 lang='ko'로 하여 "A 우승!" 형태로 말하도록 했습니다.
    필요시 영어, 다른 문구 등으로 변경 가능.
    """
    text = f"{name} 우승!"
    tts = gTTS(text=text, lang='ko')
    tts.save("winner.mp3")
    pygame.mixer.music.load("winner.mp3")
    pygame.mixer.music.play()


def mouse_callback_volume(event, x, y, flags, param):
    global last_mouse_move_time, mouse_visible
    if event == cv2.EVENT_MOUSEMOVE:
        last_mouse_move_time = time.time()
        if not mouse_visible:
            user32.ShowCursor(True)
            mouse_visible = True


cv2.setMouseCallback("Volume Control", mouse_callback_volume)






# 2) YOLO 모델 로드
# ----------------------------------------------------------------------------------------
model = YOLO(r"C:\Users\omyra\Desktop\coding\ping_pong\Ping-Pong-Detection-3\Results\weights\best.pt")
model.to("cuda")

# ----------------------------------------------------------------------------------------
# 3) 카메라 디바이스 연결
# ----------------------------------------------------------------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

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
PIXEL_THRESHOLD = 5.0
last_y = None

bounce_points = []
bounce_times = []

CONTINUOUS_TIMEOUT = 1.2
current_bounce_time = None

sound_enabled = True
ignore_zero_orange = True

button_rect = [500, 20, 120, 40]
button_rect_ignore = [500, 70, 120, 40]

# 웹캠 선택을 위한 전역 변수
current_camera_index = 0
webcam_button_rects = []

FONT_PATH = r"C:\Users\omyra\Desktop\coding\ping_pong\Digital Display.ttf"
FONT_SIZE = 400
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

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
intensity_levels = [0.5 + 0.05 * i for i in range(10)]


def get_color(count):
    """
    바운스 카운트에 따라 색상(그라데이션)을 달리하여 반환.
    """
    if count >= 1000:
        return (255, 0, 255)

    color_index = count // 100
    if color_index >= len(color_sequence):
        color_index = len(color_sequence) - 1

    step_in_block = (count % 100) // 10
    intensity = 1.0
    #10마다 밝아지고 싶으면 이거 쓰기
    # intensity = intensity_levels[step_in_block] if step_in_block < len(intensity_levels) else 1.0
    base_color = color_sequence[color_index]

    color_bgr = np.uint8([[base_color]])
    color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]

    color_hsv = color_hsv.astype(float)
    color_hsv[2] = min(color_hsv[2] * intensity, 255)
    color_hsv = color_hsv.astype(np.uint8)

    intense_color = cv2.cvtColor(np.uint8([[color_hsv]]), cv2.COLOR_HSV2BGR)[0][0]
    intense_color_rgb = (intense_color[2], intense_color[1], intense_color[0])
    return intense_color_rgb


# =============================================================================
# 드래그/리사이즈 가능한 빨간 사각형 관련 전역 변수
# =============================================================================
drag_rect_x, drag_rect_y = 0, 0
drag_rect_w, drag_rect_h = 640, 200
dragging = False
resizing_corner = None
drag_offset_x, drag_offset_y = 0, 0
corner_size = 10

# =============================================================================
# 사각형 내부에서 공이 감지된 시간을 실시간으로 표시하기 위한 변수
# =============================================================================
ball_in_rect_start = None
in_rect_time = 0.0

ball_missing_frames = 0
MISSING_FRAMES_THRESHOLD = 5
# ----------------------------------------------------------------------------------------
# (A) 우클릭 확대/복귀 기능 관련 전역 변수
# ----------------------------------------------------------------------------------------
enlarged_view = None

#8강 토너먼트 리셋 버튼(오른쪽 상단) 좌표 저장
# ----------------------------------------------------------------------------------------
reset_button_rect_tournament = (0, 0, 0, 0)  # (x1, y1, x2, y2)를 저장할 전역 변수


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
    global enlarged_view
    global current_camera_index, cap

    if event == cv2.EVENT_MOUSEMOVE:
        last_mouse_move_time = time.time()
        if not mouse_visible:
            user32.ShowCursor(True)
            mouse_visible = True

        if resizing_corner is not None:
            # 리사이즈 중
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
            # 드래그 중
            new_x = x - drag_offset_x
            new_y = y - drag_offset_y
            new_x = max(0, min(new_x, 640 - drag_rect_w))
            new_y = max(0, min(new_y, 480 - drag_rect_h))
            drag_rect_x, drag_rect_y = new_x, new_y

    elif event == cv2.EVENT_LBUTTONDOWN:
        # ---------------------[2] 버튼 클릭 처리-----------------------
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

        # ---------------------[3] 토너먼트 RESET 버튼 클릭 여부-----------------------
        else:
            # 만약 Enlarged View가 없으면, 토너먼트 이미지는 Combined 창의 (x=640~1280, y=480~960) 구역
            # 따라서 'tournament_img' 내부에서의 로컬 좌표는 (x-640, y-480)
            # Enlarged View가 'br'이면, 토너먼트 이미지는 (x=0~1280, y=0~960)
            # 따라서 로컬 좌표는 그냥 (x, y)
            local_x, local_y = x, y
            if enlarged_view is None:
                # 4분할 상태
                if not (640 <= x < 1280 and 480 <= y < 960):
                    # 토너먼트 이미지 영역 밖
                    local_x, local_y = None, None
                else:
                    local_x = x - 640
                    local_y = y - 480
            else:
                # 확대뷰 상태
                if enlarged_view == 'br':
                    # 토너먼트 풀화면
                    # local_x, local_y = x, y (이미 할당되어 있으므로 그대로 둠)
                    pass
                else:
                    # 다른 확대뷰면 토너먼트가 안 보이므로 클릭 무시
                    local_x, local_y = None, None

            # 실제로 토너먼트 영역을 클릭한 경우에만
            if local_x is not None and local_y is not None:
                x1_r, y1_r, x2_r, y2_r = reset_button_rect_tournament
                # 만약 (local_x, local_y)가 (x1_r, y1_r, x2_r, y2_r) 안에 들어있다면 RESET 클릭
                if (x1_r <= local_x <= x2_r) and (y1_r <= local_y <= y2_r):
                    print("Tournament RESET Button Clicked!")

                    # [★ 토너먼트 다시 시작: 초기화 작업들]
                    draw_tournament_img_unified.final_ended = False
                    draw_tournament_img_unified.final_bracket = None
                    bounce_history.clear()

                    # 원하면 bounce_count나 상태도 초기화
                    # (예: current_state = "waiting")
                    # (예: bounce_count = 0)
                    # ...

        # ---------------------[4] 사각형(드래그/리사이즈) 클릭 처리-----------------------
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
            if (drag_rect_x <= x < drag_rect_x + drag_rect_w and
                drag_rect_y <= y < drag_rect_y + drag_rect_h):
                dragging = True
                drag_offset_x = x - drag_rect_x
                drag_offset_y = y - drag_rect_y

        # ---------------------[5] 웹캠 버튼(6개) 클릭 처리-----------------------
        for (rx1, ry1, rx2, ry2, cam_idx) in webcam_button_rects:
            if (rx1 <= x - 640 <= rx2) and (ry1 <= y <= ry2):
                print(f"Webcam button {cam_idx} clicked!")
                if current_camera_index != cam_idx:
                    cap.release()
                    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    if cap.isOpened():
                        print(f"Switched to webcam index {cam_idx} successfully.")
                        current_camera_index = cam_idx
                    else:
                        print(f"Failed to open webcam index {cam_idx}.")
                break

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        resizing_corner = None

    elif event == cv2.EVENT_RBUTTONDOWN:
        # 우클릭 시 확대/복귀
        if enlarged_view is None:
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

    # IgnoreOg 버튼
    cv2.rectangle(
        graph_img,
        (button_rect_ignore[0], button_rect_ignore[1]),
        (button_rect_ignore[0] + button_rect_ignore[2], button_rect_ignore[1] + button_rect_ignore[3]),
        (120, 120, 120),
        -1
    )
    text_ignore = "IgnoreOg: ON" if ignore_zero_orange else "IgnoreOg: OFF"
    cv2.putText(
        graph_img,
        text_ignore,
        (button_rect_ignore[0] + 5, button_rect_ignore[1] + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )

    # 웹캠 버튼 그리기
    box_width = 40
    box_height = 40
    start_x = button_rect_ignore[0]
    start_y = button_rect_ignore[1] + 50
    draw_webcam_buttons(graph_img, start_x, start_y, box_width, box_height, margin=5)

    return graph_img


def draw_webcam_buttons(base_img, start_x, start_y, box_width, box_height, margin=5):
    global webcam_button_rects
    webcam_button_rects.clear()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_color = (255, 255, 255)
    box_color = (0, 0, 0)

    idx = 0
    for row in range(2):
        for col in range(3):
            x1 = start_x + col*(box_width + margin)
            y1 = start_y + row*(box_height + margin)
            x2 = x1 + box_width
            y2 = y1 + box_height

            cv2.rectangle(base_img, (x1, y1), (x2, y2), box_color, 2)

            text = str(idx)
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            tx = x1 + (box_width - tw)//2
            ty = y1 + (box_height + th)//2

            cv2.putText(base_img, text, (tx, ty), font, font_scale, text_color, thickness, cv2.LINE_AA)
            webcam_button_rects.append((x1, y1, x2, y2, idx))
            idx += 1


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


def draw_tournament_img_unified(bounce_history, width=640, height=480):
    """
    8강 → 4강 → 결승(2강) 토너먼트를 그려주는 함수.
    토너먼트가 '최종 우승자'까지 결정되면 그 이후로 들어오는 bounce_history는 무시하고,
    확정된 토너먼트 이미지를 계속 반환하도록 내부적으로 처리합니다.
    """
    
    global reset_button_rect_tournament  # 여기서 전역 변수에 접근

    # ----------------------------------------------------------------------------
    # [★ 추가] 함수 스코프 밖에서도 상태를 기억하기 위한 정적(Static) 변수 설정
    # ----------------------------------------------------------------------------
    # 파이썬 함수에는 정적 변수가 없지만, 아래처럼 hasattr를 활용해 보관할 수 있습니다.
    if not hasattr(draw_tournament_img_unified, "final_ended"):
        draw_tournament_img_unified.final_ended = False    # 이미 우승자가 결정되었는지 여부
    if not hasattr(draw_tournament_img_unified, "final_bracket"):
        draw_tournament_img_unified.final_bracket = None   # 확정된(우승자까지 나온) 토너먼트 이미지



    # ----------------------------------------------------------------------------
    # [A] 만약 이미 우승자가 결정된 상태라면, bounce_history가 바뀌어도 반영하지 않고
    #     그냥 기존 final_bracket 이미지를 그대로 반환해버린다.
    # ----------------------------------------------------------------------------
    if draw_tournament_img_unified.final_ended:
        # 이미 결승까지 끝난 상태이므로, bounce_history가 추가되어도
        # 이전에 만든 '최종 결과' 이미지만 계속 보여준다.
        return draw_tournament_img_unified.final_bracket

    # --------------------------------------------------------------------------------
    # [B] 헬퍼 함수들 (draw_text_centered, draw_square_with_text, draw_row_of_squares)
    # --------------------------------------------------------------------------------
    def draw_text_centered(
        img,
        text,
        center_x,
        center_y,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.5,
        color=(255, 255, 255),
        thickness=1
    ):
        """
        전달받은 이미지(img)의 (center_x, center_y)에 텍스트를 중앙정렬로 그려주는 함수
        """
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_org_x = center_x - text_w // 2
        text_org_y = center_y + text_h // 2
        cv2.putText(img, text, (text_org_x, text_org_y), font, font_scale, color, thickness, cv2.LINE_AA)

    def draw_square_with_text(
        img,
        x,
        y,
        w,
        h,
        name_text,
        number_text,
        fill_color=None,
        border_color=(255, 255, 255),
        border_thickness=2,
        text_color=(255, 255, 255)
    ):
        """
        하나의 사각형을 그린 뒤, 상단에는 name_text, 하단에는 number_text를 그려주는 함수
        """
        # 내부 채우기
        if fill_color is not None:
            cv2.rectangle(img, (x, y), (x + w, y + h), fill_color, -1)

        # 테두리 그리기
        cv2.rectangle(img, (x, y), (x + w, y + h), border_color, border_thickness)

        # 이름 텍스트(상단)
        name_center_x = x + w // 2
        name_center_y = y + 20
        draw_text_centered(
            img,
            name_text,
            name_center_x,
            name_center_y,
            font_scale=0.5,
            color=text_color,
            thickness=1
        )

        # 번호 텍스트(하단)
        number_center_x = x + w // 2
        number_center_y = y + h - 20
        draw_text_centered(
            img,
            number_text,
            number_center_x,
            number_center_y,
            font_scale=1.0,
            color=text_color,
            thickness=2
        )

    def draw_row_of_squares(
        img,
        num_squares,
        square_width,
        square_height,
        margin,
        start_y,
        names,
        numbers,
        fill_colors=None
    ):
        """
        한 줄로 num_squares 개 사각형을 그린 후, 이미지 중앙정렬로 배치해주는 보조 함수
        """
        height_, width_, _ = img.shape
        total_width = num_squares * square_width + (num_squares - 1) * margin
        start_x = (width_ - total_width) // 2

        if fill_colors is None:
            fill_colors = [None] * num_squares

        for i in range(num_squares):
            x = start_x + i * (square_width + margin)
            y = start_y

            name_text = names[i] if i < len(names) else f"Name{i+1}"
            number_text = str(numbers[i]) if i < len(numbers) else "0"
            color = fill_colors[i] if i < len(fill_colors) else None

            draw_square_with_text(
                img,
                x,
                y,
                square_width,
                square_height,
                name_text,
                number_text,
                fill_color=color
            )

    # ----------------------------------------------------------------------------
    # [C] 실제 토너먼트용 사각형 8강/4강/결승을 그려주는 내부 보조 함수 (_draw_bracket_boxes)
    # ----------------------------------------------------------------------------
    def _draw_bracket_boxes(
        base_img,
        bottom_names, bottom_scores, bottom_colors,
        middle_names, middle_scores, middle_colors,
        top_names, top_scores, top_colors
    ):
        """
        base_img 위에 8강, 4강, 결승 정보를 사각형으로 그려주는 내부 함수
        """
        # 8강 (하단)
        bottom_y = base_img.shape[0] - 80 - 20
        draw_row_of_squares(
            base_img,
            num_squares=8,
            square_width=55,
            square_height=80,
            margin=20,
            start_y=bottom_y,
            names=bottom_names,
            numbers=bottom_scores,
            fill_colors=bottom_colors
        )

        # 4강 (중단)
        middle_y = (base_img.shape[0] // 2) - 40
        draw_row_of_squares(
            base_img,
            num_squares=4,
            square_width=55,
            square_height=80,
            margin=95,
            start_y=middle_y,
            names=middle_names,
            numbers=middle_scores,
            fill_colors=middle_colors
        )

        # 결승 (상단)
        top_y = 20
        draw_row_of_squares(
            base_img,
            num_squares=2,
            square_width=100,
            square_height=100,
            margin=200,
            start_y=top_y,
            names=top_names,
            numbers=top_scores,
            fill_colors=top_colors
        )

        return base_img

    # -----------------------
    # [D] 메인 로직
    # -----------------------

    # 색상 정의
    BLUE  = (255, 0, 0)   # BGR 형식
    RED   = (0, 0, 255)   # BGR 형식
    YELLOW = (0, 255, 255) # BGR 형식
    GREEN = (0, 255, 0)
    WHITE = (255, 255, 255)

    # 배경 이미지 생성
    graph_img = np.zeros((height, width, 3), dtype=np.uint8)

    # [D-1] 8강
    bottom_names  = ["A", "B", "C", "D", "E", "F", "G", "H"]
    bottom_scores = [0]*8
    bottom_colors = [None]*8

    eight_count = min(len(bounce_history), 8)
    for i in range(eight_count):
        bottom_scores[i] = bounce_history[i]

    winners_4 = ["", "", "", ""]

    for j in range(4):  # 4쌍
        i1 = 2*j
        i2 = 2*j + 1
        if i2 < eight_count:
            s1 = bottom_scores[i1]
            s2 = bottom_scores[i2]
            if s1 > s2:
                bottom_colors[i1] = BLUE
                bottom_colors[i2] = RED
                winners_4[j] = bottom_names[i1]
            elif s1 < s2:
                bottom_colors[i1] = RED
                bottom_colors[i2] = BLUE
                winners_4[j] = bottom_names[i2]
            else:
                # 동점 -> bounce_history 마지막 2개 제거하고 조기 리턴
                if len(bounce_history) >= 2:
                    bounce_history.pop()
                    bounce_history.pop()
                # 아직 결승 확정 전이므로 final_ended = False 유지
                return _draw_bracket_boxes(
                    graph_img,
                    bottom_names, bottom_scores, bottom_colors,
                    winners_4, [0, 0, 0, 0], [None]*4,
                    ["",""], [0, 0], [None, None]
                )

    # [D-2] 4강
    middle_names  = winners_4[:]
    middle_scores = [0, 0, 0, 0]
    middle_colors = [None, None, None, None]

    four_count = max(0, min(len(bounce_history) - 8, 4))
    for i in range(four_count):
        middle_scores[i] = bounce_history[8 + i]

    winners_2 = ["", ""]

    for j in range(2):  # 2쌍
        i1 = 2*j
        i2 = 2*j + 1
        if i2 < four_count:
            s1 = middle_scores[i1]
            s2 = middle_scores[i2]
            if s1 > s2:
                middle_colors[i1] = BLUE
                middle_colors[i2] = RED
                winners_2[j] = middle_names[i1]
            elif s1 < s2:
                middle_colors[i1] = RED
                middle_colors[i2] = BLUE
                winners_2[j] = middle_names[i2]
            else:
                # 동점 -> bounce_history 마지막 2개 제거하고 조기 리턴
                if len(bounce_history) >= 2:
                    bounce_history.pop()
                    bounce_history.pop()
                return _draw_bracket_boxes(
                    graph_img,
                    bottom_names, bottom_scores, bottom_colors,
                    middle_names, middle_scores, middle_colors,
                    ["",""], [0, 0], [None, None]
                )

    # [D-3] 결승(2강)
    top_names  = winners_2[:]
    top_scores = [0, 0]
    top_colors = [None, None]

    two_count = max(0, min(len(bounce_history) - 12, 2))
    for i in range(two_count):
        top_scores[i] = bounce_history[12 + i]

    if two_count == 2:
        if top_scores[0] > top_scores[1]:
            top_colors[0] = GREEN
            top_colors[1] = RED

            # [★ 추가] 우승자 이름
            winner_name = top_names[0]

            threading.Timer(2.0, final_win_sound.play).start()

            # 사운드 길이(초) 구해서, 우승자 효과 bgm 끝난 뒤 TTS 실행
            threading.Timer(2.0 + final_win_sound.get_length(), speak_winner, args=[winner_name]).start()

            # (추가) 최종 우승자까지 결정됨
            draw_tournament_img_unified.final_ended = True
        elif top_scores[0] < top_scores[1]:
            top_colors[0] = RED
            top_colors[1] = GREEN
            
            # 우승자 이름
            winner_name = top_names[1]

            threading.Timer(2.0, final_win_sound.play).start()

            # 사운드 길이(초) 구해서, 우승자 효과 bgm 끝난 뒤 TTS 실행
            threading.Timer(2.0 + final_win_sound.get_length(), speak_winner, args=[winner_name]).start()

            # (추가) 최종 우승자까지 결정됨
            draw_tournament_img_unified.final_ended = True
        else:
            # 동점 -> bounce_history 마지막 2개 제거하고 조기 리턴
            if len(bounce_history) >= 2:
                bounce_history.pop()
                bounce_history.pop()
            return _draw_bracket_boxes(
                graph_img,
                bottom_names, bottom_scores, bottom_colors,
                middle_names, middle_scores, middle_colors,
                top_names, top_scores, top_colors
            )

    # 여기까지 정상적으로 내려왔다면, 결승까지 점수 반영이 완료된 상태
    # 만약 두_count == 2라면 우승자 결정 (final_ended = True)
    #  => 이후부터 bounce_history 늘어나도 다음 호출 때는 frozen 결과만 반환

    # 최종 토너먼트 이미지를 만든다.
    final_bracket_img = _draw_bracket_boxes(
        graph_img,
        bottom_names, bottom_scores, bottom_colors,
        middle_names, middle_scores, middle_colors,
        top_names, top_scores, top_colors
    )

    # ------------------[★ RESET 버튼 그리기!]-------------------------
    # 여기서 (x, y)는 토너먼트 이미지 내부 좌표. (width=640, height=480)
    # 상단 오른쪽 구석에 100×40 크기로 그려 보자.
    btn_w, btn_h = 100, 40
    x1 = width - btn_w - 10  # 오른쪽에서 10픽셀 여백
    y1 = 10                  # 위쪽에서 10픽셀 여백
    x2 = x1 + btn_w
    y2 = y1 + btn_h
    cv2.rectangle(final_bracket_img, (x1, y1), (x2, y2), (100, 100, 100), -1)  # 버튼 배경(회색)
    cv2.putText(
        final_bracket_img,
        "RESET",
        (x1 + 10, y1 + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # 전역 변수에 저장: 나중에 mouse_callback에서 클릭 여부를 확인하기 위함.
    reset_button_rect_tournament = (x1, y1, x2, y2)

    # ------------------[★ 우승자 결정 시 처리]-------------------------
    if draw_tournament_img_unified.final_ended:
        draw_tournament_img_unified.final_bracket = final_bracket_img.copy()

    return final_bracket_img



# ----------------------------------------------------------------------------------------
# 13) Combined, Bounce Count 창 생성 & Combined 창을 전체화면으로 시작
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
stationary_threshold = 1.0
movement_threshold = 5
last_position = None
previous_bounce_time = None

last_detection_time = None

bounce_time_diff = None

# (추가) 토너먼트 표시용 바운스 기록(=각 사람의 점수). 최대 14개면 8강+4강+2강 모두 가능.
bounce_history = []

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
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    current_time = time.time()
    time_diff = current_time - prev_time
    if time_diff > 1e-9:
        fps = 1.0 / time_diff
    prev_time = current_time

    results = model.predict(frame, imgsz=640, conf=0.3, max_det=1, show=False, device=0)
    boxes = results[0].boxes

    x_values.append(frame_count)
    frame_count += 1

    detected = False
    orange_pixels = 0

    if len(boxes) > 0:
        x1, y1, x2, y2 = boxes[0].xyxy[0].cpu().numpy()
        y_center = (y1 + y2) / 2.0
        x_center = (x1 + x2) / 2.0

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
            if orange_pixels >= 50:
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
                    if in_rect_time >= stationary_threshold and current_state != "ready":
                        current_state = "ready"
                        ready_sound.play()
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
                                    if bounce_count in hundred_unit_sounds: #100,200,300,400,500,600,700,800,900,1000
                                        hundred_unit_sounds[bounce_count].play()
                                    elif bounce_count % 100 == 0: #1100이후로 1200,1300,1400,1500...
                                        score_sound.play()
                                    elif bounce_count % 10 == 0: #10,20,30,40,50,60,70,80,90,100
                                        collect_points_sound.play()
                                    else: #1,2,3,4,5,6,7,8,9..
                                        bounce_count_sound.play()

                                bounce_points.append((x_values[-1], y_values[-1]))
                                current_bounce_time = time.time()
                                bounce_times.append(current_bounce_time)

                                if previous_bounce_time is not None:
                                    td = current_bounce_time - previous_bounce_time
                                    print(f"Time diff between last two bounces: {td:.2f} s")
                                    bounce_time_diff = td

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

    # ------------------------------
    # (수정 전역 변수) ball_missing_frames = 0
    # ------------------------------
    # 공이 감지됨(boxes가 있음) + 오렌지 픽셀도 충분히 있음(detected=True)라면
    if len(boxes) > 0 and detected:
        # 공의 중심이 사각형 내부인가?
        if (drag_rect_x <= x_center < drag_rect_x + drag_rect_w and
            drag_rect_y <= y_center < drag_rect_y + drag_rect_h):
            # 만약 처음으로 들어온 경우라면 시작 시간 기록
            if ball_in_rect_start is None:
                ball_in_rect_start = time.time()
                # 참고: tracking 상태가 아니라면 tap_notification_sound 등 재생
                if current_state != "tracking":
                    tap_notification_sound.play()

            # in_rect_time 갱신
            in_rect_time = time.time() - ball_in_rect_start if ball_in_rect_start else 0.0

            # ★ 공이 정상적으로 사각형 안에서 감지되었으므로 missing_frames 리셋
            ball_missing_frames = 0

        else:
            # ★ 사각형 내부가 아닌곳에 있으면 프레임 지연없이 바로 초기화
                in_rect_time = 0.0
                ball_in_rect_start = None

    # (박스가 없거나, 오렌지픽셀 조건 미달)
    else:
        # ★ 이번 프레임에서는 공이 전혀 감지되지 않음 -> missing_frames 증가
        ball_missing_frames += 1
        if ball_missing_frames >= MISSING_FRAMES_THRESHOLD:
            in_rect_time = 0.0
            ball_in_rect_start = None


    # ready 상태에서 공이 안 보이면 waiting으로
    if current_state == "ready":
        if last_detection_time is not None and (time.time() - last_detection_time) > 1.0:
            current_state = "waiting"
            print("State changed to WAITING (no detection for 1s in READY)")

    # tracking 중에 공 안 보이면 -> waiting
    if current_state == "tracking":
        if last_detection_time is not None and (time.time() - last_detection_time) >= 1.0: #1초 이상 공이 안보일때
            
            if bounce_count > 0:
                bounce_history.append(bounce_count)

                if len(bounce_history) > 14:
                    bounce_history.pop(0)
            bounce_count = 0
            consecutiveDownCount = 0
            consecutiveUpCount = 0
            state = None
            current_state = "waiting"
            # alert_sound.play() # 공이 사라졌을때 효과음
            print("No detection for 1 second in TRACKING => bounce_count reset to 0, state changed to WAITING")

    # 그래프 데이터 길이 제한
    if len(x_values) > MAX_POINTS:
        x_values.pop(0)
        y_values.pop(0)
        orange_pixel_values.pop(0)

    # 바운스 간 일정 시간 지나면 초기화 (옵션)
    if current_bounce_time is not None:
        if time.time() - current_bounce_time > CONTINUOUS_TIMEOUT:
            if bounce_count > 0:
                bounce_history.append(bounce_count)
            if len(bounce_history) > 14:
                bounce_history.pop(0)
            
            bounce_count = 0
            current_state = "waiting"
            alert_sound.play() # 공이 사라졌을때 효과음
            consecutiveDownCount = 0
            consecutiveUpCount = 0
            state = None
            current_bounce_time = None
            bounce_points = []
            bounce_times = []
            previous_bounce_time = None
            print("No bounce for a while -> reset bounce_count to 0")

    # ------------------ Combined 화면 구성 ------------------
    combined_img = np.zeros((960, 1280, 3), dtype=np.uint8)
    frame_resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

    # FPS, State, Bounce Dt 표시
    cv2.putText(
        frame_resized,
        f"CurrentState: {current_state.upper()}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )
    cv2.putText(
        frame_resized,
        f"FPS: {fps:.2f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
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
            (0, 0, 0),
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

    cv2.putText(
        frame_resized,
        f"In-Rect Time: {in_rect_time:.2f}s",
        (drag_rect_x + drag_rect_w - 300, drag_rect_y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    y_graph_img = draw_y_graph(
        x_values, y_values, width=640, height=480, max_y=480, bounce_pts=bounce_points
    )
    valid_orange = [v for v in orange_pixel_values if v is not None]
    max_orange = max(valid_orange) if valid_orange else 1
    orange_graph_img = draw_orange_graph(
        x_values, orange_pixel_values, width=640, height=480, max_y=max_orange
    )

    # tournament_img(토너먼트 표시: "매 경기마다 즉시 승부")
    tournament_img = draw_tournament_img_unified(bounce_history, width=640, height=480)

    # 확대뷰가 없으면 4분할
    if enlarged_view is None:
        combined_img[0:480, 0:640] = frame_resized
        combined_img[0:480, 640:1280] = y_graph_img
        combined_img[480:960, 0:640] = orange_graph_img
        combined_img[480:960, 640:1280] = tournament_img
    else:
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
            big_view = cv2.resize(tournament_img, (1280, 960), interpolation=cv2.INTER_AREA)
            combined_img = big_view

    cv2.imshow("Combined", combined_img)

    # 바운스 카운트 창
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
        is_fullscreen_combined = not is_fullscreen_combined
    elif key in [ord('b'), ord('B')]:
        if is_fullscreen_bounce:
            cv2.setWindowProperty("Bounce Count Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        else:
            cv2.setWindowProperty("Bounce Count Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        is_fullscreen_bounce = not is_fullscreen_bounce

# ----------------------------------------------------------------------------------------
# 16) 종료 처리
# ----------------------------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
