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
import random
import sys
from PyQt5.QtWidgets import (QApplication, QDialog, QVBoxLayout,
                             QLabel, QLineEdit, QPushButton)

def pyqt_text_input_mode(prompt="8강 이름을 입력 (띄어쓰기 구분)"):
    """
    PyQt 다이얼로그를 모달로 띄워서, 사용자에게 문자열 입력을 받는 간단한 예시 함수.
    - return: 사용자가 OK를 누르거나 엔터를 누를 때의 입력 문자열
              (취소 시에는 빈 문자열 "")
    """
    app = QApplication(sys.argv)  # PyQt 앱 생성

    dialog = QDialog()
    dialog.setWindowTitle("Enter 8 Names")
    dialog.resize(400, 150)

    layout = QVBoxLayout(dialog)

    label = QLabel(prompt, dialog)
    layout.addWidget(label)

    line_edit = QLineEdit(dialog)
    layout.addWidget(line_edit)

    button_ok = QPushButton("OK", dialog)
    layout.addWidget(button_ok)

    def on_ok_clicked():
        dialog.accept()  # 다이얼로그 닫고 accept() 상태로 종료

    button_ok.clicked.connect(on_ok_clicked)

    # 엔터로도 OK 누른 것처럼 처리
    line_edit.returnPressed.connect(on_ok_clicked)

    dialog.setLayout(layout)

    # 모달 실행 -> user가 닫거나 OK누를 때까지 블록
    result = dialog.exec_()
    if result == QDialog.Accepted:
        return line_edit.text().strip()
    else:
        return ""  # 취소/닫기 시 빈 문자열

#마우스 커서 숨기는데 관여하는 변수
user32 = ctypes.windll.user32

#초기화 함수
def reset_all_states():
    global bounce_count
    global bounce_points, bounce_times
    global previous_bounce_time, state
    global consecutiveDownCount, consecutiveUpCount
    global ball_in_rect_start, in_rect_time
    global current_state
    global x_values, y_values, orange_pixel_values
    global single_mode_state, active_jokers, chosen_jokers
    global current_stage, total_score, bounce_sequence
    global just_cleared_stage
    global drag_rect_x, drag_rect_y, drag_rect_w, drag_rect_h
    global bounce_history
    global prev_bounce_count, prev_total_score
    global score_color, bounce_type
    global BOUNCE_TYPE_WEIGHTS


    # 1) 바운스, 그래프 관련
    bounce_count = 0
    bounce_points.clear()
    bounce_times.clear()
    previous_bounce_time = None
    state = None
    consecutiveDownCount = 0
    consecutiveUpCount = 0

    # 2) 사각형 & in_rect_time
    ball_in_rect_start = None
    in_rect_time = 0.0
    drag_rect_x = 0
    drag_rect_y = 0
    drag_rect_w = 640
    drag_rect_h = 200

    # 3) 현재 상태 (waiting 등)
    current_state = "waiting"

    # 4) 그래프 데이터 (원하면 clear)
    x_values.clear()
    y_values.clear()
    orange_pixel_values.clear()

    # 5) 싱글 모드 변수들
    single_mode_state = "playing"
    active_jokers.clear()
    chosen_jokers.clear()
    current_stage = 1
    total_score = 0
    bounce_sequence.clear()
    just_cleared_stage = False
    prev_total_score = -1
    bounce_type = None
    score_color = (255,255,255) #흰색
    BOUNCE_TYPE_WEIGHTS = {
        "LOW": 1,
        "MIDDLE": 1,
        "HIGH": 5,
        "SUPER": 10
    }

    # 6) 토너먼트 기록
    bounce_history.clear()
    prev_bounce_count = -1
    draw_tournament_img_unified.previous_bounce_history_len = -1  # 이전 기록 초기화
    print("All states have been reset!")


# ----------------------------------------------------------------------------------------
# 1) pygame 오디오 초기화 및 사운드 로드
# ----------------------------------------------------------------------------------------
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=128)
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
#스테이지 클리어 효과음
stage_clear_sound = pygame.mixer.Sound(r"clear.mp3")
#조커 효과 발동 효과음
joker_sound = pygame.mixer.Sound(r"joker.mp3")

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
rel_stage_clear = 1.0

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
    stage_clear_sound.set_volume(master_volume * rel_stage_clear)

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

def on_trackbar_stage_clear(val):
    global rel_stage_clear
    rel_stage_clear = val / 100.0
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



# cv2.namedWindow("Camera Control")
# # 초기값은 카메라에서 읽어오거나(지원할 경우) 기본 50%로 설정
# init_brightness = int(cap.get(cv2.CAP_PROP_BRIGHTNESS) * 100) if cap.get(cv2.CAP_PROP_BRIGHTNESS) != 0 else 50
# init_contrast = int(cap.get(cv2.CAP_PROP_CONTRAST) * 100) if cap.get(cv2.CAP_PROP_CONTRAST) != 0 else 50
# init_saturation = int(cap.get(cv2.CAP_PROP_SATURATION) * 100) if cap.get(cv2.CAP_PROP_SATURATION) != 0 else 50




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
            # user32.ShowCursor(True) #하고싶을때 조절
            mouse_visible = True


cv2.setMouseCallback("Volume Control", mouse_callback_volume)






# 2) YOLO 모델 로드
# ----------------------------------------------------------------------------------------
model = YOLO(r"Ping-Pong-Detection-3-best.pt")


# ----------------------------------------------------------------------------------------
# 3) 카메라 디바이스 연결
# ----------------------------------------------------------------------------------------
cap = cv2.VideoCapture(3, cv2.CAP_DSHOW) #p눌러서 변경 가능 #FHD60F는 CAP_DSHOW, logitech은 CAP_MSMF와 호환 최상위
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# 예: 코드 상단(전역)에 다음 딕셔너리 정의
api_pref_map = {
    "MSMF": cv2.CAP_MSMF,
    "DSHOW": cv2.CAP_DSHOW,
    "V4L2": cv2.CAP_V4L2,
    "ANY": cv2.CAP_ANY,  # 필요한 항목 추가
}



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
DOWN_THRESHOLD = 1
UP_THRESHOLD = 1
PIXEL_THRESHOLD = 2.0
last_y = None

bounce_points = []
bounce_times = []

CONTINUOUS_TIMEOUT = 1.5
current_bounce_time = None

sound_enabled = True
ignore_zero_orange = True 

button_rect = [500, 20, 120, 40]
button_rect_ignore = [500, 70, 120, 40]

# 웹캠 선택을 위한 전역 변수
current_camera_index = 0
webcam_button_rects = []
# [추가] 밝기 조절을 위한 전역 변수 (기본: 1.0 → 원본 밝기)
brightness_alpha = 1.0  # 1.0이면 원본, 0.5면 어둡게, 1.5면 밝게

#점수 표시
DIGITAL_NUMBER_FONT_PATH = r"C:\Users\omyra\Desktop\coding\ping_pong\Digital Display.ttf"
HANDWRITING_FONT_PATH = r"SpoqaHanSansNeo_TTF_original\SpoqaHanSansNeo-Medium.ttf"
KOREAN_FONT_PATH = r"나눔손글씨_배은혜체.ttf"
FONT_SIZE = 400

digital_font = ImageFont.truetype(DIGITAL_NUMBER_FONT_PATH, FONT_SIZE)
handwriting_font = ImageFont.truetype(HANDWRITING_FONT_PATH, 350)
korean_font = ImageFont.truetype(KOREAN_FONT_PATH, 30)

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
    global single_mode_state, chosen_jokers, active_jokers

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
                    draw_tournament_img_unified.previous_bounce_history_len = -1  # 이전 기록 초기화
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
    
    # ↓ 우하단 single_player_img 영역 클릭 처리
    if 640 <= x < 1280 and 480 <= y < 960:
        local_x = x - 640
        local_y = y - 480

        if single_mode_state == "choosing_joker" and event == cv2.EVENT_LBUTTONDOWN:
            # 첫 번째 박스 클릭
            if (50 <= local_x <= 240) and (200 <= local_y <= 400):
                active_jokers.append(chosen_jokers[0]["id"])
                single_mode_state = "playing"
                print(f"Chose Joker: {chosen_jokers[0]['id']}")
                chosen_jokers = []  # ★ 이 위치에서만 실행 (실제로 첫 번째 박스를 골랐을 때)
                drag_rect_x, drag_rect_y, drag_rect_w, drag_rect_h = 0, 0, 640, 200 #조커 선택후 다시 준비할 수 있게 영역 만들기

            # 두 번째 박스 클릭
            elif (350 <= local_x <= 540) and (200 <= local_y <= 400):
                active_jokers.append(chosen_jokers[1]["id"])
                single_mode_state = "playing"
                print(f"Chose Joker: {chosen_jokers[1]['id']}")
                chosen_jokers = []  # ★ 이 위치에서만 실행 (실제로 두 번째 박스를 골랐을 때)
                drag_rect_x, drag_rect_y, drag_rect_w, drag_rect_h = 0, 0, 640, 200 #조커 선택후 다시 준비할 수 있게 영역 만들기

            # ※ else: 박스 범위 밖 클릭 시에는 아무 것도 안 하고 return (혹은 pass)





# ----------------------------------------------------------------------------------------
# 10) render_text_with_ttf()
# ----------------------------------------------------------------------------------------
def render_text_with_ttf(
    text,
    font,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
    width=1920,
    height=1080
):
    # 1) 배경용 Pillow 이미지 생성
    img_pil = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img_pil)

    # 2) 텍스트 바운딩박스 구하기
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w = right - left
    text_h = bottom - top

    # 3) 바운딩박스를 기준으로 정확히 가운데 오도록 위치 계산
    #   - 가운데 정렬 지점에서 (left, top)을 빼서 보정
    text_x = (width - text_w) // 2 - left
    text_y = (height - text_h) // 2 - top

    # 4) 텍스트 그리기
    draw.text((text_x, text_y), text, font=font, fill=text_color)

    # 5) NumPy 배열로 변환 후 OpenCV용 BGR로 변경
    img_np = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_bgr

def render_text_with_ttf_segments(
    text_segments,
    base_font_path,
    bg_color=(0, 0, 0),
    width=1920,
    height=1080,
    default_size=40
):
    """
    여러 텍스트 세그먼트를 렌더링하되,
    1) 첫 번째 텍스트는 화면 '상단 중앙'에 배치
    2) 나머지 텍스트들은 기존 코드처럼 화면 '정중앙'에 이어서 배치
    3) 각 텍스트마다 (텍스트, 색상, 폰트크기)로 크기를 다르게 적용 가능

    :param text_segments: 
        예) [
          ("Stage1", (255,255,255), 60),
          ("100", (0,255,0), 120),
          ("/300", (255,0,0), 80),
        ]
    :param base_font_path: 사용할 폰트 경로 (예: "SpoqaHanSansNeo-Medium.ttf")
    :param bg_color: 배경색 (기본값: 검정)
    :param width, height: 출력 이미지 크기
    :param default_size: size가 주어지지 않았을 때 사용할 폰트 크기 (기본 40)
    :return: OpenCV BGR 이미지 (numpy 배열)
    """

    # 1) 배경 이미지 & Draw 객체
    img_pil = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img_pil)

    # 2) 세그먼트가 비어 있으면 그대로 반환
    if not text_segments:
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 3) 첫 번째 텍스트/색상/크기 + 나머지 분리
    first_text, first_color, *maybe_size = text_segments[0]
    first_size = maybe_size[0] if maybe_size else default_size
    rest_segments = text_segments[1:]  # 두 번째 이후 텍스트들

    # ----------------------------------------------------------------------------
    # A. 첫 번째 텍스트 (상단 중앙 배치)
    # ----------------------------------------------------------------------------
    font_first = ImageFont.truetype(base_font_path, size=first_size)
    bbox_first = draw.textbbox((0, 0), first_text, font=font_first)
    text_w_first = bbox_first[2] - bbox_first[0]
    text_h_first = bbox_first[3] - bbox_first[1]

    offset_top = 10  # 상단에서부터 얼마나 떨어뜨릴지
    first_x = 20
    first_y = offset_top

    # PIL이 RGB 순서이므로, fill=(R, G, B) 형태로 전달
    draw.text(
        (first_x, first_y),
        first_text,
        font=font_first,
        fill=(first_color[2], first_color[1], first_color[0])
    )

    # ----------------------------------------------------------------------------
    # B. 나머지 텍스트 (가운데 정렬로 이어붙이기)
    # ----------------------------------------------------------------------------
    if rest_segments:
        # (B1) 전체 너비 및 최대 높이 계산
        total_width = 0
        max_height = 0
        # 각 세그먼트별로 폰트를 그때그때 생성하여 측정
        segment_sizes = []
        for (text, color, *maybe_sz) in rest_segments:
            sz = maybe_sz[0] if maybe_sz else default_size
            fnt = ImageFont.truetype(base_font_path, size=sz)

            bbox = draw.textbbox((0, 0), text, font=fnt)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            total_width += text_w
            max_height = max(max_height, text_h)
            segment_sizes.append((fnt, text_w, text_h))

        # (B2) 중앙 정렬 시작 위치 (현재는 x축: 중앙-50 정도에서 시작)
        current_x = (width - total_width) // 2 + 10
        current_y = (height - max_height) // 2 - 40

        # (B3) 각 텍스트 렌더링
        for i, (seg) in enumerate(rest_segments):
            text, color, *maybe_sz = seg
            sz = maybe_sz[0] if maybe_sz else default_size
            font_this_seg, text_w, text_h = segment_sizes[i]

            # 실제 그리기
            draw.text(
                (current_x, current_y),
                text,
                font=font_this_seg,
                fill=(color[2], color[1], color[0])
            )
            current_x += text_w

    # 최종 PIL → NumPy → BGR 변환 후 반환
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

    # 여기서 bounce_pts를 순회하며 빨간 점 + 'LOW/MID/HIGH' 텍스트 표시
    # -----------------------------
    for (bx_ori, by_ori, btd) in bounce_pts:
        if bx_ori < x_data[0]:
            continue

        # 그래프 좌표계로 매핑
        bx = int((bx_ori - x_data[0]) / (max_x - x_data[0] + 1e-6) * (width - 1))
        by = int(by_ori / max_y * (height - 1))

        # 빨간 점
        cv2.circle(graph_img, (bx, by), 5, (0, 0, 255), -1)

        # 바운스 간격에 따른 레이블
        if btd is not None:
            if btd <= BOUNCE_THRESHOLDS["LOW"]:
                label = "LOW"
                color = BOUNCE_TYPE_COLORS["LOW"] #회색
            elif btd < BOUNCE_THRESHOLDS["MIDDLE"]:
                label = "MIDDLE"
                color = BOUNCE_TYPE_COLORS["MIDDLE"] #주황색
            elif btd < BOUNCE_THRESHOLDS["HIGH"]:
                label = "HIGH"
                color = BOUNCE_TYPE_COLORS["HIGH"] #초록색
            else:
                label = "SUPER"
                color = BOUNCE_TYPE_COLORS["SUPER"] #보라색
            # 빨간 점 위로 텍스트 표시
            cv2.putText(
                graph_img,
                label,
                (bx + 5, by - 5),  # 점에서 약간 오른쪽/위로
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA
            )

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
    if not hasattr(draw_tournament_img_unified, "previous_bounce_history_len"):
        draw_tournament_img_unified.previous_bounce_history_len = -1  # 초기값 설정



    # ----------------------------------------------------------------------------
    # [A] 만약 이미 우승자가 결정된 상태라면, bounce_history가 바뀌어도 반영하지 않고
    #     그냥 기존 final_bracket 이미지를 그대로 반환해버린다.
    # ----------------------------------------------------------------------------
    if draw_tournament_img_unified.final_ended:
        # 이미 결승까지 끝난 상태이므로, bounce_history가 추가되어도
        # 이전에 만든 '최종 결과' 이미지만 계속 보여준다.
        return draw_tournament_img_unified.final_bracket

    # --------------------------------------------------------------------------------
    # [B] 한글 표기 토너먼트 참가자 이름 함수 (draw_text_centered, draw_square_with_text, draw_row_of_squares)
    # --------------------------------------------------------------------------------
    # ... existing code ...
    def draw_pil_text_centered(
        img,
        text,   #이름이 여기로 들어옴.
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
        # OpenCV 이미지를 PIL 이미지로 변환
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        
        # 텍스트 크기 측정
        bbox = draw.textbbox((0, 0), text, font=korean_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        # 중앙 정렬 위치 계산
        text_x = center_x - text_w // 2
        text_y = center_y - text_h // 2
        
        #  참가자 이름 텍스트 그리기 pil이용 대신 프레임 떨어져서 프레임마다 호출하면 안되고 값바뀔때만 호출
        draw.text((text_x, text_y - 10), text, font=korean_font, fill=color[::-1])  # RGB -> BGR
        
        # PIL 이미지를 다시 OpenCV 이미지로 변환 '바로' 주어진 이미지를 변환
        cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR, dst=img)
# ... existing code ...
 
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
        draw_pil_text_centered(
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
    # 이제 bottom_names 대신 tournament_players_8을 사용
    bottom_names = tournament_players_8 
    bottom_scores = [0]*8
    bottom_colors = [None]*8

    eight_count = min(len(bounce_history), 8)
    for i in range(eight_count):
        bottom_scores[i] = bounce_history[i]

    winners_4 = ["", "", "", ""]

    # [D-1] 8강
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
                
                # 동점 처리 - bounce_history는 수정하지 않고, 현재 상태만 표시
                bottom_scores[i1] = 0  
                bottom_scores[i2] = 0  

                winners_4[j] = ""  # 승자 미정

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

                # 동점 처리
                middle_scores[i1] = 0
                middle_scores[i2] = 0

                winners_2[j] = ""

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

            # 동점 발생 시, 해당 플레이어들의 점수를 0으로 설정
            top_scores[0] = 0
            top_scores[1] = 0


            # 동점이므로 승자 처리 로직은 실행하지 않음

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

# 토너먼트 모드 표시용 바운스 기록(=각 사람의 점수). 최대 14개면 8강+4강+2강 모두 가능.
bounce_history = []
tournament_players_8 = ['광운', '광조', '현조', '광조', '현조', '광운', '광조', '현조']
#8강 토너먼트 리셋 버튼(오른쪽 상단) 좌표 저장
# ----------------------------------------------------------------------------------------
reset_button_rect_tournament = (0, 0, 0, 0)  # (x1, y1, x2, y2)를 저장할 전역 변수
# 토너먼트 메인 루프 안에서 초기 그리고 bounce_history가 바뀔때마다만 렌더링 함으로써 PIL이미지 시각화를 필요할때만 함으로서 프레임 부하 줄인다.
# 루프 시작 전에 초기 tournament_img 생성
tournament_img = draw_tournament_img_unified(bounce_history, width=640, height=480)
draw_tournament_img_unified.previous_bounce_history_len = len(bounce_history)

# ----------------------------------------------------------------------------------------
# 15) 메인 루프
# ----------------------------------------------------------------------------------------
prev_time = time.time()
fps = 0.0
###########################3
fps_counter = 0
fps_start_time = time.time()
fps_display = 0.0

#############################
#싱글 플레이어 모드용 전역 변수
#############################
# ----------------------------------------------------------------------------------------
# 바운스 기준값을 위한 전역 구성
# ----------------------------------------------------------------------------------------
play_mode = "tournament"  # 토너먼트 모드시작 플래그
BOUNCE_THRESHOLDS = {
    "LOW": 0.33,
    "MIDDLE": 0.58,
    "HIGH": 0.80
    # "SUPER"는 HIGH 기준값 이상을 의미하므로 별도로 정의할 필요 없음
}

# 바운스 유형에 따라 가중치 정의
BOUNCE_TYPE_WEIGHTS = {
    "LOW": 1,
    "MIDDLE": 1,
    "HIGH": 5,
    "SUPER": 10
}

#BGR기준
BOUNCE_TYPE_COLORS = {
    "LOW": (200, 200, 200),  # 회색,
    "MIDDLE": (0, 102, 255),  # 주황색,
    "HIGH": (0, 255, 0),  # 초록색,
    "SUPER": (255, 0, 255)  # 보라색
}


# -----------------------
#전체 조커 후보 정의 (원하는 만큼)
all_jokers = [
    #직관적
    {
        "id": "JOKER_LOW_1",
        "title": "JOKER_LOW_1",
        "desc": "LOW 1x => +1"
    }, {
        "id": "JOKER_MIDDLE_1", 
        "title": "JOKER_MIDDLE_1",
        "desc": "MIDDLE 1x => +2"
    }, {
        "id": "JOKER_HIGH_1",
        "title": "JOKER_HIGH_1", 
        "desc": "HIGH 1x => +3"
    }, {
        "id": "JOKER_SUPER_1",
        "title": "JOKER_SUPER_1",
        "desc": "SUPER 1x => +5"
    }
    # Already used 3 examples
    # {
    #     "id": "JOKER_LOW_10",
    #     "title": "JOKER_LOW_10", 
    #     "desc": "LOW 10x => +10"
    # },
    # {
    #     "id": "JOKER_LMHS_50",
    #     "title": "JOKER_LMHS_50",
    #     "desc": "L->M->H->S => +50"
    # },
    # {
    #     "id": "JOKER_SUPER_3",
    #     "title": "JOKER_SUPER_3",
    #     "desc": "SUPER 3x => +30"
    # },

    # # --------------------
    # # 4 new examples added below
    # # --------------------
    # , 

]

# '스테이지 클리어후 선택지 2개 조커'를 저장할 전역 변수
chosen_jokers = []
active_jokers = []         # 현재 내가 가진 활성화된 조커들 (예: ["JOKER_LOW_10", "JOKER_LMHS_50"])


bounce_sequence = []  # 바운스 타입("LOW"/"MIDDLE"/"HIGH"/"SUPER")을 순서대로 기록
current_stage = 1
stage_thresholds = [5, 10, 30, 50, 100, 300, 500, 1000, 2000, 3000] #스테이지 클리어 조건 
single_mode_state = "playing"   # 또는 "choosing_joker"
just_cleared_stage = False  # ★ 추가: 스테이지 클리어 직후 alert_sound를 막기 위한 플래그
total_score = 0    # 조커 효과로 얻는 추가 점수만 누적
#싱글 모드 점수 변경 여부를 체크할 변수
prev_total_score = None  # total_score와 비교하기 위함 (매 프레임마다 pil로 렌더링하면 부하가 많이옴, 그래서 점수가 바뀔때만 렌더링 진행)

# 새로 만드는 전역 카운터 예시
count_any = 0              # 아무 바운스 유형이든 누적 횟수
count_low = 0
count_mid = 0
count_high = 0
count_super = 0

consecutive_low = 0
consecutive_mid = 0
consecutive_high = 0
consecutive_super = 0

no_fail_streak = 0         # 연속으로 공을 놓치지 않고 성공한 바운스 횟수

def handle_single_mode_stage():
    global bounce_count, current_stage, current_state
    global drag_rect_x, drag_rect_y, drag_rect_w, drag_rect_h
    global just_cleared_stage
    global single_mode_state
    global total_score, bounce_sequence
    global score_color
    global enlarged_view

    if current_stage > len(stage_thresholds):
        return  # 이미 모든 스테이지 클리어

    target = stage_thresholds[current_stage - 1]

    # ---- "조커 점수"가 목표를 넘으면 클리어 ----
    if total_score >= target:
        print(f"[싱글모드] Stage {current_stage} Clear! (목표={target}, 점수={total_score})")

        stage_clear_sound.play()

        enlarged_view = 'br' #클리어하고 나서 진행상황보도록.



        # 1) 바운스 카운트, 조커 점수, 바운스 시퀀스 초기화
        bounce_count = 0
        total_score = 0
        bounce_sequence = []
        

        # 2) 다음 스테이지
        current_stage += 1

        # 3) 상태: waiting
        just_cleared_stage = True
        current_state = "waiting"


        ##############################
        #조커 선택창 활성화 시킬거면 이 주석 푸세요!!!!!!
        ##############################
        # # 조커 선택 모드로 전환
        # single_mode_state = "choosing_joker"


        # # #끝났으니 원상복구
        # drag_rect_x = 0
        # drag_rect_y = 0
        # drag_rect_w = 1
        # drag_rect_h = 1


def apply_jokers_on_bounce(bounce_type):
    """
    매 바운스가 일어날 때마다 호출되는 함수.
    - bounce_type: "LOW", "MIDDLE", "HIGH", "SUPER" 중 하나 (혹은 None)
    - 1) 기본 점수 +1
    - 2) 바운스 종류별 누적/연속 카운터 갱신
    - 3) active_jokers에 있는 조커들을 전부 검사하여 조건 충족 시 보상 부여(+점수) 후 제거 (조커적용)
    """

    global bounce_sequence
    global total_score, active_jokers
    global consecutive_low, consecutive_mid, consecutive_high, consecutive_super
    global count_any, count_low, count_mid, count_high, count_super
    global no_fail_streak

    
    # ---------------------------
    # 2) 누적/연속 카운터 갱신
    # ---------------------------
    # 전체 바운스(아무 유형)
    count_any += 1
    no_fail_streak += 1  # 공 놓치지 않았으므로 +1

    if bounce_type is not None:
        bounce_sequence.append(bounce_type)

    print(bounce_sequence)

    print(f"active jokers:{active_jokers}")

    # 연속/누적 카운터 업데이트
    if bounce_type == "LOW":
        count_low += 1
        consecutive_low += 1
        # 다른 유형 연속은 끊김
        consecutive_mid = 0
        consecutive_high = 0
        consecutive_super = 0
        print(f"연속 카운터 - LOW:{consecutive_low}, MID:{consecutive_mid}, HIGH:{consecutive_high}, SUPER:{consecutive_super}")
    elif bounce_type == "MIDDLE":
        count_mid += 1
        consecutive_mid += 1
        consecutive_low = 0
        consecutive_high = 0
        consecutive_super = 0
        print(f"연속 카운터 - LOW:{consecutive_low}, MID:{consecutive_mid}, HIGH:{consecutive_high}, SUPER:{consecutive_super}")
    elif bounce_type == "HIGH":
        count_high += 1
        consecutive_high += 1
        consecutive_low = 0
        consecutive_mid = 0
        consecutive_super = 0
        print(f"연속 카운터 - LOW:{consecutive_low}, MID:{consecutive_mid}, HIGH:{consecutive_high}, SUPER:{consecutive_super}")
    elif bounce_type == "SUPER":
        count_super += 1
        consecutive_super += 1
        consecutive_low = 0
        consecutive_mid = 0
        consecutive_high = 0
        print(f"연속 카운터 - LOW:{consecutive_low}, MID:{consecutive_mid}, HIGH:{consecutive_high}, SUPER:{consecutive_super}")
    else:
        # bounce_type이 None이라면 별도 처리는 생략
        pass


    # ---------------------------
    # 3) 조커 발동 체크
    # ---------------------------
    
    




    if "JOKER_LOW_1" in active_jokers:
        if bounce_type == "LOW":
            BOUNCE_TYPE_WEIGHTS["LOW"] += 1 #이 조커는 한번쓰면 영구적으로 WEIGHT계속 적용 고로 REMOVE해야함.
            active_jokers.remove("JOKER_LOW_1")
    if "JOKER_MIDDLE_1" in active_jokers:
        if bounce_type == "MIDDLE":
            BOUNCE_TYPE_WEIGHTS["MIDDLE"] += 2
            active_jokers.remove("JOKER_MIDDLE_1")  #이 조커는 한번쓰면 영구적으로 WEIGHT계속 적용 고로 REMOVE해야함.
    if "JOKER_HIGH_1" in active_jokers:
        if bounce_type == "HIGH":
            BOUNCE_TYPE_WEIGHTS["HIGH"] += 3 #이 조커는 한번쓰면 영구적으로 WEIGHT계속 적용 고로 REMOVE해야함.
            active_jokers.remove("JOKER_HIGH_1")
    if "JOKER_SUPER_1" in active_jokers:
        if bounce_type == "MIDDLE":
            BOUNCE_TYPE_WEIGHTS["SUPER"] += 5
            active_jokers.remove("JOKER_SUPER_1")  #이 조커는 한번쓰면 영구적으로 WEIGHT계속 적용 고로 REMOVE해야함.



    #결론: 조커 사용 했을때의 가중치를 업데이트하고 그걸 포함한 total_score를 최종으로 업데이트해서 global을 통해 함수에서 전역변수로 이동
    weight = BOUNCE_TYPE_WEIGHTS.get(bounce_type, 0)  # 유형이 없을 경우 기본값 0

    #가중치 업데이트 한걸로 조커 적용후 최종적으로 total_score 업데이트
    total_score += weight
    print(f"바운스 유형: {bounce_type}, 추가된 가중치: {weight}, 총 점수: {total_score}")


    # if "JOKER_SUPER_3" in active_jokers:
    #     if consecutive_super >= 3:
    #         total_score += 30
    #         joker_sound.play()
    #         print("[JOKER_SUPER_3] SUPER 3연속 달성! +30점 획득")
    #         active_jokers.remove("JOKER_SUPER_3")
    #         consecutive_super = 0
    
    # # ------------------------------------------------------------
    # # (C) sequence(순서) 관련 조커들
    # # ------------------------------------------------------------
    # # 1) "JOKER_LMHS_50": 직전 4바운스가 [LOW,MIDDLE,HIGH,SUPER]면 +50점
    # if "JOKER_LMHS_50" in active_jokers:
    #     if len(bounce_sequence) >= 4:
    #         last4 = bounce_sequence[-4:]
    #         if last4 == ["LOW", "MIDDLE", "HIGH", "SUPER"]:
    #             total_score += 50
    #             joker_sound.play() #조커 효과 발동 효과음
    #             print("[JOKER_LMHS_50] LOW→MIDDLE→HIGH→SUPER! +50점 획득")
    #             # 여러 번 발동 가능하게 유지한다면 제거 안 함
    #             # 한 번만 발동할거라면 아래 코드 추가:
    #             active_jokers.remove("JOKER_LMHS_50")
    




    # ------------------------------------------------------------
    # (D) count(누적) 관련 조커들
    # ------------------------------------------------------------



    # ------------------------------------------------------------
    # (E) 기타 특수 조커
    # ------------------------------------------------------------



def draw_single_player_mode(
    bounce_count,
    current_stage,
    single_mode_state="playing",  # "playing" or "choosing_joker"
    width=640,
    height=480
):
    """
    싱글 플레이어 모드 (1~10단계) UI를 그리는 함수.
    - current_stage: 현재 스테이지 번호
    - bounce_count: 현재 스테이지에서의 바운스 횟수
    - single_mode_state: "playing"이면 일반 스테이지 표시, "choosing_joker"면 조커 선택 박스 표시
    - width, height: 리턴할 이미지 크기
    """

    global chosen_jokers
    global drag_rect_x, drag_rect_y, drag_rect_w, drag_rect_h

    sp_img = np.zeros((height, width, 3), dtype=np.uint8)

    # ----------------------------
    # 1) 기본 싱글 플레이 모드 표시
    # ----------------------------
    cv2.putText(sp_img, "SINGLE PLAYER MODE", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

    total_stages = len(stage_thresholds)  # 예: 10
    box_w = 50
    box_h = 50
    margin = 5
    total_w = box_w * total_stages + margin * (total_stages - 1)
    start_x = (width - total_w) // 2
    y_top = 120

    for i in range(total_stages):
        stage_num = i + 1
        x_left = start_x + i * (box_w + margin)

        if stage_num < current_stage:
            # 이미 클리어한 단계
            color = (0, 255, 0)    # 초록
        elif stage_num == current_stage:
            # 현재 도전 중
            color = (0, 255, 255)  # 노랑
        else:
            # 아직 도달 못 한 단계
            color = (180, 180, 180)  # 회색

        cv2.rectangle(sp_img, (x_left, y_top), (x_left + box_w, y_top + box_h),
                      color, -1)
        cv2.rectangle(sp_img, (x_left, y_top), (x_left + box_w, y_top + box_h),
                      (0, 0, 0), 2)

        cv2.putText(sp_img, str(stage_num), (x_left+10, y_top+35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    # 모든 스테이지 클리어한 경우 메시지
    if current_stage > total_stages:
        cv2.putText(sp_img, "ALL CLEARED!!", (80, y_top + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 215, 255), 3)
    else:
        # 아직 클리어 안 한 경우, 이번 스테이지 목표와 현재 바운스 표시
        target = stage_thresholds[current_stage - 1]
        msg = f"Stage {current_stage}: {total_score} / {target}"
        cv2.putText(sp_img, msg, (20, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # ----------------------------
    # 2) 조커 선택 모드라면, 오버레이 UI 추가
    # ----------------------------
    if single_mode_state == "choosing_joker":
        #조커 선택하는동안 시작 못하게!
        drag_rect_x = 0
        drag_rect_y = 0
        drag_rect_w = 1
        drag_rect_y = 1


        if len(chosen_jokers) < 2:
            chosen_jokers = random.sample(all_jokers, 2)
        
        overlay = sp_img.copy()
        
        # 전체 반투명 배경
        cv2.rectangle(overlay, (0, 0), (width, height), (50, 50, 50), -1)
        alpha = 1.0
        sp_img = cv2.addWeighted(overlay, alpha, sp_img, 1 - alpha, 0)

        cv2.putText(sp_img, "Choose a Joker!", (150, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        # 첫 번째 박스(왼쪽)
        (x1, y1, x2, y2) = (50, 200, 270, 300)  
        cv2.rectangle(sp_img, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # chosen_jokers[0]의 title, desc 표시
        joker0 = chosen_jokers[0]
        cv2.putText(sp_img, joker0["title"], (x1+10, y1+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(sp_img, joker0["desc"], (x1+10, y1+80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 두 번째 박스(오른쪽)
        (x1b, y1b, x2b, y2b) = (350, 200, 570, 300)
        cv2.rectangle(sp_img, (x1b, y1b), (x2b, y2b), (255, 255, 255), 2)

        joker1 = chosen_jokers[1]
        cv2.putText(sp_img, joker1["title"], (x1b+10, y1b+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(sp_img, joker1["desc"], (x1b+10, y1b+80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return sp_img



while True:
    now = time.time()
    # if now - last_mouse_move_time > 3.0:
    #     if mouse_visible:
    #         user32.ShowCursor(False)
    #         mouse_visible = False

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) #좌우 반전으로 헷갈리지 않게 하기

    # [추가] 밝기 조절 적용: 전역 변수 brightness_alpha에 따라 밝기 조절
    # cv2.convertScaleAbs는 각 픽셀에 alpha * pixel + beta (여기서는 beta=0)를 적용합니다.
    frame = cv2.convertScaleAbs(frame, alpha=brightness_alpha, beta=0)
    if not ret:
        print("No more frames or camera error.")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    current_time = time.time()
    time_diff = current_time - prev_time
    if time_diff > 1e-9:
        fps = 1.0 / time_diff
    prev_time = current_time

    #FPS 1초마다 표시하기 위한 로직직
    elapsed = current_time - fps_start_time
    if elapsed >= 1.0:  # 1초 지났으면
        fps_display = fps_counter / elapsed  # '해당 1초 간' 평균 FPS
        fps_counter = 0
        fps_start_time = current_time

    results = model.predict(frame, imgsz=640, conf=0.3, max_det=1, show=False, device=0,verbose=False)
    boxes = results[0].boxes

    x_values.append(frame_count)
    frame_count += 1
    fps_counter += 1 #FPS 1초마다 평균 구하기위한 거. 1초마다 다시 0으로로

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

        conf = boxes[0].conf.cpu().numpy()[0] #신뢰도

        roi = frame[y1i:y2i, x1i:x2i]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([10, 100, 100], dtype=np.uint8)
        upper_orange = np.array([25, 255, 255], dtype=np.uint8)
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        orange_pixels = cv2.countNonZero(mask_orange)

        if ignore_zero_orange:
            if orange_pixels >= 3:
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
                    
                    #테스트 (멀리서 공을 치고있을때 빨간색 사각형안에 들어간 상태에서 치게되면 오류발생 그래서 사각형 tracking때는 없에려 함.)
                    drag_rect_x = 0
                    drag_rect_y = 0
                    drag_rect_w = 1
                    drag_rect_h = 1 #거의 작게 해서 치는 중에 일정 시간이상 못들어가게 하려고

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
                        ready_sound.play() #준비 상황
                        
                        enlarged_view = 'tl' #트래킹 상태일때는 확대된 내 모습을 보기위해 범위파악을 위해서 자동 확대
                        
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
                                
                                bounce_count += 1 #내려갔다 올라옴.
                                print("Bounce detected!")

                                current_bounce_time = time.time()
                                bounce_times.append(current_bounce_time)

                                # [변경 후]
                                if previous_bounce_time is not None:
                                    td = current_bounce_time - previous_bounce_time
                                    bounce_time_diff = td
                                    print(f"Time diff between last two bounces: {td:.2f} s")
                                else:
                                    bounce_time_diff = None

                                previous_bounce_time = current_bounce_time

                                # (x좌표, y좌표, 이번 바운스의 직전 바운스 대비 시간차)
                                bounce_points.append((x_values[-1], y_values[-1], bounce_time_diff))

                                # 1) 바운스 타입 구하기
                                if bounce_time_diff is not None:
                                    if bounce_time_diff <= BOUNCE_THRESHOLDS["LOW"]:
                                        bounce_type = "LOW"
                                    elif bounce_time_diff < BOUNCE_THRESHOLDS["MIDDLE"]:
                                        bounce_type = "MIDDLE"
                                    elif bounce_time_diff < BOUNCE_THRESHOLDS["HIGH"]:
                                        bounce_type = "HIGH"
                                    else:
                                        bounce_type = "SUPER"
                                else:
                                    bounce_type = "LOW" #처음에는 일단 1로 시작하기 위해서 LOW



                                # === [싱글 플레이어 모드이면 처리] ===
                                if play_mode == "single":
                                    #조커 적용 (한 번만)
                                    apply_jokers_on_bounce(bounce_type)
                                    handle_single_mode_stage()
                                


                                
                                if sound_enabled:
                                    if play_mode == "tournament":
                                        if bounce_count in hundred_unit_sounds: #100,200,300,400,500,600,700,800,900,1000
                                            hundred_unit_sounds[bounce_count].play()
                                        elif bounce_count % 100 == 0: #1100이후로 1200,1300,1400,1500...
                                            score_sound.play()
                                        elif bounce_count % 10 == 0: #10,20,30,40,50,60,70,80,90,100
                                            collect_points_sound.play()
                                        else: #1,2,3,4,5,6,7,8,9..
                                            bounce_count_sound.play()
                                    elif play_mode == "single":
                                        if prev_total_score is not None:
                                            if 100 <= total_score <= 1000:
                                                # 백의 자리 변화 감지
                                                if (total_score // 100) != (prev_total_score // 100):
                                                    hundred_unit_sounds[(total_score // 100) * 100].play()
                                                # 십의 자리 변화 감지
                                                elif ((total_score // 10) % 10) != ((prev_total_score // 10) % 10):
                                                    collect_points_sound.play()
                                                else:
                                                    bounce_count_sound.play()
                                            elif 10 <= total_score <= 99:
                                                # 십의 자리 변화 감지
                                                if (total_score // 10) != (prev_total_score // 10):
                                                    collect_points_sound.play()
                                                else:
                                                    bounce_count_sound.play()
                                            else:
                                                # 일의 자리 변화 또는 단일 자리 숫자
                                                bounce_count_sound.play()
                                        else:
                                            # 초기 점수일 때
                                            if total_score in hundred_unit_sounds:
                                                hundred_unit_sounds[total_score].play()
                                            elif total_score % 10 == 0:
                                                collect_points_sound.play()
                                            else:
                                                bounce_count_sound.play()




                                


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

            cv2.putText(
                frame,
                f"Conf: {conf:.2f}",
                (x2i + 10, y1i + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),  # 파란색으로 설정
                2,
                cv2.LINE_AA
            )

            # 바운스 간격에 따른 레이블 표시 추가
            # ----------------------------------------------------------------------------------------
            if bounce_time_diff is not None:
                # 바운스 간격에 따른 레이블 결정
                if bounce_time_diff <= BOUNCE_THRESHOLDS["LOW"]:
                    label = "LOW"
                    color = BOUNCE_TYPE_COLORS["LOW"]
                elif bounce_time_diff < BOUNCE_THRESHOLDS["MIDDLE"]:
                    label = "MIDDLE"
                    color = BOUNCE_TYPE_COLORS["MIDDLE"]
                elif bounce_time_diff < BOUNCE_THRESHOLDS["HIGH"]:
                    label = "HIGH"
                    color = BOUNCE_TYPE_COLORS["HIGH"]
                else:
                    label = "SUPER"
                    color = BOUNCE_TYPE_COLORS["SUPER"]


                    
                # 레이블 표시 (공의 왼쪽에)
                cv2.putText(
                    frame,
                    f"{label}",
                    (x1i - 150, y1i + 30),  # 공의 왼쪽에 위치하도록 x좌표 조정
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,  # 글자 크기를 약간 키움
                    color,
                    2,
                    cv2.LINE_AA
                )
                
                # 시간 간격도 함께 표시 (레이블 아래에)
                cv2.putText(
                    frame,
                    f"{bounce_time_diff:.2f}s",
                    (x1i - 100, y1i + 60),  # 레이블 아래에 위치
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
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
        if last_detection_time is not None and (time.time() - last_detection_time) > 1.0: #1.0초 이상 공이 안보일때
            current_state = "waiting"

            #다시 없앤 리사이즈 사각형 원상복구 (준비 할수 있게)
            drag_rect_x = 0
            drag_rect_y = 0
            drag_rect_w = 640
            drag_rect_h = 200
            print("State changed to WAITING (no detection for 1s in READY)")

    # tracking 중에 공 '안 보이면' -> waiting
    if current_state == "tracking":
        if last_detection_time is not None and (time.time() - last_detection_time) >= 1.5: #1.5초 이상 공이 안보일때
            
            if bounce_count > 0:
                bounce_history.append(bounce_count)

                if len(bounce_history) > 14:
                    bounce_history.pop(0)

            #다시 없앤 리사이즈 사각형 원상복구 (준비 할수 있게)
            drag_rect_x = 0
            drag_rect_y = 0
            drag_rect_w = 640
            drag_rect_h = 200

            bounce_count = 0
            consecutiveDownCount = 0
            consecutiveUpCount = 0
            state = None
            current_state = "waiting"

            # 토너먼트 결과가 업데이트되었으므로, Combined 창의 토너먼트 영역(enlarged_view)을 'br'(오른쪽 하단)로 설정
            if play_mode == "tournament":
                enlarged_view = 'br'


            


            if play_mode == "single":
                no_fail_streak = 0  # 연속 실패 스택 초기화
            
            

            # ★ 수정
            if just_cleared_stage:
                # 방금 스테이지 클리어했는데 공을 놓치는 것은 '실패'가 아니라 '클리어 이후 자연스러운 종료'
                # → stage=1로 돌리지 않음
                just_cleared_stage = False  # 플래그만 해제하고 끝
                print("No bounce (just cleared stage), so not resetting to stage 1.")
            else:
                alert_sound.play()
                # 실제 실패(중도 탈락)인 경우만 stage=1로 돌아감
                if play_mode == 'single':
                    # 스테이지 실패 => 다시 1단계로
                    reset_all_states() #초기화
                print("No bounce => waiting + alert_sound!")
                

            print("No detection => bounce_count reset to 0, go WAITING (fail => stage=1)")

    # 그래프 데이터 길이 제한
    if len(x_values) > MAX_POINTS:
        x_values.pop(0)
        y_values.pop(0)
        orange_pixel_values.pop(0)

    # 공이 화면 보이지만 바운스를 일정시간 감지 못하면 아웃
    if current_bounce_time is not None:
        if time.time() - current_bounce_time > CONTINUOUS_TIMEOUT: #가장 최근 바운스 이후로 1.5초 이상 지났을때
            
            if bounce_count > 0:
                bounce_history.append(bounce_count)

                if len(bounce_history) > 14:
                    bounce_history.pop(0)
            #다시 없앤 리사이즈 사각형 원상복구 (준비 할수 있게)
            drag_rect_x = 0
            drag_rect_y = 0
            drag_rect_w = 640
            drag_rect_h = 200

                # # 토너먼트 결과가 업데이트되었으므로, Combined 창의 토너먼트 영역(enlarged_view)을 'br'(오른쪽 하단)로 설정
            if play_mode == "tournament":
                enlarged_view = 'br'

            
            bounce_count = 0
            current_state = "waiting"

            if play_mode == "single":
                no_fail_streak = 0  # 연속 실패 스택 초기화
            

            # ★ 수정
            if just_cleared_stage:
                # 방금 스테이지 클리어했는데 공을 놓치는 것은 '실패'가 아니라 '클리어 이후 자연스러운 종료'
                # → stage=1로 돌리지 않음
                just_cleared_stage = False  # 플래그만 해제하고 끝
                print("No bounce (just cleared stage), so not resetting to stage 1.")
            else:
                alert_sound.play()
                # 실제 실패(중도 탈락)인 경우만 stage=1로 돌아감
                if play_mode == 'single':
                    # 스테이지 실패 => 다시 1단계로
                    reset_all_states() #초기화
                print("No bounce => waiting + alert_sound!")


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
        f"FPS: {fps_display:.2f}", #1초마다 할거면 fps_disply, 프레임마다할거면 fps
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

    if play_mode == "tournament":
        # 토너먼트 모드에서는 bottom-right에 토너먼트 이미지를 표시
        # bounce_history 길이가 변경되었을 때만 tournament_img 업데이트
        if len(bounce_history) != draw_tournament_img_unified.previous_bounce_history_len:
            tournament_img = draw_tournament_img_unified(bounce_history, width=640, height=480)
            draw_tournament_img_unified.previous_bounce_history_len = len(bounce_history)
        combined_img[480:960, 640:1280] = tournament_img
    elif play_mode == "single":
        # 싱글 모드
        single_player_img = draw_single_player_mode(
            bounce_count,
            current_stage,
            single_mode_state,
            width=640,
            height=480
        )
        combined_img[480:960, 640:1280] = single_player_img


    # 확대뷰가 없으면 4분할
    if enlarged_view is None:
        combined_img[0:480, 0:640] = frame_resized
        combined_img[0:480, 640:1280] = y_graph_img
        combined_img[480:960, 0:640] = orange_graph_img
        # combined_img[480:960, 640:1280] = single_player_img #이건 조건에 따라서 해당 ㅗㅁ드로  테스트
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
            if play_mode == "tournament":
                big_view = cv2.resize(tournament_img, (1280, 960), interpolation=cv2.INTER_AREA)
            elif play_mode == "single":  # single mode
                big_view = cv2.resize(single_player_img, (1280, 960), interpolation=cv2.INTER_AREA)
            combined_img = big_view

    cv2.imshow("Combined", combined_img)

    # ----------------------------------------------------------------------------
    ##수정: "Bounce Count Window"는 싱글도 토너먼트도 모드별로 값이 바뀔 때만 재렌더링
    if play_mode == 'single':
        # single 모드 => total_score/target 표시
        if current_stage <= len(stage_thresholds):
            target = stage_thresholds[current_stage - 1]
        else:
            target = 99999  # 혹은 다른 값
        # total_score 값이 **바뀌었을 때만** bounce_img를 새로 만든다
        

        if prev_total_score != total_score:

            if 'bounce_type' not in locals(): #현재 bounce_type이 존재하지 않을때
                score_color = (255, 255, 255)  # 흰색 (기본)
            else:
                if bounce_type == "LOW":
                    # score_color = (200, 200, 200)  # 회색
                    score_color = BOUNCE_TYPE_COLORS["LOW"]
                elif bounce_type == "MIDDLE":
                    # score_color = (255, 102, 0)  # 주황색
                    score_color = BOUNCE_TYPE_COLORS["MIDDLE"]
                elif bounce_type == "HIGH":
                    # score_color = (255, 255, 0)  # 노란색
                    score_color = BOUNCE_TYPE_COLORS["HIGH"]
                elif bounce_type == "SUPER":
                    # score_color = (255, 0, 255)  # 보라색
                    score_color = BOUNCE_TYPE_COLORS["SUPER"]
                else:
                    score_color = (255,255,255) #흰색



            # display_value = f"{total_score}/{target}"

            # 텍스트 세그먼트 정의: [(텍스트, 색상), ...]
            display_segments = [
                (f"{current_stage}세트",(255, 255, 255),130),
                (str(total_score), score_color, 400),
                (f"/{target}", (255, 255, 255),400)  # 흰색
            ]

            # 새로운 함수로 bounce_img 생성
            bounce_img = render_text_with_ttf_segments(
                text_segments=display_segments,
                base_font_path= HANDWRITING_FONT_PATH,
                bg_color=(0, 0, 255),  # 기본 파랑 배경
                width=1920,
                height=1080
            )
            prev_total_score = total_score

    elif play_mode == "tournament":
        # 기존 로직 (싱글 모드가 아닐 때)
        if bounce_count != prev_bounce_count:
            color = get_color(bounce_count)
            bounce_img = render_text_with_ttf(
                text=str(bounce_count),
                font=digital_font,
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
    elif key == ord('t') or key == ord('T'):
        reset_all_states()
        # 모드 토글
        play_mode = 'tournament'
        

    elif key == ord('s') or key == ord('S'):
        reset_all_states()
        # 싱글 모드로 전환
        play_mode = 'single'


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
    elif key in [ord('n'), ord('N')]:
        if play_mode == "tournament":
            # 1) PyQt 창 열어서 문자열 입력
            name_string = pyqt_text_input_mode("8강 참가자 이름(띄어쓰기 구분)")
            if name_string:
                # 2) 입력된 문자열을 공백 기준으로 나누기
                name_list = name_string.split()
                if len(name_list) == 8:
                    # 3) 8강 이름 리스트 갱신
                    tournament_players_8 = name_list
                    print("8강 이름 업데이트:", tournament_players_8)

                    tournament_img = draw_tournament_img_unified(bounce_history, width=640, height=480)
                    combined_img[480:960, 640:1280] = tournament_img #8강 이름 업데이트하고 그려줌.
                else:
                    print("8명 이름이 아닙니다. 다시 입력하세요.")
            else:
                print("입력이 취소되거나 빈 문자열.")
        else:
            print("토너먼트 모드에서만 이름을 입력할 수 있습니다.")
    elif key in [ord('p'), ord('P')]:
        # 1) PyQt 이용해서 '카메라 인덱스 API' 문자열 입력
        info_string = pyqt_text_input_mode("카메라 인덱스와 apiPreference를 띄어쓰기로 입력하세요.\n예시: '0 DSHOW'")

        if info_string:
            # 2) 입력 문자열을 공백으로 나눔
            parts = info_string.split()
            if len(parts) == 2:
                # 예: "0 CAP_MSMF" → parts[0] = "0", parts[1] = "CAP_MSMF"
                try:
                    new_cam_index = int(parts[0])   # 카메라 인덱스 정수 변환
                except ValueError:
                    print("오류: 카메라 인덱스는 정수로 입력해야 합니다.")
                    continue
                
                api_str = parts[1]
                # 3) 딕셔너리에서 API Preference 상수 찾기 DSHOW, MSMF등...
                if api_str in api_pref_map:
                    new_api_pref = api_pref_map[api_str]
                else:
                    print(f"오류: '{api_str}'은(는) 인식할 수 없는 API 명칭입니다.")
                    print(f"사용 가능한 API 목록: {list(api_pref_map.keys())}")
                    continue

                # 4) 기존 cap 해제 후, 새로운 cap으로 연결
                cap.release()
                cap = cv2.VideoCapture(new_cam_index, new_api_pref)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

                # 5) 성공 여부 체크
                if cap.isOpened():
                    print(f"[성공] 카메라 인덱스={new_cam_index}, API={api_str} 로 열었습니다.")
                else:
                    print(f"[실패] 카메라 인덱스={new_cam_index} 로는 열 수 없습니다.")
            
            else:
                print("입력 형식이 잘못되었습니다. 예) 0 DSHOW")
        else:
            print("입력이 취소되었거나 빈 문자열입니다.")
    # [추가] '-'와 '=' 키를 이용한 밝기 조절
    elif key == ord('-'):
        # 밝기를 낮추기 (최소 0.1까지)
        brightness_alpha = max(0.1, brightness_alpha - 0.1)
        print(f"Brightness decreased to {brightness_alpha:.1f}")
    elif key == ord('='):
        # 밝기를 높이기 (최대 3.0까지)
        brightness_alpha = min(3.0, brightness_alpha + 0.1)
        print(f"Brightness increased to {brightness_alpha:.1f}")



# ----------------------------------------------------------------------------------------
# 16) 종료 처리
# ----------------------------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()  
