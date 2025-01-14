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
        return (255, 0, 255) #1000이상이면 다른 함수 없이 보라색 리턴함.

    color_index = count // 100  # 100단위로 기본 색상 결정 (예: count=234 -> index=2)
    if color_index >= len(color_sequence):  # 색상 시퀀스 범위 초과 시 마지막 색상 사용
        color_index = len(color_sequence) - 1  # (예: color_index=11 -> 9로 조정)

    step_in_block = (count % 100) // 10  # 각 색상 내에서 10단위로 "밝기 단계" 결정 (예: count=234 -> step=3 밝기 단계 step 0...9)
    intensity = intensity_levels[step_in_block] if step_in_block < len(intensity_levels) else 1.0  # '특정' 밝기 레벨 선택 (예: step=3 -> 0.65)
    base_color = color_sequence[color_index]  # 기본 색상 선택 (예: (0,255,0))

    color_bgr = np.uint8([[base_color]])  # BGR 색상을 numpy 3차원 배열로 변환 (예: [[[0,255,0]]])
    color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]  # BGR을 HSV로 변환 (예: 괄호가 하나만 깔끔하게 있게 하기위해 [0][0] 추가 [60,255,255])

    color_hsv = color_hsv.astype(float)  # HSV 값을 실수형으로 변환하여 연산 가능하게 함
    color_hsv[2] = min(color_hsv[2] * intensity, 255)  # Value(밝기) 값 조정 (예: 255 * 0.65 = 165.75) 밝기 적용이 됨
    color_hsv = color_hsv.astype(np.uint8)  # 다시 정수형으로 변환 [H.0,S.0,V.0] -> [H,S,밝기 필터 적용된 V] (10마다 밝기 바뀌는거 고려해서)

    intense_color = cv2.cvtColor(np.uint8([[color_hsv]]), cv2.COLOR_HSV2BGR)[0][0]  # HSV를 BGR로 다시 변환 3차원 배열에서 그냥 배열로 하기위해 [0][0] 처리
    intense_color_rgb = (intense_color[2], intense_color[1], intense_color[0])  # BGR을 RGB로 변환 (예: [1,2,3]->[3,2,1]) PILLOW에서는 RGB 쓰기때문에!
    return intense_color_rgb  # 최종 RGB 색상 반환

# =============================================================================
# 드래그/리사이즈 가능한 빨간 사각형 관련 전역 변수
# =============================================================================
drag_rect_x, drag_rect_y = 100, 100  # 사각형 왼상단 초기 위치
drag_rect_w, drag_rect_h = 150, 150  # 사각형 폭, 높이
dragging = False                     # 현재 드래그(이동) 중인지 여부
resizing_corner = None               # 현재 리사이즈 중인 corner (None, 'tl', 'tr', 'bl', 'br')
drag_offset_x, drag_offset_y = 0, 0  # (이동용) 드래그 시작점 대비 사각형 내부 오프셋 이걸 통해 드래그해서 움직였을때 사각형의 새로운 왼쪽 상단 좌표 알수 있음!
corner_size = 10                     # 각 모서리 핸들의 반지름(또는 반폭)

# =============================================================================
# 사각형 내부에서 공이 감지된 시간을 실시간으로 표시하기 위한 변수
# =============================================================================
ball_in_rect_start = None   # 사각형 안에 공이 처음 들어온 시점(초)
in_rect_time = 0.0          # 사각형 안에 있는 동안의 시간(실시간 업데이트)

# ----------------------------------------------------------------------------------------
# (A) 우클릭 확대/복귀 기능 관련 전역 변수
# ----------------------------------------------------------------------------------------
enlarged_view = None  # 'tl', 'tr', 'bl', 'br' or None (기본값: None=4분할)

# ----------------------------------------------------------------------------------------
# 9) mouse_callback 함수
# ----------------------------------------------------------------------------------------
last_mouse_move_time = time.time() #현재 시간!
mouse_visible = True #마우스 커서 처음에 보이게 하기위에 True 설정

def mouse_callback(event, x, y, flags, param):
    # 1. 소리/오렌지색 감지 관련 전역변수
    # 예: sound_enabled=True이면 소리 켜짐, ignore_zero_orange=True이면 오렌지색 픽셀에 따른 검출 필터 무시
    global sound_enabled, ignore_zero_orange

    # 2. 마우스 커서 표시/숨김 관련 전역변수
    # 예: last_mouse_move_time=현재시간, mouse_visible=True이면 커서 보임
    global last_mouse_move_time, mouse_visible

    # 3. 사각형 드래그(이동) 관련 전역변수
    # 예: dragging=True이면 드래그 중, offset=(10,20)이면 마우스 클릭점과 사각형 좌상단의 거리
    global dragging, drag_offset_x, drag_offset_y

    # 4. 사각형의 위치/크기 관련 전역변수
    # 예: drag_rect_x=100, y=100이면 좌상단 좌표가 (100,100)
    # 예: drag_rect_w=150, h=150이면 폭과 높이가 각각 150
    global drag_rect_x, drag_rect_y, drag_rect_w, drag_rect_h

    # 5. 사각형 크기조절 관련 전역변수
    # 예: resizing_corner='tl'이면 좌상단 모서리를 드래그하여 크기조절 중
    global resizing_corner

    # 6. 화면 확대/축소 관련 전역변수
    # 예: enlarged_view='tr'이면 우상단 영역이 확대되어 전체화면에 표시됨
    global enlarged_view  # (A) 우클릭 확대/복귀

    if event == cv2.EVENT_MOUSEMOVE:                                     # 마우스가 움직일 때마다
        last_mouse_move_time = time.time()                               # 마지막 마우스 움직임 시간 갱신 
        if not mouse_visible:                                            # 마우스가 숨겨져 있다면
            user32.ShowCursor(True)                                      # 마우스 커서 보이게 하고
            mouse_visible = True                                         # 마우스 보임 상태로 변경

        # 리사이즈 중이면 각 코너별로 크기 갱신
        if resizing_corner is not None:                                 # 현재 모서리를 드래그해서 크기 조절 중이라면
            if resizing_corner == 'tl':                                 # 왼쪽 상단(Top-Left) 모서리를 드래그 중일 때
                new_w = drag_rect_w + (drag_rect_x - x)                 # 예: 기존 너비 150 + (기존 x 100 - 현재 x 80) = 170
                new_h = drag_rect_h + (drag_rect_y - y)                 # 예: 기존 높이 150 + (기존 y 100 - 현재 y 80) = 170
                new_x = x                                               # 새로운 x 좌표는 마우스 현재 위치
                new_y = y                                               # 새로운 y 좌표는 마우스 현재 위치
                if new_w < 10:                                          # 너비가 10 미만이면
                    new_w = 10                                          # 최소 너비 10으로 제한
                    new_x = drag_rect_x + drag_rect_w - 10              # 좌상단(매우중요)x 좌표도 그에 맞게 조정
                if new_h < 10:                                          # 높이가 10 미만이면
                    new_h = 10                                          # 좌상단(매우중요) 최소 높이 10으로 제한
                    new_y = drag_rect_y + drag_rect_h - 10              # y 좌표도 그에 맞게 조정
                #마우스의 좌표 new_x와 우상단 x좌표와 비교함. 리사이즈 하려해도 뒤집어지지 않고 제한함.
                new_x = max(0, min(new_x, drag_rect_x + drag_rect_w))   # x 좌표가 원래 사각형 범위를 벗어나지 않도록
                #마우스의 좌표 new_y와 좌하단 y좌표와 비교함. 리사이즈 하려해도 뒤집어지지 않고 제한함. 
                new_y = max(0, min(new_y, drag_rect_y + drag_rect_h))   # y 좌표가 원래 사각형 범위를 벗어나지 않도록

                if new_x < 0: new_x = 0                                 # 화면 왼쪽 경계 체크 드래그 해서 밀어도 음수쪽으로 못감.
                if new_y < 0: new_y = 0                                 # 화면 위쪽 경계 체크 드래그 해서 밀어도 음수쪽으로 못감.   
                if new_x > 640: new_x = 640                            # 화면 오른쪽 경계 체크 드래그 해서 밀어도 640초과로 못감.
                if new_y > 480: new_y = 480                            # 화면 아래쪽 경계 체크 드래그 해서 밀어도 480초과로 못감.

                drag_rect_w = new_w                                     # 계산된 새로운 너비 적용
                drag_rect_h = new_h                                     # 계산된 새로운 높이 적용
                drag_rect_x = new_x                                     # 계산된 새로운 좌상단 x 좌표 적용
                drag_rect_y = new_y                                     # 계산된 새로운 좌상단 y 좌표 적용

            elif resizing_corner == 'tr':                               # 오른쪽 상단(Top-Right) 모서리를 드래그 중일 때
                new_w = x - drag_rect_x                                 # 예: 현재 x 200 - 기존 x 100 = 100 (새 너비)
                new_h = drag_rect_h + (drag_rect_y - y)                 # 예: 기존 높이 150 + (기존 y 100 - 현재 y 80) = 170
                new_y = y                 
                #new_x는 tr잡고 리사이징 할때 좌상단은 아무리해도 그대로여서 그대로 두면됨.                      
                if new_w < 10:                                          # 너비가 10 미만이면
                    new_w = 10                                          # 최소 너비 10으로 제한
                if new_h < 10:                                          # 높이가 10 미만이면
                    new_h = 10                                          # 최소 높이 10으로 제한
                    new_y = drag_rect_y + drag_rect_h - 10              # y 좌표도 그에 맞게 조정
                if new_w > 640 - drag_rect_x:                          # 화면 오른쪽 경계를 넘어가면
                    new_w = 640 - drag_rect_x                          # 최대 너비로 제한
                if new_y < 0:                                          # 화면 위쪽 경계를 넘어가면
                    new_y = 0                                          # y 좌표를 0으로 제한

                drag_rect_w = new_w                                     # 계산된 새로운 너비 적용
                drag_rect_h = new_h                                     # 계산된 새로운 높이 적용
                drag_rect_y = new_y                                     # 계산된 새로운 y 좌표 적용

            elif resizing_corner == 'bl':                               # 왼쪽 하단(Bottom-Left) 모서리를 드래그 중일 때
                new_w = drag_rect_w + (drag_rect_x - x)                 # 예: 기존 너비 150 + (기존 x 100 - 현재 x 80) = 170
                new_h = y - drag_rect_y                                 # 예: 현재 y 200 - 기존 y 100 = 100 (새 높이)
                new_x = x                                               # 새로운 x 좌표는 마우스 현재 위치
                if new_w < 10:                                          # 너비가 10 미만이면
                    new_w = 10                                          # 최소 너비 10으로 제한
                    new_x = drag_rect_x + drag_rect_w - 10              # x 좌표도 그에 맞게 조정
                if new_h < 10:                                          # 높이가 10 미만이면
                    new_h = 10                                          # 최소 높이 10으로 제한
                if new_x < 0:                                          # 화면 왼쪽 경계를 넘어가면
                    new_x = 0                                          # x 좌표를 0으로 제한
                if new_h > 480 - drag_rect_y:                          # 화면 아래쪽 경계를 넘어가면
                    new_h = 480 - drag_rect_y                          # 최대 높이로 제한

                drag_rect_w = new_w                                     # 계산된 새로운 너비 적용
                drag_rect_h = new_h                                     # 계산된 새로운 높이 적용
                drag_rect_x = new_x                                     # 계산된 새로운 x 좌표 적용

            elif resizing_corner == 'br':                               # 오른쪽 하단(Bottom-Right) 모서리를 드래그 중일 때
                new_w = x - drag_rect_x                                 # 예: 현재 x 200 - 기존 x 100 = 100 (새 너비)
                new_h = y - drag_rect_y                                 # 예: 현재 y 200 - 기존 y 100 = 100 (새 높이)
                if new_w < 10:                                          # 너비가 10 미만이면
                    new_w = 10                                          # 최소 너비 10으로 제한
                if new_h < 10:                                          # 높이가 10 미만이면
                    new_h = 10                                          # 최소 높이 10으로 제한
                if new_w > 640 - drag_rect_x:                          # 화면 오른쪽 경계를 넘어가면
                    new_w = 640 - drag_rect_x                          # 최대 너비로 제한
                if new_h > 480 - drag_rect_y:                          # 화면 아래쪽 경계를 넘어가면
                    new_h = 480 - drag_rect_y                          # 최대 높이로 제한

                drag_rect_w = new_w                                     # 계산된 새로운 너비 적용
                drag_rect_h = new_h                                     # 계산된 새로운 높이 적용

        elif dragging:                                                  # 사각형을 드래그해서 이동 중이라면
            new_x = x - drag_offset_x                                   # 예: 현재 x 200 - 오프셋 50 = 150 (새 x 좌표)
            new_y = y - drag_offset_y                                   # 예: 현재 y 200 - 오프셋 50 = 150 (새 y 좌표)
            new_x = max(0, min(new_x, 640 - drag_rect_w))              # x 좌표가 화면을 벗어나지 않도록 제한
            new_y = max(0, min(new_y, 480 - drag_rect_h))              # y 좌표가 화면을 벗어나지 않도록 제한
            drag_rect_x, drag_rect_y = new_x, new_y                    # 계산된 새로운 위치 적용

    elif event == cv2.EVENT_LBUTTONDOWN:                                # 마우스 왼쪽 버튼을 눌렀을 때
        # 사운드 ON/OFF 버튼
        if (button_rect[0] <= x - 640 <= button_rect[0] + button_rect[2] and    # 예: 500 <= x-640 <= 620 (버튼 x범위 체크)
            button_rect[1] <= y <= button_rect[1] + button_rect[3]):            # 예: 20 <= y <= 60 (버튼 y범위 체크)
            sound_enabled = not sound_enabled                           # 소리 설정을 반전 (예: True -> False)
            print(f"Sound Enabled: {sound_enabled}")                   # 현재 소리 설정 상태 출력 (예: "Sound Enabled: False")
        
        # Ignore Zero Orange 버튼
        elif (button_rect_ignore[0] <= x - 640 <= button_rect_ignore[0] + button_rect_ignore[2] and    # 예: 500 <= x-640 <= 620 (무시 버튼 x범위)
              button_rect_ignore[1] <= y <= button_rect_ignore[1] + button_rect_ignore[3]):            # 예: 70 <= y <= 110 (무시 버튼 y범위)
            ignore_zero_orange = not ignore_zero_orange                # 오렌지픽셀 무시 설정 반전 (예: False -> True)
            print(f"Ignore Zero Orange Pixels: {ignore_zero_orange}")  # 현재 무시 설정 상태 출력 (예: "Ignore Zero Orange Pixels: True")
        
        else:
            corners = {                                               # 사각형의 4개 모서리 좌표 저장
                'tl': (drag_rect_x, drag_rect_y),                    # 예: 좌상단 (100, 100)
                'tr': (drag_rect_x + drag_rect_w, drag_rect_y),      # 예: 우상단 (200, 100)
                'bl': (drag_rect_x, drag_rect_y + drag_rect_h),      # 예: 좌하단 (100, 200)
                'br': (drag_rect_x + drag_rect_w, drag_rect_y + drag_rect_h)  # 예: 우하단 (200, 200)
            }
            corner_clicked = None                                     # 클릭된 모서리 초기화
            for ckey, cpos in corners.items():                       # 각 모서리 확인
                cx, cy = cpos                                        # 예: cx=100, cy=100 (좌상단)
                if (cx - corner_size <= x <= cx + corner_size and    # 예: 95 <= x <= 105 (모서리 x범위)
                    cy - corner_size <= y <= cy + corner_size):      # 예: 95 <= y <= 105 (모서리 y범위)
                    corner_clicked = ckey                            # 클릭된 모서리 저장 (예: 'tl')
                    break

            if corner_clicked:                                       # 모서리가 클릭되었다면
                resizing_corner = corner_clicked                     # 크기 조절할 모서리 설정 (예: resizing_corner = 'tl')
            else:
                # 사각형 내부라면 드래그(이동) 시작
                if (drag_rect_x <= x < drag_rect_x + drag_rect_w and     # 예: 100 <= x < 200 (사각형 x범위)
                    drag_rect_y <= y < drag_rect_y + drag_rect_h):       # 예: 100 <= y < 200 (사각형 y범위)
                    dragging = True                                       # 드래그 시작
                    drag_offset_x = x - drag_rect_x                      # 드래그 시작점과 사각형 좌상단의 x차이 (예: 150-100=50)
                    drag_offset_y = y - drag_rect_y                      # 드래그 시작점과 사각형 좌상단의 y차이 (예: 150-100=50)

    elif event == cv2.EVENT_LBUTTONUP:                              # 마우스 왼쪽 버튼을 뗐을 때
        dragging = False                                            # 드래그 종료
        resizing_corner = None                                      # 크기 조절 모서리 초기화

    # (A) 우클릭 시 해당 쿼드런트만 확대 or 복귀
    elif event == cv2.EVENT_RBUTTONDOWN:                           # 마우스 오른쪽 버튼을 눌렀을 때
        if enlarged_view is None:                                  # 현재 확대된 화면이 없다면
            # 4개 쿼드런트 범위:
            # top-left:    y in [0,480), x in [0,640)             # 예: (300,200)은 top-left
            # top-right:   y in [0,480), x in [640,1280)          # 예: (800,200)은 top-right
            # bottom-left: y in [480,960), x in [0,640)           # 예: (300,600)은 bottom-left
            # bottom-right:y in [480,960), x in [640,1280)        # 예: (800,600)은 bottom-right

            if 0 <= y < 480 and 0 <= x < 640:                     # 좌상단 영역 클릭 시
                enlarged_view = 'tl'                               # 좌상단 확대 모드로 설정
            elif 0 <= y < 480 and 640 <= x < 1280:                # 우상단 영역 클릭 시
                enlarged_view = 'tr'                               # 우상단 확대 모드로 설정
            elif 480 <= y < 960 and 0 <= x < 640:                 # 좌하단 영역 클릭 시
                enlarged_view = 'bl'                               # 좌하단 확대 모드로 설정
            elif 480 <= y < 960 and 640 <= x < 1280:              # 우하단 영역 클릭 시
                enlarged_view = 'br'                               # 우하단 확대 모드로 설정

            if enlarged_view is not None:                         # 확대 모드가 설정되었다면
                print(f"Enlarged => {enlarged_view}")             # 예: "Enlarged => tl" 출력

        else:                                                     # 이미 확대된 상태라면
            print(f"Return to 4-split from: {enlarged_view}")     # 예: "Return to 4-split from: tl" 출력
            enlarged_view = None                                  # 확대 모드 해제


# ----------------------------------------------------------------------------------------
# 10) render_text_with_ttf()
# ----------------------------------------------------------------------------------------
def render_text_with_ttf(                                        # TTF 폰트로 텍스트를 렌더링하는 함수
    text,                                                        # 예: "123"
    font=font,                                                   # 예: ImageFont.truetype("Digital Display.ttf", 400)
    text_color=(255, 255, 255),                                 # 예: 흰색 (255,255,255)
    bg_color=(0, 0, 0),                                         # 예: 검은색 (0,0,0)
    width=960,                                                  # 예: 이미지 너비 960픽셀
    height=540                                                  # 예: 이미지 높이 540픽셀
):
    img_pil = Image.new("RGB", (width, height), bg_color)       # 예: 960x540 크기의 검은색 배경 이미지 생성
    draw = ImageDraw.Draw(img_pil)                              # 이미지에 그리기 위한 Draw 객체 생성

    text_bbox = draw.textbbox((0, 0), text, font=font)          # 텍스트의 경계 상자 계산 (예: (10,10,200,100))
    text_w = text_bbox[2] - text_bbox[0]                        # 텍스트 너비 계산 (예: 190)
    text_h = text_bbox[3] - text_bbox[1]                        # 텍스트 높이 계산 (예: 90)

    text_x = (width - text_w) // 2                              # 텍스트 x 중앙 위치 계산 (예: 385)
    text_y = (height - text_h) // 2                             # 텍스트 y 중앙 위치 계산 (예: 225)
    draw.text((text_x, text_y), text, font=font, fill=text_color)  # 텍스트 그리기

    img_np = np.array(img_pil)                                  # PIL 이미지를 numpy 배열로 변환
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)          # RGB를 BGR로 변환
    return img_bgr                                              # BGR 이미지 반환

# ----------------------------------------------------------------------------------------
# 11) y좌표 그래프 그리기 함수
# ----------------------------------------------------------------------------------------
def draw_y_graph(x_data, y_data, width=640, height=480, max_y=480, bounce_pts=None):
    if bounce_pts is None:
        bounce_pts = []  # bounce_pts가 None이면 빈 리스트로 초기화 (예: bounce_pts = [])

    graph_img = np.zeros((height, width, 3), dtype=np.uint8)  # 검은색 배경 이미지 생성 (예: 640x480 크기의 검은색 이미지)
    if len(x_data) < 2:
        return graph_img  # 데이터가 2개 미만이면 빈 이미지 반환 (예: x_data=[1]일 때)

    max_x = x_data[-1] if x_data[-1] != 0 else 1  # x축 최대값 설정 (예: x_data=[0,1,2,3]이면 max_x=3)

    for i in range(len(x_data) - 1):  # 연속된 두 점을 선으로 연결
        if y_data[i] is None or y_data[i+1] is None:
            continue  # None 값이 있으면 건너뜀 (예: y_data=[100,None,300]일 때 None 건너뜀)
        x1_ori, y1_ori = x_data[i], y_data[i]  # 첫 번째 점의 원본 좌표 (예: x1_ori=1, y1_ori=100)
        x2_ori, y2_ori = x_data[i+1], y_data[i+1]  # 두 번째 점의 원본 좌표 (예: x2_ori=2, y2_ori=200)

        x1 = int((x1_ori - x_data[0]) / (max_x - x_data[0] + 1e-6) * (width - 1))  # x1 화면 좌표 변환 (예: 1 -> 213)
        x2 = int((x2_ori - x_data[0]) / (max_x - x_data[0] + 1e-6) * (width - 1))  # x2 화면 좌표 변환 (예: 2 -> 426)

        y1 = int(y1_ori / max_y * (height - 1))  # y1 화면 좌표 변환 (예: 100 -> 100)
        y2 = int(y2_ori / max_y * (height - 1))  # y2 화면 좌표 변환 (예: 200 -> 200)

        cv2.line(graph_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 초록색 선 그리기 (예: (213,100)에서 (426,200)까지)

    for i in range(len(x_data)):  # 각 데이터 포인트에 파란색 원 그리기
        if y_data[i] is None:
            continue  # None 값 건너뜀 (예: y_data=[100,None,300]일 때 None 건너뜀)
        x_ori, y_ori = x_data[i], y_data[i]  # 원본 좌표 (예: x_ori=1, y_ori=100)
        x_pt = int((x_ori - x_data[0]) / (max_x - x_data[0] + 1e-6) * (width - 1))  # x 화면 좌표 변환 (예: 1 -> 213)
        y_pt = int(y_ori / max_y * (height - 1))  # y 화면 좌표 변환 (예: 100 -> 100)
        cv2.circle(graph_img, (x_pt, y_pt), 4, (255, 0, 0), -1)  # 파란색 원 그리기 (예: (213,100)에 반지름 4 원)

    for (bx_ori, by_ori) in bounce_pts:  # 바운스 포인트에 빨간색 원 그리기
        if bx_ori < x_data[0]:
            continue  # x축 범위 밖의 바운스 포인트 건너뜀 (예: bx_ori=0, x_data[0]=1일 때)
        bx = int((bx_ori - x_data[0]) / (max_x - x_data[0] + 1e-6) * (width - 1))  # x 화면 좌표 변환 (예: 2 -> 426)
        by = int(by_ori / max_y * (height - 1))  # y 화면 좌표 변환 (예: 300 -> 300)
        cv2.circle(graph_img, (bx, by), 5, (0, 0, 255), -1)  # 빨간색 원 그리기 (예: (426,300)에 반지름 5 원)

    # 사운드 ON/OFF 버튼 그리기
    cv2.rectangle(
        graph_img,
        (button_rect[0], button_rect[1]),  # 버튼 좌상단 좌표 (예: (10,10))
        (button_rect[0] + button_rect[2], button_rect[1] + button_rect[3]),  # 버튼 우하단 좌표 (예: (110,40))
        (120, 120, 120),  # 회색
        -1
    )
    text_sound = "Sound: ON" if sound_enabled else "Sound: OFF"  # 사운드 상태 텍스트 (예: "Sound: ON")
    cv2.putText(
        graph_img,
        text_sound,
        (button_rect[0] + 10, button_rect[1] + 25),  # 텍스트 위치 (예: (20,35))
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,  # 폰트 크기
        (255, 255, 255),  # 흰색
        2  # 텍스트 두께
    )

    # Ignore0 버튼 그리기
    cv2.rectangle(
        graph_img,
        (button_rect_ignore[0], button_rect_ignore[1]),  # 버튼 좌상단 좌표 (예: (120,10))
        (button_rect_ignore[0] + button_rect_ignore[2], button_rect_ignore[1] + button_rect_ignore[3]),  # 버튼 우하단 좌표 (예: (220,40))
        (120, 120, 120),  # 회색
        -1
    )
    text_ignore = "Ignore0: ON" if ignore_zero_orange else "Ignore0: OFF"  # Ignore0 상태 텍스트 (예: "Ignore0: ON")
    cv2.putText(
        graph_img,
        text_ignore,
        (button_rect_ignore[0] + 5, button_rect_ignore[1] + 25),  # 텍스트 위치 (예: (125,35))
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,  # 폰트 크기
        (255, 255, 255),  # 흰색
        2  # 텍스트 두께
    )

    return graph_img  # 완성된 그래프 이미지 반환

# ----------------------------------------------------------------------------------------
# 12) 오렌지 픽셀 그래프 그리기 함수
# ----------------------------------------------------------------------------------------
def draw_orange_graph(x_data, orange_data, width=640, height=480, max_y=None):  # 예: x_data=[1,2,3], orange_data=[100,200,300]
    if max_y is None:  # max_y가 지정되지 않은 경우
        valid_orange_data = [v for v in orange_data if v is not None]  # 예: [100,200,300] 
        max_y = max(valid_orange_data) if valid_orange_data else 1  # 예: max_y = 300

    graph_img = np.zeros((height, width, 3), dtype=np.uint8)  # 예: 480x640 크기의 검은색 이미지 생성
    if len(x_data) < 2:  # 데이터가 2개 미만이면 빈 이미지 반환
        return graph_img

    max_x = x_data[-1] if x_data[-1] != 0 else 1  # 예: x축 최대값 = 3

    for i in range(len(x_data) - 1):  # 각 데이터 포인트를 선으로 연결
        if orange_data[i] is None or orange_data[i+1] is None:  # None 값은 건너뜀
            continue
        x1_ori, y1_ori = x_data[i], orange_data[i]  # 예: (1,100)
        x2_ori, y2_ori = x_data[i+1], orange_data[i+1]  # 예: (2,200)

        x1 = int((x1_ori - x_data[0]) / (max_x - x_data[0] + 1e-6) * (width - 1))  # 예: 1 -> 213
        x2 = int((x2_ori - x_data[0]) / (max_x - x_data[0] + 1e-6) * (width - 1))  # 예: 2 -> 426

        y1 = int(y1_ori / max_y * (height - 1)) if max_y > 0 else 0  # 예: 100 -> 160
        y2 = int(y2_ori / max_y * (height - 1)) if max_y > 0 else 0  # 예: 200 -> 320

        cv2.line(graph_img, (x1, height - y1), (x2, height - y2), (0, 165, 255), 2)  # 예: (213,320)-(426,160)에 주황색 선

    for i in range(len(x_data)):  # 각 데이터 포인트에 원과 값 표시
        if orange_data[i] is None:  # None 값은 건너뜀
            continue
        x_ori, y_ori = x_data[i], orange_data[i]  # 예: (1,100)
        x_pt = int((x_ori - x_data[0]) / (max_x - x_data[0] + 1e-6) * (width - 1))  # 예: 1 -> 213
        y_pt = int(y_ori / max_y * (height - 1)) if max_y > 0 else 0  # 예: 100 -> 160
        cv2.circle(graph_img, (x_pt, height - y_pt), 4, (0, 165, 255), -1)  # 예: (213,320)에 주황색 원
        cv2.putText(  # 값 텍스트 표시
            graph_img,
            f"{y_ori}",  # 예: "100"
            (x_pt + 5, height - y_pt - 5),  # 예: (218,315)
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),  # 주황색
            1,
            cv2.LINE_AA
        )

    cv2.line(graph_img, (0, height - 1), (width - 1, height - 1), (255, 255, 255), 1)  # 예: x축 흰색 선
    cv2.line(graph_img, (0, 0), (0, height - 1), (255, 255, 255), 1)  # 예: y축 흰색 선

    cv2.putText(  # 그래프 제목 표시
        graph_img,
        "Orange Pixel Count",  # 제목 텍스트
        (10, 30),  # 예: 좌상단 (10,30)
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),  # 흰색
        2,
        cv2.LINE_AA
    )

    return graph_img  # 완성된 그래프 이미지 반환

# ----------------------------------------------------------------------------------------
# 13) Combined, Bounce Count 창을 생성 & Combined 창을 전체화면으로 시작
# ----------------------------------------------------------------------------------------
cv2.namedWindow("Combined", cv2.WINDOW_NORMAL)  # Combined 창 생성 (크기 조절 가능)
cv2.setWindowProperty("Combined", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Combined 창을 전체화면으로 설정
is_fullscreen_combined = True  # Combined 창이 전체화면 모드인지 여부 (예: True)

cv2.namedWindow("Bounce Count Window", cv2.WINDOW_NORMAL)  # Bounce Count 창 생성 (크기 조절 가능) 
cv2.setWindowProperty("Bounce Count Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Bounce Count 창을 전체화면으로 설정
is_fullscreen_bounce = True  # Bounce Count 창이 전체화면 모드인지 여부 (예: True)

cv2.setMouseCallback("Combined", mouse_callback)  # Combined 창에 마우스 이벤트 콜백 함수 설정

prev_bounce_count = None  # 이전 바운스 카운트 값 (예: 3)
bounce_img = None  # 바운스 카운트 표시용 이미지
is_fullscreen = False  # 전체 화면 모드 여부 (예: False)

# ----------------------------------------------------------------------------------------
# 14) 추가된 전역 변수: 상태 관리
# ----------------------------------------------------------------------------------------
current_state = "waiting"  # 현재 상태 (예: "waiting", "tracking", "finished")
state_display_text = "Waiting"  # 화면에 표시할 상태 텍스트 (예: "Waiting")
state_font = cv2.FONT_HERSHEY_SIMPLEX  # 상태 텍스트 폰트
state_font_scale = 1.0  # 상태 텍스트 크기 (예: 1.0)
state_font_color = (255, 255, 255)  # 상태 텍스트 색상 (예: 흰색)
state_font_thickness = 2  # 상태 텍스트 두께 (예: 2)
state_change_time = None  # 상태가 마지막으로 변경된 시간 (예: 1234567890.123)

stationary_start_time = None  # 공이 정지 상태로 진입한 시작 시간 (예: 1234567890.123)
stationary_threshold = 2.0  # 공이 정지했다고 판단할 시간 임계값 (초) (예: 2.0초)
movement_threshold = 5  # 공의 움직임을 감지할 픽셀 거리 임계값 (예: 5픽셀)
last_position = None  # 마지막으로 감지된 공의 위치 (예: (100, 200))

previous_bounce_time = None  # 이전 바운스가 발생한 시간 (예: 1234567890.123)

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
prev_time = time.time()  # 이전 프레임의 시간을 저장 (FPS 계산을 위해 필요)
fps = 0.0  # FPS 값을 저장할 변수 초기화 (화면에 FPS를 표시하기 위해 필요)

while True:  # 무한 루프로 비디오/카메라 프레임을 계속 처리
    now = time.time()  # 현재 시간을 가져옴 (마우스 커서 숨김 기능을 위해 필요)
    if now - last_mouse_move_time > 3.0:  # 마우스가 3초 이상 움직이지 않았는지 확인
        if mouse_visible:  # 마우스가 보이는 상태라면
            user32.ShowCursor(False)  # 마우스 커서를 숨김
            mouse_visible = False  # 마우스 상태를 숨김으로 변경

    ret, frame = cap.read()  # 카메라/비디오에서 새 프레임을 읽어옴
    if not ret:  # 프레임을 읽지 못했다면
        print("No more frames or camera error.")  # 에러 메시지 출력
        break  # 루프 종료

    current_time = time.time()  # 현재 시간을 가져옴 (FPS 계산을 위해 필요)
    time_diff = current_time - prev_time  # 이전 프레임과의 시간 차이 계산
    if time_diff > 1e-9:  # 0으로 나누기를 방지하기 위한 조건
        fps = 1.0 / time_diff  # FPS 계산 (초당 프레임 수)
    prev_time = current_time  # 다음 계산을 위해 현재 시간을 저장

    results = model.predict(frame, imgsz=640, conf=0.5, max_det=1, show=False, device=0)  # YOLO 모델로 공을 검출
    boxes = results[0].boxes  # 검출된 객체의 바운딩 박스 정보를 가져옴

    x_values.append(frame_count)  # 프레임 번호를 x축 값으로 저장 (그래프 표시용)
    frame_count += 1  # 프레임 카운터 증가

    detected = False  # 공 검출 여부를 저장할 플래그 초기화
    orange_pixels = 0  # 주황색 픽셀 수를 저장할 변수 초기화

    # ------------------------------------------------------------------------------------
    # 공 검출 여부 확인
    # ------------------------------------------------------------------------------------
    if len(boxes) > 0:  # 검출된 객체가 있다면
        x1, y1, x2, y2 = boxes[0].xyxy[0].cpu().numpy()  # 첫 번째 객체의 바운딩 박스 좌표를 가져옴
        y_center = (y1 + y2) / 2.0  # 바운딩 박스의 y축 중심점 계산
        x_center = (x1 + x2) / 2.0  # 바운딩 박스의 x축 중심점 계산

        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])  # 좌표값을 정수로 변환 (픽셀 인덱싱을 위해)
        x1i = max(0, x1i)  # x 좌표가 음수가 되지 않도록 보정
        y1i = max(0, y1i)  # y 좌표가 음수가 되지 않도록 보정
        x2i = min(frame.shape[1], x2i)  # x 좌표가 프레임 너비를 넘지 않도록 보정
        y2i = min(frame.shape[0], y2i)  # y 좌표가 프레임 높이를 넘지 않도록 보정

        roi = frame[y1i:y2i, x1i:x2i]  # 바운딩 박스 영역을 추출
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # RGB를 HSV 색공간으로 변환 (색상 검출을 위해)
        lower_orange = np.array([10, 100, 100], dtype=np.uint8)  # 주황색의 하한값 설정
        upper_orange = np.array([25, 255, 255], dtype=np.uint8)  # 주황색의 상한값 설정
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)  # 주황색 영역을 마스크로 추출
        orange_pixels = cv2.countNonZero(mask_orange)  # 주황색 픽셀의 개수를 계산

        if ignore_zero_orange:                            # 주황색 픽셀 무시 옵션이 켜져있는 경우
            if orange_pixels >= 5:                        # 주황색 픽셀이 5개 이상이면 (예: 작은 공이라도 최소 5픽셀은 있어야 함)
                detected = True                           # 공이 검출되었다고 판단
        else:                                            # 주황색 픽셀 무시 옵션이 꺼져있는 경우
            detected = True                              # YOLO가 검출한 것을 그대로 신뢰

        if detected:                                     # 공이 검출된 경우
            last_detection_time = time.time()            # 마지막 검출 시간을 현재 시간으로 업데이트 (예: 공이 사라졌다가 다시 나타나는 것을 추적하기 위해)

            y_values.append(y_center)                    # 공의 y좌표를 기록 (예: 그래프 그리기 위해)
            orange_pixel_values.append(orange_pixels)    # 주황색 픽셀 수를 기록 (예: 공의 크기 변화를 추적하기 위해)

            # --------------------------------------------------------------------------
            # 상태 전환(ready / tracking) 확인
            # --------------------------------------------------------------------------
            if last_position is not None:                # 이전 위치 정보가 있는 경우
                dy = y_center - last_position            # 이전 위치와의 y축 변화량 계산 (예: y=100에서 y=120으로 이동했다면 dy=20)
                movement = abs(dy)                       # 변화량의 절대값 계산 (예: 위로 이동하든 아래로 이동하든 움직임의 크기만 필요)
            else:                                        # 이전 위치 정보가 없는 경우 (첫 프레임)
                movement = 0                             # 움직임을 0으로 설정

            if current_state == "ready":                 # 현재 준비 상태인 경우
                if movement > movement_threshold:        # 움직임이 임계값보다 큰 경우 (예: 10픽셀 이상 움직였다면)
                    current_state = "tracking"           # 상태를 추적 모드로 변경
                    bounce_count = 0                     # 바운스 카운트 초기화
                    bounce_points = []                   # 바운스 발생 지점 목록 초기화
                    bounce_times = []                    # 바운스 발생 시간 목록 초기화
                    previous_bounce_time = None          # 이전 바운스 시간 초기화
                    print("State changed to TRACKING")   # 상태 변경 로그 출력

            if movement > movement_threshold:            # 움직임이 임계값보다 큰 경우 (예: 공이 활발히 움직이는 중)
                if stationary_start_time is not None:    # 정지 시작 시간이 기록되어 있다면
                    stationary_start_time = None         # 정지 시작 시간을 초기화 (공이 다시 움직이기 시작했으므로)
            else:                                        # 움직임이 임계값보다 작은 경우 (예: 공이 거의 정지 상태)
                if stationary_start_time is None:        # 정지 시작 시간이 기록되어 있지 않다면
                    stationary_start_time = time.time()  # 현재 시간을 정지 시작 시간으로 기록
                elif (time.time() - stationary_start_time) >= stationary_threshold:  # 정지 상태가 임계 시간을 넘어선 경우 (예: 2초 이상 정지)
                    if in_rect_time >= 2.0 and current_state != "ready":            # 빨간 사각형 안에 2초 이상 있고, waiting 상태인경우(ready 상태가 아닌 경우)
                        current_state = "ready"          # 상태를 준비 상태로 변경
                        state_change_time = time.time()  # 상태 변경 시간 기록
                        print("State changed to READY")  # 상태 변경 로그 출력

            last_position = y_center                     # 현재 위치를 다음 프레임의 이전 위치로 저장

            # --------------------------------------------------------------------------
            # 바운스 카운트 로직
            # --------------------------------------------------------------------------
            if current_state == "tracking":                                    # 현재 상태가 추적 모드인 경우 (예: 공이 움직이기 시작한 후)
                if last_y is not None:                                        # 이전 y좌표가 있는 경우 (예: 두 번째 프레임부터)
                    dy_tracking = y_center - last_y                           # 현재 y좌표와 이전 y좌표의 차이 계산 (예: y=100에서 y=120으로 이동했다면 dy_tracking=20) opencv에서 밑으로 갈수록 y증가, 위로 갈수록 y감소
                    if abs(dy_tracking) > PIXEL_THRESHOLD:                    # y좌표 변화량이 임계값보다 큰 경우 (예: 5픽셀 이상 움직였을 때)
                        if dy_tracking > 0:                                   # 아래로 움직이는 경우 (예: dy_tracking이 양수)
                            consecutiveDownCount += 1                         # 연속 하강 카운트 증가 (예: 3프레임 연속 하강하면 consecutiveDownCount=3)
                            consecutiveUpCount = 0                            # 연속 상승 카운트 초기화
                        else:                                                 # 위로 움직이는 경우 (예: dy_tracking이 음수)
                            consecutiveUpCount += 1                           # 연속 상승 카운트 증가 (예: 3프레임 연속 상승하면 consecutiveUpCount=3)
                            consecutiveDownCount = 0                          # 연속 하강 카운트 초기화

                        if state is None:                                     # 초기 상태인 경우 (예: 처음 공을 감지했을 때)
                            if consecutiveDownCount >= DOWN_THRESHOLD:        # 연속 하강 횟수가 임계값 이상인 경우 (예: 2프레임 이상 연속 하강)
                                state = "down"                               # 상태를 하강으로 변경
                        elif state == "down":                                # 현재 하강 상태인 경우
                            if consecutiveUpCount >= UP_THRESHOLD:           # 연속 상승 횟수가 임계값 이상인 경우 (예: 1프레임 이상 연속 상승)
                                bounce_count += 1                            # 바운스 횟수 증가 (예: 첫 번째 바운스면 1, 두 번째 바운스면 2)
                                print("Bounce detected!")                    # 바운스 감지 메시지 출력
                                if sound_enabled:                            # 소리 재생이 활성화된 경우
                                    sound.play()                            # 바운스 소리 재생

                                bounce_points.append((x_values[-1], y_values[-1]))  # 바운스 발생 위치 저장 (예: x=100, y=200에서 바운스)
                                current_bounce_time = time.time()                   # 현재 바운스 시간 기록
                                bounce_times.append(current_bounce_time)            # 바운스 시간 목록에 추가

                                if previous_bounce_time is not None:                # 이전 바운스 시간이 있는 경우 (예: 두 번째 이상의 바운스)
                                    td = current_bounce_time - previous_bounce_time # 이전 바운스와의 시간 차이 계산 (예: 1.5초 후 다시 바운스)
                                    print(f"Time diff between last two bounces: {td:.2f} s")
                                    bounce_time_diff = td
                                    if td > 1.0:                                   # 바운스 간격이 1초 이상인 경우 (예: 공이 멈춘 것으로 판단)
                                        bounce_history.append(bounce_count)        # 바운스 기록에 현재까지의 바운스 횟수 저장
                                        if len(bounce_history) > 8:               # 바운스 기록이 8개를 초과하는 경우
                                            bounce_history.pop(0)                 # 가장 오래된 기록 삭제

                                        current_state = "waiting"                 # 상태를 대기 상태로 변경
                                        bounce_count = 0                         # 바운스 카운트 초기화
                                        bounce_points = []                       # 바운스 위치 목록 초기화
                                        bounce_times = []                        # 바운스 시간 목록 초기화
                                        previous_bounce_time = None              # 이전 바운스 시간 초기화
                                        print("State changed to WAITING (timeout)")
                                previous_bounce_time = current_bounce_time       # 현재 바운스 시간을 이전 바운스 시간으로 저장

                                state = "up"                                    # 상태를 상승으로 변경 (예: 바운스 후 공이 올라가는 상태)
                                consecutiveDownCount = 0                        # 연속 하강 카운트 초기화
                                consecutiveUpCount = 0                          # 연속 상승 카운트 초기화
                        elif state == "up":                                     # 현재 상승 상태인 경우
                            if consecutiveDownCount >= DOWN_THRESHOLD:          # 연속 하강 횟수가 임계값 이상인 경우 (예: 2프레임 이상 연속 하강)
                                state = "down"                                 # 상태를 하강으로 변경
                                consecutiveUpCount = 0                         # 연속 상승 카운트 초기화
                                consecutiveDownCount = 0                       # 연속 하강 카운트 초기화

                last_y = y_center                                              # 현재 y좌표를 다음 프레임의 이전 y좌표로 저장

            # --------------------------------------------------------------------------
            # 디버그용 사각형 & 텍스트
            # --------------------------------------------------------------------------
            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)                    # 공 주변에 초록색 사각형 그리기 (예: (100,200)에서 (150,250)까지 두께 2로 그림)
            cv2.putText(
                frame,                                                                       # 텍스트를 그릴 프레임
                f"y_center={int(y_center)}",                                                # y좌표 값 표시 (예: "y_center=200")
                (x1i, y1i - 10),                                                            # 텍스트 위치 - 사각형 위 10픽셀 (예: (100,190))
                cv2.FONT_HERSHEY_SIMPLEX,                                                   # 폰트 종류 - 심플렉스 폰트
                0.7,                                                                        # 폰트 크기 - 0.7배
                (0, 255, 0),                                                               # 텍스트 색상 - 초록색
                2,                                                                         # 텍스트 두께 - 2픽셀
                cv2.LINE_AA                                                                # 선 종류 - 부드러운 선
            )
            cv2.putText(
                frame,                                                                     # 텍스트를 그릴 프레임
                f"Orange px: {orange_pixels}",                                            # 오렌지색 픽셀 수 표시 (예: "Orange px: 150")
                (x1i, y2i + 25),                                                          # 텍스트 위치 - 사각형 아래 25픽셀 (예: (100,275))
                cv2.FONT_HERSHEY_SIMPLEX,                                                 # 폰트 종류 - 심플렉스 폰트
                0.7,                                                                      # 폰트 크기 - 0.7배
                (0, 165, 255),                                                           # 텍스트 색상 - 주황색
                2,                                                                       # 텍스트 두께 - 2픽셀
                cv2.LINE_AA                                                              # 선 종류 - 부드러운 선
            )
        else:
            y_values.append(None)                                                        #(공이 화면에 있는데 오렌지 픽셀 필터링 때문에) 공이 감지되지 않으면 y좌표에 None 추가 (예: y_values=[200, None, 195])
            orange_pixel_values.append(None)                                            # (공이 화면에 있는데 오렌지 픽셀 필터링 때문에) 공이 감지되지 않으면 오렌지픽셀 수에 None 추가 (예: orange_pixel_values=[150, None, 148])
    else:
        y_values.append(None)                                                          #(실제로 공이 화면에 없어서) 공이 감지되지 않으면 y좌표에 None 추가 (예: y_values=[200, None, 195])
        orange_pixel_values.append(None)                                              #(실제로 공이 화면에 없어서) 공이 감지되지 않으면 오렌지픽셀 수에 None 추가 (예: orange_pixel_values=[150, None, 148])

    # -------------------------------------------------------------------------
    # (1) 공이 사각형 안에 있는 동안 => in_rect_time = 현재시간 - 진입시점
    # (2) 공이 나가면 => in_rect_time = 0
    # -------------------------------------------------------------------------
    if len(boxes) > 0 and detected:                                                # 공이 감지되었는지 확인 (예: boxes=[array([100,200,150,250])])
        # 공 중심좌표 (x_center, y_center)가 사각형 내부인지 확인
        if (drag_rect_x <= x_center < drag_rect_x + drag_rect_w and              # x좌표가 사각형 내부인지 확인 (예: 100 <= 120 < 100+200)
            drag_rect_y <= y_center < drag_rect_y + drag_rect_h):                # y좌표가 사각형 내부인지 확인 (예: 200 <= 220 < 200+150)
            if ball_in_rect_start is None:                                       # 공이 처음 사각형에 들어온 경우 (예: ball_in_rect_start=None)
                ball_in_rect_start = time.time()                                 # 진입 시점 기록 (예: ball_in_rect_start=1234567.89)
            in_rect_time = time.time() - ball_in_rect_start                     # 사각형 내 체류 시간 계산 (예: in_rect_time=1.23)
        else:                                                                    # 공이 사각형 밖에 있는 경우
            in_rect_time = 0.0                                                  # 체류 시간 초기화 (예: in_rect_time=0.0)
            ball_in_rect_start = None                                           # 진입 시점 초기화 (예: ball_in_rect_start=None)
    else:                                                                       # 공이 감지되지 않은 경우
        in_rect_time = 0.0                                                     # 체류 시간 초기화 (예: in_rect_time=0.0)
        ball_in_rect_start = None                                              # 진입 시점 초기화 (예: ball_in_rect_start=None)

    # -------------------------------------------------------------------------
    # state==ready 인 상태에서 공이 안보이는(=detected=False) 1초 경과 시 waiting으로
    # -------------------------------------------------------------------------
    if current_state == "ready":                                                                # 현재 상태가 "ready"인지 확인 (예: current_state="ready")
        if last_detection_time is not None and (time.time() - last_detection_time) > 1.0:      # 마지막 감지 시간이 존재하고 1초 이상 지났는지 확인 (예: last_detection_time=1234567.89, time.time()=1234569.0)
            current_state = "waiting"                                                           # 상태를 "waiting"으로 변경 (예: current_state="waiting")
            print("State changed to WAITING (no detection for 1s in READY)")                   # 상태 변경 메시지 출력 (예: "State changed to WAITING (no detection for 1s in READY)")

    # -------------------------------------------------------------------------
    # "TRACKING" → "WAITING" 조건(1): 마지막 검출 이후 1초 이상 감지 X
    # -------------------------------------------------------------------------
    if current_state == "tracking":                                                                # 현재 상태가 "tracking"인지 확인 (예: current_state="tracking")
        if last_detection_time is not None and (time.time() - last_detection_time) >= 1.0:        # 마지막 감지 시간이 존재하고 1초 이상 지났는지 확인 (예: last_detection_time=1234567.89, time.time()=1234569.0)
            bounce_history.append(bounce_count)                                                    # 현재 바운스 카운트를 기록에 추가 (예: bounce_history=[5,7,3] -> [5,7,3,4])
            if len(bounce_history) > 8:                                                           # 바운스 기록이 8개를 초과하는지 확인 (예: len(bounce_history)=9)
                bounce_history.pop(0)                                                             # 가장 오래된 바운스 기록 제거 (예: bounce_history=[5,7,3,4] -> [7,3,4])

            bounce_count = 0                                                                      # 바운스 카운트 초기화 (예: bounce_count=4 -> bounce_count=0)
            consecutiveDownCount = 0                                                              # 연속 하강 카운트 초기화 (예: consecutiveDownCount=2 -> consecutiveDownCount=0)
            consecutiveUpCount = 0                                                                # 연속 상승 카운트 초기화 (예: consecutiveUpCount=1 -> consecutiveUpCount=0)
            state = None                                                                          # 공의 이동 상태 초기화 (예: state="down" -> state=None)
            current_state = "waiting"                                                             # 현재 상태를 "waiting"으로 변경 (예: current_state="tracking" -> current_state="waiting")
            print("No detection for 1 second in TRACKING => bounce_count reset to 0, state changed to WAITING")  # 상태 변경 메시지 출력

    # 그래프 데이터 길이 제한
    if len(x_values) > MAX_POINTS:                                                # x좌표 리스트가 최대 길이를 초과하는지 확인 (예: len(x_values)=101 > MAX_POINTS=100)
        x_values.pop(0)                                                          # x좌표 리스트의 첫 번째 요소 제거 (예: x_values=[100,105,110] -> [105,110])
        y_values.pop(0)                                                          # y좌표 리스트의 첫 번째 요소 제거 (예: y_values=[200,195,190] -> [195,190])
        orange_pixel_values.pop(0)                                              # 오렌지픽셀 수 리스트의 첫 번째 요소 제거 (예: orange_pixel_values=[150,148,152] -> [148,152])

    # 바운스 연속 감지 제한 (Optional)
    if last_bounce_time is not None:                                                # 마지막 바운스 시간이 존재하는지 확인 (예: last_bounce_time=1234567.89)
        if time.time() - last_bounce_time > CONTINUOUS_TIMEOUT:                     # 마지막 바운스로부터 1초 이상 지났는지 확인 (예: time.time()=1234569.0, last_bounce_time=1234567.89)
            bounce_count = 0                                                        # 바운스 카운트 초기화 (예: bounce_count=5 -> bounce_count=0)
            last_bounce_time = None                                                 # 마지막 바운스 시간 초기화 (예: last_bounce_time=1234567.89 -> None)
            print("No bounce for a while -> reset bounce_count to 0")              # 초기화 메시지 출력

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
