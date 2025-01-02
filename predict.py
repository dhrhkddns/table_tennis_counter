import cv2
import numpy as np
import time
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO(r"C:\Users\omyra\Desktop\coding\ping_pong\Ping-Pong-Detection-3\Results\weights\best.pt")

# 카메라 열기
cap = cv2.VideoCapture(1)  # 본인 환경에 맞춰 조정

# 그래프 데이터 저장용
x_values = []
y_values = []
frame_count = 0
MAX_POINTS = 100

# ----- 바운스 측정 관련 변수 -----
bounce_count = 0

# '연속 down', '연속 up' 카운트
consecutiveDownCount = 0
consecutiveUpCount = 0

# 상태: None, "down", "up"
state = None

# 바운스 판단 기준(연속 n 프레임 down → 연속 m 프레임 up)
DOWN_THRESHOLD = 2
UP_THRESHOLD = 1

# 픽셀 차 임계값(잔떨림 노이즈 방지용)
PIXEL_THRESHOLD = 4.0

last_y = None

# ----- 바운스 발생 시점의 (x, y) 좌표를 저장할 리스트 -----
bounce_points = []  # 예: [(x_val, y_val), (x_val, y_val), ...]

# ----- 연속 모드 관련 설정 -----
CONTINUOUS_TIMEOUT = 2.0  # 3초
last_bounce_time = None   # 마지막 바운스 발생 시각

def draw_graph(x_data, y_data, width=640, height=480, max_y=480, bounce_pts=None):
    """
    x_data: [0, 1, 2, ...] 프레임 번호(혹은 index)
    y_data: 각 프레임에서 구한 공의 중심 y좌표
    width, height: 그래프를 그릴 OpenCV 이미지 크기
    max_y: 실제 y좌표의 최대값(카메라 해상도 등 고려)
    bounce_pts: 바운스 발생 시점 좌표 목록 [(x, y), ...]
    return: 그래프가 그려진 BGR 이미지 (numpy array)
    """
    if bounce_pts is None:
        bounce_pts = []

    graph_img = np.zeros((height, width, 3), dtype=np.uint8)
    if len(x_data) < 2:
        return graph_img

    # x_data 중 마지막 값(가장 큰 프레임 번호)
    max_x = x_data[-1] if x_data[-1] != 0 else 1
    
    # ----- 선(Line) 그리기 -----
    for i in range(len(x_data) - 1):
        x1_ori, y1_ori = x_data[i],   y_data[i]
        x2_ori, y2_ori = x_data[i+1], y_data[i+1]

        # X 좌표 스케일
        x1 = int((x1_ori - x_data[0]) / (max_x - x_data[0] + 1e-6) * (width - 1))
        x2 = int((x2_ori - x_data[0]) / (max_x - x_data[0] + 1e-6) * (width - 1))

        # Y 좌표 스케일 (y=0 -> 아래, y=max_y -> 위)
        y1 = int(y1_ori / max_y * (height - 1))
        y2 = int(y2_ori / max_y * (height - 1))

        # 선은 녹색
        cv2.line(graph_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ----- 파란색 점 찍기 -----
    for i in range(len(x_data)):
        x_ori, y_ori = x_data[i], y_data[i]

        x_pt = int((x_ori - x_data[0]) / (max_x - x_data[0] + 1e-6) * (width - 1))
        y_pt = int(y_ori / max_y * (height - 1))

        # 파란색 원 (반지름 4)
        cv2.circle(graph_img, (x_pt, y_pt), 4, (255, 0, 0), -1)

    # ----- 바운스 점(빨간색) 그리기 -----
    for (bx_ori, by_ori) in bounce_pts:
        if bx_ori < x_data[0]:
            continue
        bx = int((bx_ori - x_data[0]) / (max_x - x_data[0] + 1e-6) * (width - 1))
        by = int(by_ori / max_y * (height - 1))

        # 빨간색 원 (반지름=5)
        cv2.circle(graph_img, (bx, by), 5, (0, 0, 255), -1)

    return graph_img

while True:
    ret, frame = cap.read()
    if not ret:
        print("No more frames or camera error.")
        break

    # ----- YOLO 추론 -----
    results = model.predict(frame, imgsz=640, conf=0.5, max_det=1, show=False)
    boxes = results[0].boxes

    if len(boxes) > 0:
        x1, y1, x2, y2 = boxes[0].xyxy[0].cpu().numpy()

        # 공의 중심 y좌표
        y_center = (y1 + y2) / 2.0

        # 그래프 데이터
        x_values.append(frame_count)
        y_values.append(y_center)
        frame_count += 1

        if len(x_values) > MAX_POINTS:
            x_values.pop(0)
            y_values.pop(0)

        # ----- Down/Up 판단 로직 -----
        if last_y is not None:
            dy = y_center - last_y

            # 노이즈 제거용 임계값
            if abs(dy) > PIXEL_THRESHOLD:
                if dy > 0:
                    # Down
                    consecutiveDownCount += 1
                    consecutiveUpCount = 0
                else:
                    # Up
                    consecutiveUpCount += 1
                    consecutiveDownCount = 0

                # ----- state 전환 -----
                if state is None:
                    if consecutiveDownCount >= DOWN_THRESHOLD:
                        state = "down"

                elif state == "down":
                    # Down 상태에서 Up으로 전환되면 bounce
                    if consecutiveUpCount >= UP_THRESHOLD:
                        bounce_count += 1
                        print("Bounce detected!")

                        # 바운스 발생 시점 좌표 저장
                        bounce_points.append((x_values[-1], y_values[-1]))

                        # 마지막 바운스 시각 갱신
                        last_bounce_time = time.time()

                        state = "up"
                        consecutiveDownCount = 0
                        consecutiveUpCount = 0

                elif state == "up":
                    # Up->Down 전환 시에도 바운스 치고 싶으면 여기에 로직 추가
                    if consecutiveDownCount >= DOWN_THRESHOLD:
                        # bounce_count += 1 # 필요한 경우 활성화
                        # last_bounce_time = time.time()
                        state = "down"
                        consecutiveUpCount = 0
                        consecutiveDownCount = 0

        last_y = y_center

        # 바운딩 박스 표시
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"y_center={int(y_center)}",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, 
            (0,255,0),
            2,
            cv2.LINE_AA
        )

    # ----- 연속 모드: 일정 시간 바운스가 없으면 0으로 리셋 -----
    # last_bounce_time이 None이 아니고,
    # (현재시간 - last_bounce_time)가 CONTINUOUS_TIMEOUT를 넘으면 bounce_count=0
    if last_bounce_time is not None:
        if time.time() - last_bounce_time > CONTINUOUS_TIMEOUT:
            # reset
            bounce_count = 0
            last_bounce_time = None  # 다시 None으로 초기화 (선택사항)
            print("No bounce for a while -> reset bounce_count to 0")

    # 원본 프레임 리사이즈 (그래프와 동일 크기)
    frame_resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

    # 그래프 생성
    graph_img = draw_graph(
        x_values,
        y_values,
        width=640,
        height=480,
        max_y=480,
        bounce_pts=bounce_points
    )

    # 가로로 합침
    combined_img = np.hstack((frame_resized, graph_img))

    # 바운스 카운트 표시
    cv2.putText(
        combined_img,
        f"Bounce Count: {bounce_count}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2
    )

    cv2.imshow("Combined", combined_img)

    # ESC 키(27) 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
