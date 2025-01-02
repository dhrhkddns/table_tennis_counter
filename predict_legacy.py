import cv2
import numpy as np
from ultralytics import YOLO

# ----- YOLO 모델 로드 -----
model = YOLO(r"C:\Users\omyra\Desktop\coding\ping_pong\Ping-Pong-Detection-3\Results\weights\best.pt")

# ----- 카메라 열기 -----
cap = cv2.VideoCapture(1)  # 0 또는 1 등 자신의 웹캠 번호

# ----- 그래프 데이터 저장용 -----
x_values = []  # 프레임 번호
y_values = []  # 공의 중심 y좌표
frame_count = 0

# ----- 그래프 그리는 함수 -----
def draw_graph(x_data, y_data, width=800, height=400, max_y=720):
    """
    x_data: [0, 1, 2, ...] 프레임 번호
    y_data: 각 프레임에서 구한 공의 중심 y좌표
    width, height: 그래프를 그릴 OpenCV 이미지 크기
    max_y: 실제 y좌표의 최대값(카메라 해상도 등 고려)
    return: 그래프가 그려진 BGR 이미지 (numpy array)
    """
    # 새 검은 배경 이미지 생성
    graph_img = np.zeros((height, width, 3), dtype=np.uint8)

    # 데이터가 2점 미만이면 라인을 그릴 수 없음
    if len(x_data) < 2:
        return graph_img

    # x축 최대값 (프레임 카운트가 계속 늘어날 수 있으므로)
    # 여기서는 x_data의 마지막 값으로 스케일링
    max_x = x_data[-1] if x_data[-1] != 0 else 1

    # (x_data[i], y_data[i]) -> 그래프 상 좌표로 매핑
    for i in range(len(x_data) - 1):
        # 원본 좌표
        x1_ori, y1_ori = x_data[i],   y_data[i]
        x2_ori, y2_ori = x_data[i+1], y_data[i+1]

        # 그래프 폭에 맞춰 x좌표 스케일링
        #   x=0 -> left=0,  x=max_x -> right=width-1
        x1 = int(x1_ori / max_x * (width - 1))
        x2 = int(x2_ori / max_x * (width - 1))

        # 그래프 높이에 맞춰 y좌표 스케일링
        #   y=0   -> bottom=height-1
        #   y=max_y -> top=0
        y1 = height - 1 - int(y1_ori / max_y * (height - 1))
        y2 = height - 1 - int(y2_ori / max_y * (height - 1))

        # 두 점을 연결하는 선 그리기 (녹색)
        cv2.line(graph_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
        # 첫 번째 박스의 xyxy 좌표 (GPU 텐서이므로 CPU로 변환)
        x1, y1, x2, y2 = boxes[0].xyxy[0].cpu().numpy()

        # 공의 중심 y좌표 (y_center)
        y_center = (y1 + y2) / 2.0

        # 그래프 데이터 저장
        x_values.append(frame_count)
        y_values.append(y_center)

        frame_count += 1

        # # 만약 데이터가 너무 많아지면 앞부분 일부를 버릴 수도 있음 (옵션)
        # if len(x_values) > 500:
        #     x_values.pop(0)
        #     y_values.pop(0)

        # 바운딩 박스 그리기
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"y_center={int(y_center)}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

    # ----- Detection 결과 프레임 표시 -----
    cv2.imshow("Detection", frame)

    # ----- 그래프 그리기 -----
    graph_img = draw_graph(x_values, y_values, width=800, height=400, max_y=720)
    cv2.imshow("Graph", graph_img)

    # ESC 키(27) 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
