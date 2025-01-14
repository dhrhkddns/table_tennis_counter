
FPS 벽인 60일 뛰어넘는 법. (소 뒷걸음치다 쥐를 잡은 느낌)

1.웹캠이 일단 지원해야함.
2.USB도 빠른 처리 속도 3.0 3.1 지원하는지 확인
3.허브는 되도록 사용 
4.cv2.CAP_DSHOW 사용하기. (적용하지 않으면 cv2.CAP_MSMF 기본값으로 설정)
5.코덱도 웹캠 제조사에서 지원하는 걸로 적용해야함. 일반적으로 적용 안하면 YUV2 코덱으로 압축없이 되서 30 혹은 60에서 못벗어남. 
6.cap.set으로 width, height 혹은 fps 설정해버리면, 카메라가 없는 거 설정한다고 드라이버에서 받아들여서 기본값 (30)으로 갈 위험이 있음. (아무리 고프레임을 지원해도
고로 cap.set은 사용하지 않을것.


https://github.com/user-attachments/assets/1e34558e-639d-42d1-b98b-b72000bafc64





https://github.com/user-attachments/assets/1ceb9769-e8b9-4c84-a8cf-a78e7988e6fb

