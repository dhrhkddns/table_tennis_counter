-pc방이나 다른 컴퓨터에서 conda gpu로 테스트 할때 콘다 gpu환경이 설치가 안되는 오류가 나올때가 있다. (똑같이 내 노트북에서는 됨에도 불구하고)
그럴때는 버전을 바꿔서 설치해보는 걸 추천한다.

conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=12.4 ultralytics

![conda_gpu_설치세팅](https://github.com/user-attachments/assets/2184e887-56a9-4c23-808b-cd1eda280c46)

(ultralytics 홈페이지에서는 pytorch-cuda 패키지가 11.8로 있는데, 이 줄을 그대로 복사해서 실행하면 설치 오류가 나는 경우가 있는데 12.4로 변경하면 잘되는 경우가 있다.)



-FHD60F 웹캠은 직접 USB 단자 연결시 60FPS, 그러나 USB 허브를 이용해서 간접적 연결을 할시 50FPS 나오는 현상이 있다. (단순파일, 복잡 게임파일 동일)
![KakaoTalk_20250202_005853950_01](https://github.com/user-attachments/assets/fd4ee805-fabe-4fe7-8ad9-914282cad236)

-스피커 싱크가 어떤거는 살짝 안맞기도 함.

![설정 2025-01-22 오후 9_27_15](https://github.com/user-attachments/assets/b5b1263c-6c79-4c8a-9229-493a353d634f)

HDMI연결시 어떤 이유 때문인지 스피커에 따라 속도가 달라진다.
추가 모니터의 소리는 낮으므로, 노트북을 원래 소리로하고.
노트북에서도 2가지로 나눠지는데 REALTECH 오디오 >> OMEN CAM & VOICE보다 조금 더 빠르고, 이게 실제로 탁구를 할때 씹히는 느낌이 훨씬 덜든다.
참고하도록.

-웹캠 카메라에 따라서 어떤거는 야외에서 죽을쑤고, 어떤거는 야외에서도 충분히 무난한 플레이가 된다
프레임의 강자 FHD60F는 60프레임을 CAP_DSHOW와 코덱 MJPG으로 하면 진짜 좋지만, 야외에서는 빛번짐이 너무 심해서 실내 원툴이다.
로지텍의 C230 뭐시기는 30프레임을 유지하지만 야외에서도 충분히 쓰고 화각도 나쁘지 않다.
상황에 따라 유연한 선택을 할것.

![LOGITEC](https://github.com/user-attachments/assets/8ad6ec0f-0187-4b61-85d9-36a6d6507566)

![Video 2025-01-22 오후 2_50_53](https://github.com/user-attachments/assets/f672c42d-9cf7-47e4-8224-2271ebd89b3b)


![Uploading Combined 2025-01-22 오후 3_00_08.png…]()

GLOBAL을 이용해서 변수를 바꿔서 체크해야할일이 있으면 일일이 바꾸고 다시 종료후 실행보다
트랙바를 만들어서 유동적인 변수 조절로 최적의 값 범위를 찾아보자.

ORANGE 픽셀 필터를 적용할 때의 장점은 FP 실제로 있지 않은데 POSITIVE 있다고 바운딩 박스가 뜨는 현상 (목 부분을 탁구공으로 인식한다는지)
그치만 멀리서 적용할때는 30픽셀도 안되기 때문에 50 픽셀을 THRESHOLD로 잡으면 이게 왜 안되지라고 생각해버린다.
(데이터셋도 일일이 다시 추가하는 노가다 할뻔...) 필터링이 우선은 아니라는것! 거리를 생각하면 해제 하는게 맞다.

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

상대적인 공간 기하학적 조정 없이

그냥 조커에서

특정 위치 안에 공의 궤적중 한점이 포함되면 추가점수 주도록

(이러면 사각형을 임의로 생기게 해야함 사라지는것도)

혹은 공의 각도가 수평을 유지하면 주는 조커도...

