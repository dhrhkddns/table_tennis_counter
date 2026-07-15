"""오른쪽 → P → 3초 대기 → 반복. Ctrl+C로 종료."""

import ctypes
import time

VK_RIGHT = 0x27
VK_P = 0x50
KEYEVENTF_KEYUP = 0x0002


def press_key(vk: int) -> None:
    ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
    ctypes.windll.user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)


def main() -> None:
    wait_sec = 11.0
    print("매크로: Right → P → 3초 대기 → 반복")
    print("3초 후 시작합니다. 대상 창을 활성화하세요.")
    print("종료: Ctrl+C\n")

    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    count = 0
    try:
        while True:
            press_key(VK_RIGHT)
            time.sleep(0.15)
            press_key(VK_P)
            count += 1
            print(f"[{count}] Right → P (3초 대기)")
            time.sleep(wait_sec)
    except KeyboardInterrupt:
        print(f"\n종료 (총 {count}회)")


if __name__ == "__main__":
    main()
