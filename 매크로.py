"""특정 창(백그라운드)에 Right → P → 대기 → 반복 키를 보내는 매크로.

keybd_event 는 활성 창에만 입력되므로, 여기서는 PostMessage 로
사용자가 선택한 창(HWND)에 직접 키 메시지를 보낸다.
이렇게 하면 그 창이 백그라운드에 있어도(다른 작업 중이어도) 동작한다.

주의: 일부 게임(DirectInput/RawInput 사용)은 PostMessage 입력을 무시할 수 있다.

종료: Ctrl+C
"""

import ctypes
from ctypes import wintypes
import time

user32 = ctypes.windll.user32

# --- Windows 메시지/가상키 상수 ---
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
VK_RIGHT = 0x27
VK_P = 0x50

# 확장키(방향키 등)는 lParam 24번 비트를 세팅해야 한다.
KF_EXTENDED = 0x01000000

# 확장키로 취급할 가상키 (방향키/Home/End/PageUp/Down/Insert/Delete 등)
EXTENDED_KEYS = {0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E, 0x24, 0x23, 0x21, 0x22}


def _make_lparam(vk: int, key_up: bool) -> int:
    """PostMessage 용 lParam 구성 (repeat=1, scancode, 확장키/릴리즈 플래그)."""
    scan = user32.MapVirtualKeyW(vk, 0)  # MAPVK_VK_TO_VSC
    lparam = 1 | (scan << 16)
    if vk in EXTENDED_KEYS:
        lparam |= KF_EXTENDED
    if key_up:
        lparam |= 0xC0000000  # transition(31) + previous-state(30) 비트
    return lparam


def list_windows():
    """제목이 있는 최상위 보이는 창 목록 반환: [(hwnd, title), ...]."""
    results = []

    EnumWindowsProc = ctypes.WINFUNCTYPE(
        wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
    )

    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.strip()
        if title:
            results.append((hwnd, title))
        return True

    user32.EnumWindows(EnumWindowsProc(callback), 0)
    return results


def choose_window():
    """창 목록을 보여주고 사용자가 번호로 선택하게 한다."""
    windows = list_windows()
    print("=== 열려 있는 창 목록 ===")
    for i, (hwnd, title) in enumerate(windows):
        print(f"  [{i}] {title}")
    print()

    while True:
        raw = input("명령을 보낼 창 번호를 입력하세요: ").strip()
        if not raw.isdigit():
            print("숫자를 입력하세요.")
            continue
        idx = int(raw)
        if 0 <= idx < len(windows):
            return windows[idx]
        print("범위를 벗어난 번호입니다.")


def send_key(hwnd: int, vk: int) -> None:
    """지정 창에 키 다운/업 메시지를 보낸다(백그라운드 동작)."""
    user32.PostMessageW(hwnd, WM_KEYDOWN, vk, _make_lparam(vk, False))
    time.sleep(0.03)
    user32.PostMessageW(hwnd, WM_KEYUP, vk, _make_lparam(vk, True))


def main() -> None:
    wait_sec = 10.0

    hwnd, title = choose_window()
    print(f"\n선택된 창: {title}")
    print(f"매크로: Right → P → {wait_sec:.0f}초 대기 → 반복")
    print("이 창(콘솔)은 그대로 두고 다른 작업을 하셔도 됩니다.")
    print("종료: Ctrl+C\n")

    count = 0
    try:
        while True:
            # 창이 닫혔는지 확인
            if not user32.IsWindow(hwnd):
                print("대상 창이 닫혔습니다. 종료합니다.")
                break
            send_key(hwnd, VK_RIGHT)
            time.sleep(0.15)
            send_key(hwnd, VK_P)
            count += 1
            print(f"[{count}] Right → P ({wait_sec:.0f}초 대기)")
            time.sleep(wait_sec)
    except KeyboardInterrupt:
        print(f"\n종료 (총 {count}회)")


if __name__ == "__main__":
    main()
