"""위로 밀려 나가는 카운터

STYLE_ID = 1
STYLE_NAME = "slide_up"
"""
from __future__ import annotations

from transitions._base import (
    clamp,
    lerp,
    ease_linear,
    ease_smooth,
    ease_in_cubic,
    ease_out_cubic,
    ease_in_out_cubic,
    ease_out_back,
    ease_in_back,
    ease_out_elastic,
    ease_out_bounce,
)

STYLE_ID = 1
STYLE_NAME = "slide_up"
STYLE_DESC = "위로 밀려 나가는 카운터"


def layers(p: float) -> list[dict]:
    """progress(0~1) → 그릴 숫자 레이어 스펙 목록.

    각 dict: role('old'|'new'), alpha, ox, oy, scale
    """
    p = clamp(p)

    e = ease_smooth(p)
    s = 22.0
    # (dx,dy): old가 빠져나가는 방향 (위=-y)
    dx, dy = 0.0, -1.0
    return [
        {"role": "old", "alpha": 1.0 - e, "ox": s * dx * e, "oy": s * dy * e, "scale": 1.0},
        {"role": "new", "alpha": e, "ox": -s * dx * (1.0 - e), "oy": -s * dy * (1.0 - e), "scale": 1.0},
    ]
