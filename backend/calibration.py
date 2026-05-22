"""RGB ↔ Thermal 호모그래피 캘리브레이션.

자동 손끝 매칭 방식:
  - 사용자가 손을 화면 한 위치에 들고 capture 트리거
  - RGB pose의 wrist 좌표 + Thermal max 좌표를 한 페어로 저장
  - 4쌍 이상 모이면 cv2.findHomography로 RGB→Thermal 매핑 행렬 계산
  - 결과는 calibration.npz로 저장, 다음 실행에 자동 로드
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "calibration.npz"


class Calibration:
    """RGB → Thermal 호모그래피 보유 + 페어 누적 관리."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.pairs: list[dict] = []
        self.H: Optional[np.ndarray] = None
        self._load()

    def _load(self) -> None:
        if not CALIBRATION_PATH.exists():
            return
        try:
            data = np.load(CALIBRATION_PATH, allow_pickle=True)
            self.H = data["H"]
            if "pairs" in data:
                self.pairs = list(data["pairs"])
            print(f"[calib] 기존 호모그래피 로드: {CALIBRATION_PATH}")
        except Exception as e:
            print(f"[calib] 로드 실패: {e}")

    def _save(self) -> None:
        if self.H is None:
            return
        np.savez(
            CALIBRATION_PATH,
            H=self.H,
            pairs=np.array(self.pairs, dtype=object),
        )

    def add_pair(self, rgb: dict, thermal: dict) -> None:
        with self._lock:
            self.pairs.append({"rgb": rgb, "thermal": thermal})
            if len(self.pairs) >= 4:
                self._compute_locked()

    def reset(self) -> None:
        with self._lock:
            self.pairs.clear()
            self.H = None
            if CALIBRATION_PATH.exists():
                try:
                    CALIBRATION_PATH.unlink()
                except Exception:
                    pass

    def _compute_locked(self) -> None:
        src = np.array(
            [[p["rgb"]["x"], p["rgb"]["y"]] for p in self.pairs],
            dtype=np.float32,
        )
        dst = np.array(
            [[p["thermal"]["x"], p["thermal"]["y"]] for p in self.pairs],
            dtype=np.float32,
        )
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            print("[calib] findHomography 실패 — 점이 일직선이거나 너무 비슷할 수 있음")
            return
        self.H = H
        self._save()
        print(f"[calib] H 계산 완료 ({len(self.pairs)}쌍)")

    def status(self) -> dict:
        with self._lock:
            return {
                "pairs": list(self.pairs),
                "pair_count": len(self.pairs),
                "homography_ready": self.H is not None,
                "homography": self.H.tolist() if self.H is not None else None,
            }

    def map_point(self, rgb_x: float, rgb_y: float) -> Optional[tuple[int, int]]:
        """RGB 좌표 → Thermal 좌표 변환. H 없으면 None."""
        with self._lock:
            if self.H is None:
                return None
            pt = np.array([[[rgb_x, rgb_y]]], dtype=np.float32)
            mapped = cv2.perspectiveTransform(pt, self.H)
        return int(mapped[0, 0, 0]), int(mapped[0, 0, 1])

    def map_bbox(self, x: int, y: int, w: int, h: int) -> Optional[dict]:
        """RGB 사각형 → Thermal 좌표계의 4코너 + AABB."""
        with self._lock:
            if self.H is None:
                return None
            corners = np.array(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                dtype=np.float32,
            ).reshape(-1, 1, 2)
            mapped = cv2.perspectiveTransform(corners, self.H).reshape(-1, 2)
        xs = mapped[:, 0]
        ys = mapped[:, 1]
        return {
            "corners": mapped.astype(int).tolist(),
            "bbox": {
                "x": int(xs.min()),
                "y": int(ys.min()),
                "w": int(xs.max() - xs.min()),
                "h": int(ys.max() - ys.min()),
            },
        }
