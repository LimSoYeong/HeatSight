"""RGB + 열화상 듀얼 라이브 뷰어 + 호모그래피 캘리브레이션.

화면 구성:
  좌측 패널 (640x480): MacBook 내장 RGB 카메라
  우측 패널 (640x480): Cellplus 열화상 카메라

키 조작:
  q     종료
  s     합성 프레임을 snapshots/에 PNG로 저장
  c     캘리브레이션 모드 토글
  r     캘리브레이션 점 초기화
  o     오버레이 모드 토글 (캘리브레이션 결과로 RGB 위에 열화상을 정합·합성)
  ]     컬러맵 다음 (Cellplus 24개 중 +1)
  [     컬러맵 이전 (Cellplus 24개 중 -1)

캘리브레이션 사용법:
  1. 'c'로 캘리브레이션 모드 진입.
  2. 좌측(RGB)에서 한 점을 클릭한 뒤, 우측(Thermal)에서 같은 실세계 지점을 클릭.
     이 한 쌍이 1개의 대응점이 된다.
  3. 4쌍 이상 모이면 자동으로 호모그래피가 계산되어 calibration.npz에 저장된다.
  4. 'o'로 오버레이를 켜면 열화상이 RGB 좌표계로 워프되어 inferno 컬러맵으로 합성된다.

캘리브레이션 점을 찍을 때 팁:
  - 두 카메라가 화각이 다르므로 양쪽에 모두 또렷이 보이는 지점을 고를 것.
  - 사람 몸은 3D라 정확도가 떨어진다. 평면(노트북 화면 모서리, 종이 사각형)을
    사용자 위치와 비슷한 거리에 두고 그 코너 4점을 잡는 게 가장 정확.
  - 열화상에 보이도록 따뜻한 머그컵·핫팩 등을 사각형으로 배치하면 양쪽 모두 명확.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from capture import RGBSource, ThermalSource
from thermal_control import (
    CellplusControl,
    COLORMAP_PRESETS,
    find_control_port,
    next_preset,
)

CALIB_PATH = Path("calibration.npz")
SNAPSHOT_DIR = Path("snapshots")
PANEL_W, PANEL_H = 640, 480


def _verify_sources(rgb_frame: np.ndarray, thermal_frame: np.ndarray) -> None:
    """RGB가 진짜 컬러이고 Thermal이 사실상 grayscale인지 빠른 점검.

    AVFoundation enumeration이 흔들려 두 소스가 같은 카메라로 매핑되는
    과거 버그가 있었으므로 매 실행마다 첫 프레임으로 검증한다.
    """
    def color_score(img: np.ndarray) -> float:
        b = img[:, :, 0].astype(np.int32)
        g = img[:, :, 1].astype(np.int32)
        r = img[:, :, 2].astype(np.int32)
        return float(max(np.abs(b - g).mean(),
                         np.abs(g - r).mean(),
                         np.abs(b - r).mean()))

    rgb_c = color_score(rgb_frame)
    th_c = color_score(thermal_frame)
    print(f"[verify] RGB color={rgb_c:.2f}  Thermal color={th_c:.2f}")
    if rgb_c < 3:
        print(f"[verify] ⚠️  RGB 카메라가 컬러가 아님 — 매핑 잘못됐을 가능성")
    if th_c > 3:
        print(f"[verify] ⚠️  Thermal 카메라에서 컬러가 검출됨 — 매핑 잘못됐을 가능성")


def fit_to(img: np.ndarray, w: int, h: int) -> tuple[np.ndarray, tuple[int, int, float]]:
    """비율 유지로 (w, h) 박스에 맞추고 여백은 검정. (캔버스, (offset_x, offset_y, scale)) 반환."""
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas, (x, y, scale)


def draw_text(img: np.ndarray, text: str, org: tuple[int, int],
              color: tuple[int, int, int] = (255, 255, 255), scale: float = 0.5) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


class DualViewer:
    def __init__(self) -> None:
        self.rgb_pts: list[tuple[float, float]] = []
        self.thermal_pts: list[tuple[float, float]] = []
        self.expecting: str = "rgb"
        self.calibrating: bool = False
        self.overlay: bool = False
        self.H_t_to_r: np.ndarray | None = None
        self._rgb_meta = (0, 0, 1.0)
        self._th_meta = (0, 0, 1.0)
        # Cellplus 시리얼 컨트롤 상태
        self.cellplus: CellplusControl | None = None
        self.colormap_idx: int = 0
        self.colormap_count: int = 0
        self._load_calibration()

    def _open_cellplus(self) -> None:
        """카메라 시리얼 컨트롤 포트를 찾아 열고 현재 컬러맵 상태를 읽는다."""
        port = find_control_port()
        if port is None:
            print("[cellplus] 컨트롤 포트를 못 찾음 — 컬러맵 키([/])는 비활성")
            return
        try:
            ctl = CellplusControl(port).open()
            self.colormap_count = ctl.get_colormap_count()
            current = ctl.get_colormap()
            if current not in COLORMAP_PRESETS:
                current = COLORMAP_PRESETS[0]
                ctl.set_colormap(current)
            self.colormap_idx = current
            self.cellplus = ctl
            print(f"[cellplus] {port}  colormap={current}  presets={COLORMAP_PRESETS}")
        except Exception as e:
            print(f"[cellplus] 초기화 실패: {e}")

    def _bump_colormap(self, delta: int) -> None:
        if self.cellplus is None or self.colormap_count == 0:
            print("[cellplus] 컨트롤이 활성화되지 않음")
            return
        new_idx = next_preset(self.colormap_idx, delta)
        try:
            self.cellplus.set_colormap(new_idx)
            self.colormap_idx = new_idx
            pos = COLORMAP_PRESETS.index(new_idx)
            print(f"[cellplus] colormap → {new_idx}  (preset {pos+1}/{len(COLORMAP_PRESETS)})")
        except Exception as e:
            print(f"[cellplus] 컬러맵 변경 실패: {e}")

    def _load_calibration(self) -> None:
        if CALIB_PATH.exists():
            data = np.load(CALIB_PATH)
            self.H_t_to_r = data["H_t_to_r"]
            print(f"[calib] 기존 호모그래피 로드: {CALIB_PATH}")

    def _save_calibration(self) -> None:
        np.savez(
            CALIB_PATH,
            H_t_to_r=self.H_t_to_r,
            rgb_pts=np.array(self.rgb_pts, dtype=np.float32),
            thermal_pts=np.array(self.thermal_pts, dtype=np.float32),
        )
        print(f"[calib] 저장: {CALIB_PATH}")

    def _reset_points(self) -> None:
        self.rgb_pts.clear()
        self.thermal_pts.clear()
        self.expecting = "rgb"

    def _compute_h(self) -> None:
        if len(self.rgb_pts) < 4 or len(self.thermal_pts) < 4:
            return
        src = np.array(self.thermal_pts, dtype=np.float32)
        dst = np.array(self.rgb_pts, dtype=np.float32)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            print("[calib] findHomography 실패 — 점이 일직선이거나 너무 가까울 수 있음")
            return
        self.H_t_to_r = H
        self._save_calibration()
        print(f"[calib] H 계산 완료 ({len(self.rgb_pts)}쌍)")

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or not self.calibrating:
            return
        if x < PANEL_W:  # 좌측 RGB 패널
            if self.expecting != "rgb":
                return
            ox, oy, scale = self._rgb_meta
            orig = ((x - ox) / scale, (y - oy) / scale)
            self.rgb_pts.append(orig)
            self.expecting = "thermal"
        else:             # 우측 Thermal 패널
            if self.expecting != "thermal":
                return
            px, py = x - PANEL_W, y
            ox, oy, scale = self._th_meta
            orig = ((px - ox) / scale, (py - oy) / scale)
            self.thermal_pts.append(orig)
            self.expecting = "rgb"
            self._compute_h()

    def compose(self, rgb_frame: np.ndarray, thermal_frame: np.ndarray) -> np.ndarray:
        if self.overlay and self.H_t_to_r is not None:
            warped = cv2.warpPerspective(
                thermal_frame, self.H_t_to_r,
                (rgb_frame.shape[1], rgb_frame.shape[0]),
            )
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            heat = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
            mask = (gray > 0).astype(np.uint8) * 255
            mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            blended = np.where(mask3 > 0,
                               cv2.addWeighted(rgb_frame, 0.55, heat, 0.45, 0),
                               rgb_frame)
            rgb_panel, rmeta = fit_to(blended, PANEL_W, PANEL_H)
        else:
            rgb_panel, rmeta = fit_to(rgb_frame, PANEL_W, PANEL_H)

        th_panel, tmeta = fit_to(thermal_frame, PANEL_W, PANEL_H)
        self._rgb_meta, self._th_meta = rmeta, tmeta

        for i, p in enumerate(self.rgb_pts):
            ox, oy, s = rmeta
            cx, cy = int(p[0] * s + ox), int(p[1] * s + oy)
            cv2.circle(rgb_panel, (cx, cy), 6, (0, 255, 0), 2)
            draw_text(rgb_panel, str(i + 1), (cx + 8, cy - 8), (0, 255, 0))
        for i, p in enumerate(self.thermal_pts):
            ox, oy, s = tmeta
            cx, cy = int(p[0] * s + ox), int(p[1] * s + oy)
            cv2.circle(th_panel, (cx, cy), 6, (0, 255, 0), 2)
            draw_text(th_panel, str(i + 1), (cx + 8, cy - 8), (0, 255, 0))

        combined = np.hstack([rgb_panel, th_panel])
        cv2.line(combined, (PANEL_W, 0), (PANEL_W, PANEL_H), (100, 100, 100), 1)

        status = [
            f"mode={'CALIB' if self.calibrating else 'VIEW'}",
            f"pts={len(self.rgb_pts)}/{len(self.thermal_pts)}",
        ]
        if self.H_t_to_r is not None:
            status.append("H:OK")
        if self.overlay:
            status.append("OVERLAY")
        if self.colormap_count > 0:
            status.append(f"cmap={self.colormap_idx}/{self.colormap_count - 1}")
        draw_text(combined, " | ".join(status), (10, 22), (255, 255, 0), 0.6)
        draw_text(combined, "[q]uit [s]nap [c]alib [o]verlay [r]eset  [/] colormap",
                  (10, PANEL_H - 12), (200, 200, 200), 0.5)
        draw_text(combined, "RGB", (PANEL_W - 60, 22), (180, 220, 255), 0.6)
        # THERMAL 라벨 옆에 현재 컬러맵 인덱스 큼지막하게 표시
        thermal_label = "THERMAL"
        if self.colormap_count > 0:
            thermal_label += f"  cmap={self.colormap_idx:2d}"
        draw_text(combined, thermal_label, (PANEL_W + 10, 22), (255, 180, 180), 0.7)
        if self.colormap_count > 0:
            # 우측 패널 좌상단에 큰 폰트로 인덱스 강조
            draw_text(combined, f"#{self.colormap_idx:02d}",
                      (PANEL_W + 10, 60), (0, 255, 255), 1.2)

        if self.calibrating:
            hint = f"다음 클릭: {self.expecting.upper()}"
            draw_text(combined, hint, (10, 48), (0, 255, 255), 0.6)

        return combined

    def run(self) -> None:
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        win = "HeatSight - RGB | Thermal"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, self.on_mouse)

        self._open_cellplus()
        with RGBSource() as rgb, ThermalSource() as thermal:
            print(f"RGB     name='{rgb.resolved_name}'")
            print(f"Thermal name='{thermal.resolved_name}'")
            print("창에 포커스가 있어야 키 입력이 받아집니다.")
            time.sleep(1.0)
            try:
                _verify_sources(rgb.read(timeout=3.0).image,
                                thermal.read(timeout=3.0).image)
            except TimeoutError as e:
                print(f"[verify] 워밍업 실패: {e}")

            while True:
                try:
                    rf = rgb.read(timeout=0.5).image
                    tf = thermal.read(timeout=0.5).image
                except TimeoutError as e:
                    print(f"[warn] {e}")
                    continue

                frame = self.compose(rf, tf)
                cv2.imshow(win, frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                elif k == ord("s"):
                    path = SNAPSHOT_DIR / f"snap_{int(time.time())}.png"
                    cv2.imwrite(str(path), frame)
                    print(f"[snap] {path}")
                elif k == ord("c"):
                    self.calibrating = not self.calibrating
                    if self.calibrating:
                        self._reset_points()
                    print(f"[mode] calibrating={self.calibrating}")
                elif k == ord("o"):
                    if self.H_t_to_r is None:
                        print("[overlay] 캘리브레이션 먼저 필요")
                    else:
                        self.overlay = not self.overlay
                        print(f"[mode] overlay={self.overlay}")
                elif k == ord("r"):
                    self._reset_points()
                    print("[calib] 점 초기화")
                elif k == ord("]"):
                    self._bump_colormap(+1)
                elif k == ord("["):
                    self._bump_colormap(-1)

        if self.cellplus is not None:
            self.cellplus.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    DualViewer().run()
