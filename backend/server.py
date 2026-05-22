"""HeatSight 백엔드 — FastAPI.

엔드포인트:
  GET  /api/status              상태 (카메라/시리얼 연결 여부, 컬러맵 정보)
  GET  /api/video/rgb           MJPEG 스트림 (FaceTime HD)
  GET  /api/video/thermal       MJPEG 스트림 (Cellplus, 카메라 자체 컬러맵 적용 상태)
  GET  /api/colormap            현재 컬러맵 + 사용 가능한 프리셋
  POST /api/colormap  {index}   컬러맵 변경

설계 메모:
  - 카메라(UVC)는 한 클라이언트만 잡을 수 있으므로 서버가 한 번만 잡고 frame을 캐시한다.
  - 여러 브라우저 클라이언트가 같은 frame을 동시에 받는다.
  - dual_viewer.py와 서버를 동시에 실행하면 카메라 충돌이 난다 — 둘 중 하나만.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import AsyncIterator, Optional

# 상위 디렉토리(capture.py, thermal_control.py) import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from capture import RGBSource, ThermalSource
from thermal_control import (
    COLORMAP_PRESETS,
    CellplusControl,
    find_control_port,
)
from face_analyzer import FaceAnalyzer
from pose_analyzer import PoseAnalyzer
from calibration import Calibration


app = FastAPI(title="HeatSight Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용. 배포 시 도메인 명시
    allow_methods=["*"],
    allow_headers=["*"],
)


class CameraHub:
    """카메라와 시리얼 컨트롤을 단일 인스턴스로 보유."""

    def __init__(self) -> None:
        self.rgb: Optional[RGBSource] = None
        self.thermal: Optional[ThermalSource] = None
        self.cellplus: Optional[CellplusControl] = None
        self.face: Optional[FaceAnalyzer] = None
        self.pose: Optional[PoseAnalyzer] = None
        self.colormap_idx: int = COLORMAP_PRESETS[0]

    def start(self) -> None:
        self.rgb = RGBSource().start()
        self.thermal = ThermalSource().start()

        try:
            self.face = FaceAnalyzer(self.rgb, fps=15.0).start()
            print("[hub] FaceAnalyzer 시작")
        except Exception as e:
            print(f"[hub] FaceAnalyzer 시작 실패: {e}")

        try:
            self.pose = PoseAnalyzer(self.rgb, fps=10.0).start()
            print("[hub] PoseAnalyzer 시작")
        except Exception as e:
            print(f"[hub] PoseAnalyzer 시작 실패: {e}")

        port = find_control_port()
        if port is None:
            print("[hub] Cellplus 컨트롤 포트를 찾지 못함 — 컬러맵 컨트롤 비활성")
            return
        try:
            ctl = CellplusControl(port).open()
            current = ctl.get_colormap()
            if current not in COLORMAP_PRESETS:
                current = COLORMAP_PRESETS[0]
                ctl.set_colormap(current)
            self.colormap_idx = current
            self.cellplus = ctl
            print(f"[hub] Cellplus {port}, colormap={current}, presets={COLORMAP_PRESETS}")
        except Exception as e:
            print(f"[hub] 시리얼 컨트롤 초기화 실패: {e}")

    def stop(self) -> None:
        if self.pose is not None:
            self.pose.stop()
        if self.face is not None:
            self.face.stop()
        if self.rgb is not None:
            self.rgb.stop()
        if self.thermal is not None:
            self.thermal.stop()
        if self.cellplus is not None:
            self.cellplus.close()

    def set_colormap(self, index: int) -> None:
        if index not in COLORMAP_PRESETS:
            raise ValueError(f"index {index}는 프리셋이 아님: {COLORMAP_PRESETS}")
        if self.cellplus is None:
            raise RuntimeError("Cellplus 시리얼 컨트롤이 활성화되지 않음")
        self.cellplus.set_colormap(index)
        self.colormap_idx = index

    def measure_pixel(self, x: int, y: int) -> int:
        if self.cellplus is None:
            raise RuntimeError("Cellplus 시리얼 컨트롤이 활성화되지 않음")
        return self.cellplus.measure_pixel_raw(x, y)

    def read_minmax(self) -> dict:
        if self.cellplus is None:
            raise RuntimeError("Cellplus 시리얼 컨트롤이 활성화되지 않음")
        return self.cellplus.get_minmax()


hub = CameraHub()
calibration = Calibration()


@app.on_event("startup")
def on_startup() -> None:
    hub.start()
    time.sleep(1.2)  # 카메라 워밍업


@app.on_event("shutdown")
def on_shutdown() -> None:
    hub.stop()


def _mjpeg_iterator(source_name: str) -> AsyncIterator[bytes]:
    """무한 MJPEG 스트림 생성기. 클라이언트 disconnect 시 자연 종료."""
    boundary = b"--frame"
    last_ts = -1.0

    def encode(frame: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else b""

    while True:
        src = hub.rgb if source_name == "rgb" else hub.thermal
        if src is None:
            yield boundary + b"\r\n"
            time.sleep(0.5)
            continue
        try:
            frame = src.read(timeout=1.0)
        except Exception:
            time.sleep(0.05)
            continue
        if frame.timestamp == last_ts:
            time.sleep(0.005)
            continue
        last_ts = frame.timestamp

        jpeg = encode(frame.image)
        if not jpeg:
            continue
        chunk = (
            boundary + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
            + jpeg + b"\r\n"
        )
        yield chunk


@app.get("/api/status")
def get_status() -> dict:
    return {
        "rgb_connected": hub.rgb is not None,
        "thermal_connected": hub.thermal is not None,
        "cellplus_connected": hub.cellplus is not None,
        "face_connected": hub.face is not None,
        "pose_connected": hub.pose is not None,
        "colormap_idx": hub.colormap_idx,
        "colormap_presets": COLORMAP_PRESETS,
    }


@app.get("/api/face")
def get_face() -> dict:
    if hub.face is None:
        raise HTTPException(status_code=503, detail="FaceAnalyzer 비활성")
    return hub.face.latest()


@app.get("/api/pose")
def get_pose() -> dict:
    if hub.pose is None:
        raise HTTPException(status_code=503, detail="PoseAnalyzer 비활성")
    return hub.pose.latest()


# --- 캘리브레이션 ---

@app.get("/api/calibration")
def get_calibration() -> dict:
    return calibration.status()


THERMAL_W, THERMAL_H = 640, 480
WRIST_VISIBILITY_TH = 0.7  # MediaPipe가 화면 밖에서도 visible=True를 주는 경우 방지


def _pick_wrist_strict(pose_latest: dict) -> Optional[dict]:
    """visibility가 충분히 높은 wrist만 반환. 없으면 None."""
    poses = pose_latest.get("poses", [])
    if not poses:
        return None
    p = poses[0]
    # 백엔드 pose_analyzer는 visible(bool)만 반환. 강화하려면 raw visibility를 받아야 하나
    # 우선 visible=True인 wrist 중 화면 안에 들어있는지 추가 체크.
    for key in ("wrist_r", "wrist_l"):
        w = p.get(key)
        if not (w and w.get("visible")):
            continue
        x, y = w["x"], w["y"]
        if 0 <= x < THERMAL_W and 0 <= y < THERMAL_H:
            return {"x": x, "y": y, "side": key}
    return None


def _find_thermal_hotspot_near(init_x: int, init_y: int, search: int = 120) -> tuple[int, int]:
    """thermal frame의 (init_x, init_y) 주변 search px 박스에서 가장 밝은 픽셀.

    영상의 brightness ≒ raw temperature (우리 프리셋들이 모두 hot=밝음 매핑이므로).
    이 방법은 thermal 전체 max(보통 얼굴/몸)에 휘둘리지 않고 손 근방만 본다.
    """
    if hub.thermal is None:
        raise RuntimeError("ThermalSource 비활성")
    frame = hub.thermal.read(timeout=1.0).image  # BGR uint8
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x0 = max(0, init_x - search)
    y0 = max(0, init_y - search)
    x1 = min(THERMAL_W, init_x + search)
    y1 = min(THERMAL_H, init_y + search)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        raise RuntimeError("검색 영역이 비어있음")
    my, mx = np.unravel_index(int(np.argmax(roi)), roi.shape)
    return x0 + int(mx), y0 + int(my)


@app.post("/api/calibration/capture")
def calibration_capture() -> dict:
    """현재 RGB wrist + 손 근방 thermal hotspot을 한 페어로 저장."""
    if hub.pose is None:
        raise HTTPException(status_code=503, detail="PoseAnalyzer 비활성")
    if hub.thermal is None:
        raise HTTPException(status_code=503, detail="ThermalSource 비활성")

    wrist = _pick_wrist_strict(hub.pose.latest())
    if wrist is None:
        raise HTTPException(
            status_code=400,
            detail="손목이 화면 안에서 또렷이 검출되지 않음 — 손을 카메라 시야 중앙쪽으로",
        )

    # RGB wrist의 thermal 좌표 1차 근사 = 좌우 반전 (이미 검증한 변환)
    init_x = THERMAL_W - 1 - wrist["x"]
    init_y = wrist["y"]
    try:
        tx, ty = _find_thermal_hotspot_near(init_x, init_y, search=120)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    calibration.add_pair(
        rgb={"x": wrist["x"], "y": wrist["y"]},
        thermal={"x": tx, "y": ty},
    )
    return {
        "captured": {
            "rgb": wrist,
            "thermal": {"x": tx, "y": ty, "search_center": [init_x, init_y]},
        },
        **calibration.status(),
    }


@app.post("/api/calibration/reset")
def calibration_reset() -> dict:
    calibration.reset()
    return calibration.status()


class PairBody(BaseModel):
    rgb_x: int
    rgb_y: int
    thermal_x: int
    thermal_y: int


@app.post("/api/calibration/pair")
def calibration_pair(body: PairBody) -> dict:
    """사용자가 양쪽 화면에서 직접 클릭한 좌표 쌍을 페어로 저장."""
    calibration.add_pair(
        rgb={"x": int(body.rgb_x), "y": int(body.rgb_y)},
        thermal={"x": int(body.thermal_x), "y": int(body.thermal_y)},
    )
    return calibration.status()


@app.get("/api/colormap")
def get_colormap() -> dict:
    return {
        "current": hub.colormap_idx,
        "presets": COLORMAP_PRESETS,
        "controllable": hub.cellplus is not None,
    }


class ColormapBody(BaseModel):
    index: int


@app.post("/api/colormap")
def post_colormap(body: ColormapBody) -> dict:
    try:
        hub.set_colormap(body.index)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"current": hub.colormap_idx}


class PixelBody(BaseModel):
    x: int
    y: int


@app.post("/api/temperature/pixel")
def temperature_pixel(body: PixelBody) -> dict:
    try:
        raw = hub.measure_pixel(body.x, body.y)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"x": body.x, "y": body.y, "raw": raw}


@app.get("/api/temperature/minmax")
def temperature_minmax() -> dict:
    try:
        return hub.read_minmax()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/video/rgb")
def video_rgb() -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_iterator("rgb"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/video/thermal")
def video_thermal() -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_iterator("thermal"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
