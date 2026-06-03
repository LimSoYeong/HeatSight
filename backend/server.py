"""HeatSight 백엔드 — FastAPI.

엔드포인트:
  GET  /api/status              상태 (카메라/시리얼 연결 여부, 컬러맵 정보)
  GET  /api/hvac/recommendation 재실자 체온 기반 냉난방 제어 추천
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
import threading
import time
from pathlib import Path
from typing import AsyncIterator, Optional

# 상위 디렉토리(capture.py, thermal_control.py) import
BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(BACKEND_DIR))

import cv2
import httpx
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
from pose_analyzer import PoseAnalyzer  # legacy (사용 안 함, 호환용)
from remote_pose_analyzer import RemotePoseAnalyzer
from behavior_analyzer import BehaviorAnalyzer
from calibration import Calibration

GPU_SERVICE_URL = "http://101.79.21.220:9000"


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
        self.face_thermal: Optional[FaceAnalyzer] = None
        self.pose: Optional[RemotePoseAnalyzer] = None
        self.behavior: Optional[BehaviorAnalyzer] = None
        self.colormap_idx: int = COLORMAP_PRESETS[0]
        self.serial_lock = threading.RLock()

    def start(self) -> None:
        self.rgb = RGBSource().start()
        self.thermal = ThermalSource().start()

        try:
            self.face = FaceAnalyzer(self.rgb, fps=15.0).start()
            print("[hub] FaceAnalyzer (RGB) 시작")
        except Exception as e:
            print(f"[hub] FaceAnalyzer (RGB) 시작 실패: {e}")

        # Thermal source용 별도 인스턴스. colormap 적용된 BGR이라 MediaPipe RGB
        # 모델로는 confidence가 낮으니 threshold를 내려서 시도.
        try:
            self.face_thermal = FaceAnalyzer(
                self.thermal,
                fps=10.0,
                detection_confidence=0.2,
                presence_confidence=0.2,
                tracking_confidence=0.2,
            ).start()
            print("[hub] FaceAnalyzer (Thermal) 시작 (low threshold)")
        except Exception as e:
            print(f"[hub] FaceAnalyzer (Thermal) 시작 실패: {e}")

        try:
            self.pose = RemotePoseAnalyzer(
                self.rgb,
                url=f"{GPU_SERVICE_URL}/pose",
                fps=8.0,
            ).start()
            print(f"[hub] RemotePoseAnalyzer 시작 → {GPU_SERVICE_URL}/pose")
        except Exception as e:
            print(f"[hub] RemotePoseAnalyzer 시작 실패: {e}")

        try:
            self.behavior = BehaviorAnalyzer(self, gpu_url=GPU_SERVICE_URL).start()
            print(f"[hub] BehaviorAnalyzer 시작 (heuristic + CLIP + VLM → {GPU_SERVICE_URL})")
        except Exception as e:
            print(f"[hub] BehaviorAnalyzer 시작 실패: {e}")

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
        if self.behavior is not None:
            self.behavior.stop()
        if self.pose is not None:
            self.pose.stop()
        if self.face is not None:
            self.face.stop()
        if self.face_thermal is not None:
            self.face_thermal.stop()
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
        with self.serial_lock:
            self.cellplus.set_colormap(index)
            self.colormap_idx = index

    def measure_pixel(self, x: int, y: int) -> int:
        if self.cellplus is None:
            raise RuntimeError("Cellplus 시리얼 컨트롤이 활성화되지 않음")
        with self.serial_lock:
            return self.cellplus.measure_pixel_raw(x, y)

    def read_minmax(self) -> dict:
        if self.cellplus is None:
            raise RuntimeError("Cellplus 시리얼 컨트롤이 활성화되지 않음")
        with self.serial_lock:
            return self.cellplus.get_minmax()


hub = CameraHub()
calibration = Calibration()
gpu_http = httpx.Client(timeout=10.0)


def _gpu_register_params() -> bool:
    """startup 시 1회: 카메라에서 정적 radiometric 5 params 읽어 GPU에 등록."""
    if hub.cellplus is None:
        return False
    try:
        with hub.serial_lock:
            params = hub.cellplus.get_radiometric_static_params()
        r = gpu_http.post(f"{GPU_SERVICE_URL}/params", json={"params": params})
        if r.status_code == 200:
            print(f"[hub] GPU /params 등록 OK: {params}")
            return True
        print(f"[hub] GPU /params 실패: HTTP {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"[hub] GPU /params 예외: {e}")
    return False


def _gpu_convert_celsius(raw: int, dyn: dict) -> Optional[float]:
    try:
        r = gpu_http.post(
            f"{GPU_SERVICE_URL}/convert",
            json={
                "raw": int(raw),
                "board_temp_c": dyn["board_temp_c"],
                "gain": dyn["gain"],
                "pwm_target": dyn["pwm_target"],
            },
        )
        if r.status_code == 200:
            return float(r.json()["celsius"])
    except Exception:
        pass
    return None


SKIN_TARGET_C = 34.0
SKIN_NEUTRAL_BAND_C = 1.8


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _measure_face_temperature_results(grid: int = 4) -> dict:
    """검출된 face bbox를 thermal 좌표로 매핑해 평균 피부 온도를 측정."""
    if hub.face is None:
        raise HTTPException(status_code=503, detail="FaceAnalyzer 비활성")
    if hub.cellplus is None:
        raise HTTPException(status_code=503, detail="Cellplus 시리얼 컨트롤 비활성")
    if calibration.H is None:
        raise HTTPException(status_code=400, detail="캘리브레이션 필요 (4쌍 이상)")

    face_latest = hub.face.latest()
    faces = face_latest.get("faces", [])

    results: list[dict] = []
    with hub.serial_lock:
        try:
            dyn = hub.cellplus.get_temperature_dynamic_params()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"dynamic params 읽기 실패: {e}")

        for idx, f in enumerate(faces):
            rgb_bbox = f["bbox"]
            mapped = calibration.map_bbox(
                rgb_bbox["x"], rgb_bbox["y"], rgb_bbox["w"], rgb_bbox["h"]
            )
            if mapped is None:
                continue
            tbb = mapped["bbox"]
            try:
                mean_raw, samples = hub.cellplus.measure_bbox_mean_raw(
                    tbb["x"], tbb["y"], tbb["w"], tbb["h"], grid=grid
                )
            except ValueError:
                continue
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"thermal 측정 실패: {e}")

            mean_celsius = _gpu_convert_celsius(mean_raw, dyn)
            results.append({
                "index": idx,
                "rgb_bbox": rgb_bbox,
                "thermal_bbox": tbb,
                "thermal_corners": mapped["corners"],
                "mean_raw": mean_raw,
                "mean_celsius": mean_celsius,
                "sample_count": len(samples),
            })

    return {
        "timestamp": face_latest.get("timestamp", 0.0),
        "grid": grid,
        "faces": results,
    }


def _behavior_score(behavior: Optional[dict]) -> float:
    """행동 신호를 -1(춥다) ~ +1(덥다) 스코어로 변환."""
    if not behavior:
        return 0.0
    fused = behavior.get("fused_comfort") or behavior.get("comfort")
    hot_motion = max(
        float(behavior.get("hands_up", 0.0)),
        float(behavior.get("fanning", 0.0)),
    )
    cold_motion = max(
        float(behavior.get("arms_crossed", 0.0)),
        float(behavior.get("hunched", 0.0)),
    )
    raw_score = _clamp(hot_motion - cold_motion, -1.0, 1.0)
    if fused == "hot":
        return max(raw_score, 0.55)
    if fused == "cold":
        return min(raw_score, -0.55)
    return raw_score * 0.2


def _comfort_from_score(score: float) -> str:
    if score >= 0.25:
        return "hot"
    if score <= -0.25:
        return "cold"
    return "neutral"


def _skin_score(skin_celsius: Optional[float]) -> Optional[float]:
    if skin_celsius is None:
        return None
    return _clamp((skin_celsius - SKIN_TARGET_C) / SKIN_NEUTRAL_BAND_C, -1.0, 1.0)


def _occupant_details(face_latest: dict, face_temps: list[dict]) -> list[dict]:
    temp_by_index = {int(f["index"]): f for f in face_temps}
    details: list[dict] = []
    for idx, face in enumerate(face_latest.get("faces", [])):
        temp = temp_by_index.get(idx)
        skin_c = (
            float(temp["mean_celsius"])
            if temp is not None and temp.get("mean_celsius") is not None
            else None
        )
        score = _skin_score(skin_c)
        details.append({
            "id": f"person-{idx + 1}",
            "index": idx,
            "label": f"Person {idx + 1}",
            "comfort": _comfort_from_score(score) if score is not None else "unknown",
            "comfort_score": round(score, 3) if score is not None else None,
            "skin_temperature_c": skin_c,
            "rgb_bbox": face.get("bbox"),
            "thermal_bbox": temp.get("thermal_bbox") if temp is not None else None,
            "sample_count": temp.get("sample_count", 0) if temp is not None else 0,
        })
    return details


def _control_from_score(score: float, occupants: int) -> dict:
    if occupants <= 0:
        return {
            "mode": "standby",
            "mode_label": "재실 없음",
            "target_setpoint_c": None,
            "target_delta_c": 0.0,
            "fan_percent": 20,
            "reason": "재실자가 검출되지 않아 공조 출력을 낮춥니다.",
        }
    if score >= 0.65:
        return {
            "mode": "cooling",
            "mode_label": "강한 냉방",
            "target_setpoint_c": 24.5,
            "target_delta_c": -1.5,
            "fan_percent": 75,
            "reason": "피부 온도 또는 행동 신호가 더운 상태로 강하게 나타납니다.",
        }
    if score >= 0.25:
        return {
            "mode": "cooling",
            "mode_label": "완만한 냉방",
            "target_setpoint_c": 25.5,
            "target_delta_c": -0.8,
            "fan_percent": 55,
            "reason": "재실자가 덥다고 판단되어 최소 냉방 보정을 적용합니다.",
        }
    if score <= -0.65:
        return {
            "mode": "heating",
            "mode_label": "강한 난방",
            "target_setpoint_c": 22.5,
            "target_delta_c": 1.5,
            "fan_percent": 75,
            "reason": "피부 온도 또는 행동 신호가 추운 상태로 강하게 나타납니다.",
        }
    if score <= -0.25:
        return {
            "mode": "heating",
            "mode_label": "완만한 난방",
            "target_setpoint_c": 21.5,
            "target_delta_c": 0.8,
            "fan_percent": 55,
            "reason": "재실자가 춥다고 판단되어 최소 난방 보정을 적용합니다.",
        }
    return {
        "mode": "eco",
        "mode_label": "쾌적 유지",
        "target_setpoint_c": None,
        "target_delta_c": 0.0,
        "fan_percent": 30,
        "reason": "현재 피부 온도와 행동 신호가 중립 범위입니다.",
    }


def _energy_saving_estimate(control: dict, thermal_available: bool) -> dict:
    mode = control["mode"]
    if mode == "standby":
        saving = 28
    elif mode == "eco":
        saving = 18
    elif abs(float(control["target_delta_c"])) < 1.0:
        saving = 12
    else:
        saving = 7
    if thermal_available:
        saving += 3
    return {
        "estimated_saving_percent": min(32, saving),
        "strategy": "재실자 체온 기반 최소 보정",
    }


@app.on_event("startup")
def on_startup() -> None:
    hub.start()
    time.sleep(1.2)  # 카메라 워밍업
    _gpu_register_params()


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
        "face_thermal_connected": hub.face_thermal is not None,
        "pose_connected": hub.pose is not None,
        "colormap_idx": hub.colormap_idx,
        "colormap_presets": COLORMAP_PRESETS,
    }


@app.get("/api/face")
def get_face() -> dict:
    if hub.face is None:
        raise HTTPException(status_code=503, detail="FaceAnalyzer 비활성")
    return hub.face.latest()


@app.get("/api/face/thermal")
def get_face_thermal() -> dict:
    if hub.face_thermal is None:
        raise HTTPException(status_code=503, detail="Thermal FaceAnalyzer 비활성")
    return hub.face_thermal.latest()


@app.get("/api/pose")
def get_pose() -> dict:
    if hub.pose is None:
        raise HTTPException(status_code=503, detail="PoseAnalyzer 비활성")
    return hub.pose.latest()


@app.get("/api/behavior")
def get_behavior() -> dict:
    if hub.behavior is None:
        raise HTTPException(status_code=503, detail="BehaviorAnalyzer 비활성")
    return hub.behavior.latest()


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
    celsius = None
    if hub.cellplus is not None:
        try:
            with hub.serial_lock:
                dyn = hub.cellplus.get_temperature_dynamic_params()
            celsius = _gpu_convert_celsius(raw, dyn)
        except Exception:
            celsius = None
    return {"x": body.x, "y": body.y, "raw": raw, "celsius": celsius}


@app.get("/api/temperature/faces")
def temperature_faces(grid: int = 4) -> dict:
    """각 검출된 face bbox의 thermal 평균 °C를 측정해 반환.

    - face bbox (RGB) → homography로 thermal 좌표계 변환
    - thermal bbox 내 grid×grid 균일 샘플의 raw 평균 → °C 변환
    - 캘리브레이션이 없으면 400, cellplus 없으면 503.
    """
    return _measure_face_temperature_results(grid=grid)


@app.get("/api/hvac/recommendation")
def hvac_recommendation() -> dict:
    """피부 온도 + 행동 쾌적도 기반 냉난방 제어 추천."""
    face_latest = hub.face.latest() if hub.face is not None else {"faces": []}
    pose_latest = hub.pose.latest() if hub.pose is not None else {"poses": []}
    behavior_latest = hub.behavior.latest() if hub.behavior is not None else None

    face_count = len(face_latest.get("faces", []))
    pose_count = len(pose_latest.get("poses", []))
    occupants = face_count if face_count > 0 else min(pose_count, 1)

    face_temps: list[dict] = []
    thermal_error = None
    if face_count > 0 and hub.cellplus is not None and calibration.H is not None:
        try:
            # 제어용은 빠른 반응이 우선이므로 2x2 샘플로 계산한다.
            face_temps = _measure_face_temperature_results(grid=2).get("faces", [])
        except HTTPException as e:
            thermal_error = str(e.detail)

    skin_values = [
        float(f["mean_celsius"])
        for f in face_temps
        if f.get("mean_celsius") is not None
    ]
    occupant_details = _occupant_details(face_latest, face_temps)
    if not occupant_details and occupants > 0:
        occupant_details = [{
            "id": "person-1",
            "index": 0,
            "label": "Person 1",
            "comfort": "unknown",
            "comfort_score": None,
            "skin_temperature_c": None,
            "rgb_bbox": None,
            "thermal_bbox": None,
            "sample_count": 0,
        }]
    skin_avg = sum(skin_values) / len(skin_values) if skin_values else None
    thermal_score = _skin_score(skin_avg)
    behavior_score = _behavior_score(behavior_latest)

    if occupants <= 0:
        comfort_score = 0.0
    elif thermal_score is not None:
        comfort_score = 0.7 * thermal_score + 0.3 * behavior_score
    else:
        comfort_score = behavior_score
    comfort_score = round(_clamp(comfort_score, -1.0, 1.0), 3)

    comfort = _comfort_from_score(comfort_score)
    control = _control_from_score(comfort_score, occupants)
    energy = _energy_saving_estimate(control, thermal_score is not None)

    if thermal_score is not None and behavior_latest is not None:
        data_quality = "thermal+behavior"
    elif thermal_score is not None:
        data_quality = "thermal"
    elif behavior_latest is not None:
        data_quality = "behavior"
    else:
        data_quality = "camera"

    return {
        "timestamp": time.time(),
        "occupants": occupants,
        "occupant_details": occupant_details,
        "comfort": comfort,
        "comfort_score": comfort_score,
        "skin_temperature_c": skin_avg,
        "skin_temperatures_c": skin_values,
        "control": control,
        "energy": energy,
        "signals": {
            "thermal_score": thermal_score,
            "behavior_score": round(behavior_score, 3),
            "behavior_comfort": (
                behavior_latest.get("fused_comfort") or behavior_latest.get("comfort")
                if behavior_latest
                else None
            ),
            "calibrated": calibration.H is not None,
            "thermal_error": thermal_error,
            "data_quality": data_quality,
        },
        "camera": {
            "rgb_connected": hub.rgb is not None,
            "thermal_connected": hub.thermal is not None,
            "cellplus_connected": hub.cellplus is not None,
        },
    }


@app.get("/api/temperature/minmax")
def temperature_minmax() -> dict:
    try:
        data = hub.read_minmax()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    if hub.cellplus is not None:
        try:
            with hub.serial_lock:
                dyn = hub.cellplus.get_temperature_dynamic_params()
            data["min_celsius"] = _gpu_convert_celsius(data["min_raw"], dyn)
            data["max_celsius"] = _gpu_convert_celsius(data["max_raw"], dyn)
        except Exception:
            pass
    return data


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
