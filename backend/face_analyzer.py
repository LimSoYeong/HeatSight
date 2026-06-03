"""RGB 영상 얼굴 분석 — MediaPipe Tasks API (FaceLandmarker).

mediapipe 0.10.x부터 legacy Solutions API가 제거돼 Tasks API만 남았다.
모델 파일(`backend/models/face_landmarker.task`)을 미리 다운로드해 두어야 한다.

백그라운드 스레드가 RGBSource에서 frame을 받아 추론.
결과(얼굴 bbox + 핵심 영역)를 캐시. 메인 스레드는 latest()로 즉시 가져감.

핵심 영역은 향후 thermal 매핑에서 평균 온도 측정에 쓰일 ROI 후보:
  - forehead_box: 이마 중앙 사각형
  - cheek_left_box / cheek_right_box: 양 뺨 사각형
  - nose_tip / forehead_center 등 점
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"


# MediaPipe Face Mesh 468 keypoint 중 핵심 인덱스
LM_NOSE_TIP    = 1
LM_FOREHEAD    = 10        # 이마 중앙
LM_CHEEK_LEFT  = 50        # 화면상 좌측 뺨
LM_CHEEK_RIGHT = 280       # 화면상 우측 뺨
LM_LEFT_BROW   = 105
LM_RIGHT_BROW  = 334
LM_LEFT_EYE_OUTER  = 33    # 화면상 좌측 눈 바깥
LM_RIGHT_EYE_OUTER = 263   # 화면상 우측 눈 바깥


@dataclass
class Point:
    x: int
    y: int


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int


@dataclass
class FaceRegions:
    bbox: BBox
    nose_tip: Point
    forehead_center: Point
    cheek_left: Point
    cheek_right: Point
    forehead_box: BBox
    cheek_left_box: BBox
    cheek_right_box: BBox
    # 얼굴 roll에 맞춰 회전된 외접 사각형의 4 corner (좌상→우상→우하→좌하)
    quad: List[Point] = field(default_factory=list)
    forehead_quad: List[Point] = field(default_factory=list)
    cheek_left_quad: List[Point] = field(default_factory=list)
    cheek_right_quad: List[Point] = field(default_factory=list)
    roll_deg: float = 0.0


@dataclass
class FaceResult:
    timestamp: float
    faces: List[FaceRegions]


class FaceAnalyzer:
    def __init__(
        self,
        rgb_source,
        fps: float = 15.0,
        detection_confidence: float = 0.5,
        presence_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
    ) -> None:
        self.rgb_source = rgb_source
        self.interval = 1.0 / max(fps, 1.0)
        self.detection_confidence = detection_confidence
        self.presence_confidence = presence_confidence
        self.tracking_confidence = tracking_confidence
        self._latest: Optional[FaceResult] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._landmarker: Optional[mp_vision.FaceLandmarker] = None

    def start(self) -> "FaceAnalyzer":
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"face_landmarker 모델이 없음: {MODEL_PATH}")
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=4,
            min_face_detection_confidence=self.detection_confidence,
            min_face_presence_confidence=self.presence_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._landmarker is not None:
            self._landmarker.close()

    def latest(self) -> dict:
        with self._lock:
            if self._latest is None:
                return {"timestamp": 0.0, "faces": []}
            return {
                "timestamp": self._latest.timestamp,
                "faces": [asdict(f) for f in self._latest.faces],
            }

    def _run(self) -> None:
        last_seen = -1.0
        while not self._stop.is_set():
            try:
                frame = self.rgb_source.read(timeout=0.5)
            except Exception:
                time.sleep(0.05)
                continue
            if frame.timestamp == last_seen:
                time.sleep(self.interval / 2)
                continue
            last_seen = frame.timestamp

            # MediaPipe는 RGB 기대 (우리 frame은 BGR uint8)
            image_rgb = np.ascontiguousarray(frame.image[:, :, ::-1])
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            ts_ms = int(frame.timestamp * 1000)
            assert self._landmarker is not None
            try:
                result = self._landmarker.detect_for_video(mp_image, ts_ms)
            except Exception:
                time.sleep(self.interval)
                continue
            faces = self._extract_faces(result, frame.image.shape[:2])

            with self._lock:
                self._latest = FaceResult(timestamp=frame.timestamp, faces=faces)

            time.sleep(self.interval)

    @staticmethod
    def _extract_faces(result, shape) -> List[FaceRegions]:
        h, w = shape
        out: List[FaceRegions] = []
        face_landmarks_list = getattr(result, "face_landmarks", None) or []
        if not face_landmarks_list:
            return out

        for landmarks in face_landmarks_list:
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            x0 = int(max(0.0, min(xs)) * w)
            y0 = int(max(0.0, min(ys)) * h)
            x1 = int(min(1.0, max(xs)) * w)
            y1 = int(min(1.0, max(ys)) * h)
            bbox = BBox(x0, y0, max(1, x1 - x0), max(1, y1 - y0))

            # roll(고개 기울임)에 맞춘 회전된 외접 사각형 4 corner.
            # 두 눈 바깥 점으로 roll 각도를 측정 → landmarks를 역회전 후
            # AABB → 다시 정회전. minAreaRect보다 frame 간 코너 순서가 안정적.
            pts = np.empty((len(landmarks), 2), dtype=np.float32)
            for i, lm in enumerate(landmarks):
                pts[i, 0] = lm.x * w
                pts[i, 1] = lm.y * h
            le = landmarks[LM_LEFT_EYE_OUTER]
            re = landmarks[LM_RIGHT_EYE_OUTER]
            roll = math.atan2((re.y - le.y) * h, (re.x - le.x) * w)
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            cos_a = math.cos(-roll)
            sin_a = math.sin(-roll)
            R_inv = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
            shifted = pts - np.array([cx, cy], dtype=np.float32)
            rotated = shifted @ R_inv.T
            rx0, ry0 = rotated.min(axis=0)
            rx1, ry1 = rotated.max(axis=0)
            corners_rot = np.array(
                [[rx0, ry0], [rx1, ry0], [rx1, ry1], [rx0, ry1]],
                dtype=np.float32,
            )
            cos_b = math.cos(roll)
            sin_b = math.sin(roll)
            R = np.array([[cos_b, -sin_b], [sin_b, cos_b]], dtype=np.float32)
            corners = corners_rot @ R.T + np.array([cx, cy], dtype=np.float32)
            quad = [
                Point(
                    int(round(float(np.clip(c[0], 0, w - 1)))),
                    int(round(float(np.clip(c[1], 0, h - 1)))),
                )
                for c in corners
            ]
            roll_deg = math.degrees(roll)

            def pt(idx: int) -> Point:
                lm = landmarks[idx]
                return Point(int(lm.x * w), int(lm.y * h))

            nose       = pt(LM_NOSE_TIP)
            forehead_c = pt(LM_FOREHEAD)
            cheek_l    = pt(LM_CHEEK_LEFT)
            cheek_r    = pt(LM_CHEEK_RIGHT)
            brow_l     = pt(LM_LEFT_BROW)
            brow_r     = pt(LM_RIGHT_BROW)

            f_x = min(brow_l.x, brow_r.x)
            f_w = max(20, abs(brow_r.x - brow_l.x))
            top_y = forehead_c.y
            bottom_y = min(brow_l.y, brow_r.y) - 4
            f_h = max(12, bottom_y - top_y)
            forehead_box = BBox(f_x, top_y, f_w, f_h)

            cheek_size = max(20, f_w // 5)
            cheek_l_box = BBox(
                cheek_l.x - cheek_size // 2,
                cheek_l.y - cheek_size // 2,
                cheek_size,
                cheek_size,
            )
            cheek_r_box = BBox(
                cheek_r.x - cheek_size // 2,
                cheek_r.y - cheek_size // 2,
                cheek_size,
                cheek_size,
            )

            # 각 영역 박스를 face roll 각도로 회전시킨 4-corner quad.
            # 좌상→우상→우하→좌하 순서.
            cos_r = math.cos(roll)
            sin_r = math.sin(roll)
            R_face = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float32)

            def rotated_quad(cx_f: float, cy_f: float, bw: int, bh: int) -> List[Point]:
                hw, hh = bw / 2.0, bh / 2.0
                base = np.array(
                    [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]],
                    dtype=np.float32,
                )
                rot = base @ R_face.T + np.array([cx_f, cy_f], dtype=np.float32)
                return [
                    Point(
                        int(round(float(np.clip(p[0], 0, w - 1)))),
                        int(round(float(np.clip(p[1], 0, h - 1)))),
                    )
                    for p in rot
                ]

            forehead_quad = rotated_quad(f_x + f_w / 2.0, top_y + f_h / 2.0, f_w, f_h)
            cheek_l_quad = rotated_quad(cheek_l.x, cheek_l.y, cheek_size, cheek_size)
            cheek_r_quad = rotated_quad(cheek_r.x, cheek_r.y, cheek_size, cheek_size)

            out.append(FaceRegions(
                bbox=bbox,
                nose_tip=nose,
                forehead_center=forehead_c,
                cheek_left=cheek_l,
                cheek_right=cheek_r,
                forehead_box=forehead_box,
                cheek_left_box=cheek_l_box,
                cheek_right_box=cheek_r_box,
                quad=quad,
                forehead_quad=forehead_quad,
                cheek_left_quad=cheek_l_quad,
                cheek_right_quad=cheek_r_quad,
                roll_deg=roll_deg,
            ))
        return out
