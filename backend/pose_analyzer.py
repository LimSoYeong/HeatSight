"""RGB 영상 자세/손 분석 — MediaPipe Tasks API (PoseLandmarker).

33 keypoints (어깨/팔꿈치/손목/손가락/엉덩이 등) 중 HeatSight에 쓸 부분만 추출:
  - 양 어깨/팔꿈치/손목/엉덩이 점
  - 양 손 박스 (손목 + 손가락 키포인트 묶음)
  - 몸통 박스 (어깨~엉덩이)

향후 thermal 매핑에서 활용:
  - 손 영역의 raw 온도 → 추위 신호 ("손이 차가워졌다")
  - 몸통 영역 → 옷차림/활동량 추정
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


MODEL_PATH = Path(__file__).resolve().parent / "models" / "pose_landmarker_lite.task"

# Pose Landmarker 33 keypoint 인덱스
LM_NOSE        = 0
LM_SHOULDER_L  = 11
LM_SHOULDER_R  = 12
LM_ELBOW_L     = 13
LM_ELBOW_R     = 14
LM_WRIST_L     = 15
LM_WRIST_R     = 16
LM_PINKY_L     = 17
LM_PINKY_R     = 18
LM_INDEX_L     = 19
LM_INDEX_R     = 20
LM_THUMB_L     = 21
LM_THUMB_R     = 22
LM_HIP_L       = 23
LM_HIP_R       = 24


@dataclass
class Pt:
    x: int
    y: int
    visible: bool


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int


@dataclass
class PoseRegions:
    shoulder_l: Pt
    shoulder_r: Pt
    elbow_l: Pt
    elbow_r: Pt
    wrist_l: Pt
    wrist_r: Pt
    hip_l: Pt
    hip_r: Pt
    hand_l_box: Optional[BBox]
    hand_r_box: Optional[BBox]
    torso_box: Optional[BBox]


@dataclass
class PoseResult:
    timestamp: float
    poses: List[PoseRegions]


class PoseAnalyzer:
    def __init__(self, rgb_source, fps: float = 10.0) -> None:
        self.rgb_source = rgb_source
        self.interval = 1.0 / max(fps, 1.0)
        self._latest: Optional[PoseResult] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._landmarker: Optional[mp_vision.PoseLandmarker] = None

    def start(self) -> "PoseAnalyzer":
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"pose_landmarker 모델이 없음: {MODEL_PATH}")
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
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
                return {"timestamp": 0.0, "poses": []}
            return {
                "timestamp": self._latest.timestamp,
                "poses": [asdict(p) for p in self._latest.poses],
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

            image_rgb = np.ascontiguousarray(frame.image[:, :, ::-1])
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            ts_ms = int(frame.timestamp * 1000)
            assert self._landmarker is not None
            try:
                result = self._landmarker.detect_for_video(mp_image, ts_ms)
            except Exception:
                time.sleep(self.interval)
                continue
            poses = self._extract(result, frame.image.shape[:2])

            with self._lock:
                self._latest = PoseResult(timestamp=frame.timestamp, poses=poses)

            time.sleep(self.interval)

    @staticmethod
    def _extract(result, shape) -> List[PoseRegions]:
        h, w = shape
        out: List[PoseRegions] = []
        pose_landmarks_list = getattr(result, "pose_landmarks", None) or []
        if not pose_landmarks_list:
            return out

        for landmarks in pose_landmarks_list:
            def pt(idx: int, th: float = 0.5) -> Pt:
                lm = landmarks[idx]
                return Pt(
                    x=int(lm.x * w),
                    y=int(lm.y * h),
                    visible=bool(getattr(lm, "visibility", 1.0) > th),
                )

            shoulder_l = pt(LM_SHOULDER_L)
            shoulder_r = pt(LM_SHOULDER_R)
            elbow_l    = pt(LM_ELBOW_L)
            elbow_r    = pt(LM_ELBOW_R)
            wrist_l    = pt(LM_WRIST_L)
            wrist_r    = pt(LM_WRIST_R)
            hip_l      = pt(LM_HIP_L, th=0.3)
            hip_r      = pt(LM_HIP_R, th=0.3)

            hand_l_box = PoseAnalyzer._hand_box(
                landmarks, w, h, LM_WRIST_L, [LM_PINKY_L, LM_INDEX_L, LM_THUMB_L]
            )
            hand_r_box = PoseAnalyzer._hand_box(
                landmarks, w, h, LM_WRIST_R, [LM_PINKY_R, LM_INDEX_R, LM_THUMB_R]
            )

            torso_pts: list[tuple[int, int]] = []
            for idx in (LM_SHOULDER_L, LM_SHOULDER_R, LM_HIP_L, LM_HIP_R):
                lm = landmarks[idx]
                if getattr(lm, "visibility", 1.0) > 0.3:
                    torso_pts.append((int(lm.x * w), int(lm.y * h)))
            torso_box: Optional[BBox] = None
            if len(torso_pts) >= 3:
                xs = [p[0] for p in torso_pts]
                ys = [p[1] for p in torso_pts]
                torso_box = BBox(
                    x=min(xs), y=min(ys),
                    w=max(1, max(xs) - min(xs)),
                    h=max(1, max(ys) - min(ys)),
                )

            out.append(PoseRegions(
                shoulder_l=shoulder_l,
                shoulder_r=shoulder_r,
                elbow_l=elbow_l,
                elbow_r=elbow_r,
                wrist_l=wrist_l,
                wrist_r=wrist_r,
                hip_l=hip_l,
                hip_r=hip_r,
                hand_l_box=hand_l_box,
                hand_r_box=hand_r_box,
                torso_box=torso_box,
            ))
        return out

    @staticmethod
    def _hand_box(landmarks, w: int, h: int, wrist_idx: int,
                  finger_indices: list[int]) -> Optional[BBox]:
        wrist = landmarks[wrist_idx]
        if getattr(wrist, "visibility", 1.0) < 0.4:
            return None
        pts = [landmarks[i] for i in [wrist_idx, *finger_indices]
               if getattr(landmarks[i], "visibility", 1.0) > 0.3]
        if not pts:
            return None
        xs = [p.x * w for p in pts]
        ys = [p.y * h for p in pts]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        spread = max(max(xs) - min(xs), max(ys) - min(ys))
        size = max(40, int(spread * 1.6))
        return BBox(
            x=max(0, int(cx - size / 2)),
            y=max(0, int(cy - size / 2)),
            w=size,
            h=size,
        )
