"""원격 GPU 서비스(RTMPose-L) 호출 기반 Pose 분석.

기존 PoseAnalyzer(MediaPipe lite)와 같은 latest() dict 인터페이스를 유지하므로
server.py와 프론트엔드는 변경 없이 그대로 동작한다.

RGBSource에서 latest frame을 가져와 JPEG 인코딩 후 GPU /pose에 POST.
응답의 17 COCO keypoints를 PoseRegions 구조(어깨/팔꿈치/손목/엉덩이/손 박스/몸통 박스)로 변환.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import cv2
import httpx
import numpy as np


# COCO 17 keypoints — RTMLib Body 모델 출력 순서
COCO_NOSE       = 0
COCO_SHOULDER_L = 5
COCO_SHOULDER_R = 6
COCO_ELBOW_L    = 7
COCO_ELBOW_R    = 8
COCO_WRIST_L    = 9
COCO_WRIST_R    = 10
COCO_HIP_L      = 11
COCO_HIP_R      = 12

VIS_TH = 0.3


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


class RemotePoseAnalyzer:
    def __init__(self, rgb_source, url: str, fps: float = 8.0,
                 jpeg_quality: int = 70) -> None:
        self.rgb_source = rgb_source
        self.url = url
        self.interval = 1.0 / max(fps, 1.0)
        self.jpeg_quality = jpeg_quality
        self._latest: Optional[PoseResult] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._client: Optional[httpx.Client] = None
        self._last_error: Optional[str] = None

    def start(self) -> "RemotePoseAnalyzer":
        self._client = httpx.Client(timeout=8.0)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._client is not None:
            self._client.close()

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

            ok, jpeg = cv2.imencode(
                ".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )
            if not ok:
                time.sleep(self.interval)
                continue

            try:
                assert self._client is not None
                r = self._client.post(
                    self.url,
                    files={"file": ("frame.jpg", jpeg.tobytes(), "image/jpeg")},
                )
                if r.status_code != 200:
                    self._last_error = f"HTTP {r.status_code}"
                    time.sleep(self.interval)
                    continue
                data = r.json()
                self._last_error = None
            except Exception as e:
                self._last_error = f"{type(e).__name__}: {e}"
                time.sleep(self.interval)
                continue

            poses = self._convert(data.get("poses", []))
            with self._lock:
                self._latest = PoseResult(timestamp=frame.timestamp, poses=poses)
            time.sleep(self.interval)

    @staticmethod
    def _convert(rtm_poses: list) -> List[PoseRegions]:
        out: List[PoseRegions] = []
        for p in rtm_poses:
            kp = p.get("keypoints", [])
            if len(kp) < 13:
                continue

            def pt(idx: int) -> Pt:
                k = kp[idx]
                return Pt(
                    x=int(k["x"]),
                    y=int(k["y"]),
                    visible=bool(k["score"] > VIS_TH),
                )

            shoulder_l = pt(COCO_SHOULDER_L)
            shoulder_r = pt(COCO_SHOULDER_R)
            elbow_l    = pt(COCO_ELBOW_L)
            elbow_r    = pt(COCO_ELBOW_R)
            wrist_l    = pt(COCO_WRIST_L)
            wrist_r    = pt(COCO_WRIST_R)
            hip_l      = pt(COCO_HIP_L)
            hip_r      = pt(COCO_HIP_R)

            def hand_box(wrist: Pt) -> Optional[BBox]:
                if not wrist.visible:
                    return None
                size = 60
                return BBox(
                    x=max(0, wrist.x - size // 2),
                    y=max(0, wrist.y - size // 2),
                    w=size, h=size,
                )

            torso_pts = [t for t in (shoulder_l, shoulder_r, hip_l, hip_r) if t.visible]
            torso_box: Optional[BBox] = None
            if len(torso_pts) >= 3:
                xs = [t.x for t in torso_pts]
                ys = [t.y for t in torso_pts]
                torso_box = BBox(
                    x=min(xs), y=min(ys),
                    w=max(1, max(xs) - min(xs)),
                    h=max(1, max(ys) - min(ys)),
                )

            out.append(PoseRegions(
                shoulder_l=shoulder_l, shoulder_r=shoulder_r,
                elbow_l=elbow_l, elbow_r=elbow_r,
                wrist_l=wrist_l, wrist_r=wrist_r,
                hip_l=hip_l, hip_r=hip_r,
                hand_l_box=hand_box(wrist_l),
                hand_r_box=hand_box(wrist_r),
                torso_box=torso_box,
            ))
        return out
