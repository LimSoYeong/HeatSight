"""RTMLib (RTMPose) wrapper. 한 프로세스에 하나의 모델 인스턴스."""
from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np
from rtmlib import Body


class PoseRuntime:
    def __init__(self, mode: str = "performance", device: str = "cuda") -> None:
        self.mode = mode
        self.device = device
        self._body: Optional[Body] = None
        self._lock = threading.Lock()

    def _ensure(self) -> Body:
        if self._body is None:
            with self._lock:
                if self._body is None:
                    self._body = Body(
                        mode=self.mode,
                        to_openpose=False,
                        backend="onnxruntime",
                        device=self.device,
                    )
        return self._body

    def infer(self, image_bgr: np.ndarray) -> list[dict]:
        body = self._ensure()
        keypoints, scores = body(image_bgr)
        if keypoints is None or len(keypoints) == 0:
            return []
        result = []
        for kp_set, sc_set in zip(keypoints, scores):
            pts = []
            xs, ys = [], []
            for (x, y), s in zip(kp_set, sc_set):
                pts.append({"x": float(x), "y": float(y), "score": float(s)})
                if s > 0.3:
                    xs.append(float(x))
                    ys.append(float(y))
            bbox = None
            if xs:
                bbox = {
                    "x": int(min(xs)),
                    "y": int(min(ys)),
                    "w": int(max(xs) - min(xs)),
                    "h": int(max(ys) - min(ys)),
                }
            result.append({"keypoints": pts, "bbox": bbox})
        return result


pose_runtime = PoseRuntime(mode="performance", device="cuda")
