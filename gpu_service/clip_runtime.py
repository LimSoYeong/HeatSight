"""SigLIP zero-shot 텍스트-이미지 매칭.

각 query 텍스트에 대해 독립적 sigmoid 점수(0~1) 반환.
"""
from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL_ID = "google/siglip-base-patch16-256-multilingual"


class ClipRuntime:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    def _ensure(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._processor = AutoProcessor.from_pretrained(MODEL_ID)
                    self._model = (
                        AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
                        .to(self.device)
                        .eval()
                    )
        return self._processor, self._model

    @torch.inference_mode()
    def score(self, image_bgr: np.ndarray, queries: List[str]) -> dict:
        processor, model = self._ensure()
        image = Image.fromarray(image_bgr[..., ::-1])
        inputs = processor(
            text=queries,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        ).to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]
        probs = torch.sigmoid(logits).float().cpu().numpy()
        return {q: float(probs[i]) for i, q in enumerate(queries)}


clip_runtime = ClipRuntime()
