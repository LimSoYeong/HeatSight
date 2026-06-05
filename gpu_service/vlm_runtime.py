"""Qwen2-VL-2B wrapper — 자연어 추론으로 더위/추위 판단."""
from __future__ import annotations

import threading

import numpy as np
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration as _Qwen2VL, AutoProcessor

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

DEFAULT_PROMPT = (
    "Look at the person in this image. Are they showing signs of feeling hot, "
    "cold, or comfortable? Consider their clothing, posture, hand gestures "
    "(fanning, hugging arms, holding a blanket, etc.). "
    "First word of your answer must be exactly one of: hot, cold, neutral. "
    "Then in 5 words explain why."
)


class VlmRuntime:
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
                        _Qwen2VL.from_pretrained(
                            MODEL_ID, torch_dtype=torch.float16
                        )
                        .to(self.device)
                        .eval()
                    )
        return self._processor, self._model

    @torch.inference_mode()
    def comfort(self, image_bgr: np.ndarray, prompt: str = DEFAULT_PROMPT) -> dict:
        processor, model = self._ensure()
        image = Image.fromarray(image_bgr[..., ::-1])
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        ).to(self.device)
        out_ids = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        generated = out_ids[:, inputs.input_ids.shape[1] :]
        answer = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

        first = answer.lower().split()[0] if answer else "neutral"
        first = first.strip(".,:!?")
        if first not in ("hot", "cold", "neutral"):
            first = "neutral"
        return {"comfort": first, "answer": answer}


vlm_runtime = VlmRuntime()
