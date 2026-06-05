"""HeatSight GPU service — °C 변환 + RTMPose + SigLIP zero-shot + Qwen2-VL."""
from __future__ import annotations

import json
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from clip_runtime import clip_runtime
from pose_runtime import pose_runtime
from temperature import TemperatureCalculator
from vlm_runtime import vlm_runtime

app = FastAPI(title="HeatSight GPU Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

calc = TemperatureCalculator()
params_set = False


class ParamsBody(BaseModel):
    params: List[float]


class ConvertBody(BaseModel):
    raw: int
    board_temp_c: float
    gain: float
    pwm_target: float


def _decode_image(raw: bytes) -> np.ndarray:
    if not raw:
        raise HTTPException(400, "빈 이미지")
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "이미지 디코딩 실패")
    return img


@app.get("/health")
def health() -> dict:
    return {"ok": True, "params_set": params_set}


@app.post("/params")
def set_params(body: ParamsBody) -> dict:
    global params_set
    if len(body.params) < 5:
        raise HTTPException(400, f"params 최소 5개 필요. 받음: {len(body.params)}")
    try:
        calc.set_parameters(body.params)
    except Exception as e:
        raise HTTPException(500, str(e))
    params_set = True
    return {"ok": True, "params_count": len(body.params)}


@app.post("/convert")
def convert(body: ConvertBody) -> dict:
    if not params_set:
        raise HTTPException(409, "POST /params 먼저 호출")
    try:
        celsius = calc.compute(body.raw, body.board_temp_c, body.gain, body.pwm_target)
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"celsius": celsius}


@app.post("/pose")
async def pose(file: UploadFile = File(...)) -> dict:
    img = _decode_image(await file.read())
    try:
        poses = pose_runtime.infer(img)
    except Exception as e:
        raise HTTPException(500, f"pose infer 실패: {e}")
    h, w = img.shape[:2]
    return {"poses": poses, "image_w": w, "image_h": h}


@app.post("/behavior_clip")
async def behavior_clip(
    queries: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    img = _decode_image(await file.read())
    try:
        qs = json.loads(queries)
        if not isinstance(qs, list):
            raise ValueError
    except Exception:
        qs = [q.strip() for q in queries.split(",") if q.strip()]
    if not qs:
        raise HTTPException(400, "queries 비어있음")
    try:
        scores = clip_runtime.score(img, qs)
    except Exception as e:
        raise HTTPException(500, f"clip infer 실패: {e}")
    return {"scores": scores}


@app.post("/vlm/comfort")
async def vlm_comfort(file: UploadFile = File(...)) -> dict:
    img = _decode_image(await file.read())
    try:
        result = vlm_runtime.comfort(img)
    except Exception as e:
        raise HTTPException(500, f"vlm infer 실패: {e}")
    return result
