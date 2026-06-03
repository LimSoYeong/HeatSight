"""휴리스틱 행동 신호 분석.

Pose + Face의 최근 결과를 polling해서 행동 점수와 쾌적도 신호 계산.

신호:
  arms_crossed  : 양 wrist가 가슴 영역에서 X자 (춥다)
  hunched       : 어깨 폭이 좁아지거나 어깨가 안쪽으로 모임 (춥다)
  hands_up      : wrist가 어깨 위에 자주 (덥다 / 부채질)
  fanning       : wrist의 최근 시간축 진동 (덥다)
  touching_face : wrist가 face bbox 안 (애매)

comfort_signal:
  hot     = fanning ↑ or hands_up ↑
  cold    = arms_crossed ↑ or hunched ↑
  neutral = 둘 다 약함
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Deque, Dict, Optional

import cv2
import httpx


@dataclass
class BehaviorSnapshot:
    timestamp: float
    arms_crossed: float
    hunched: float
    hands_up: float
    fanning: float
    touching_face: float
    comfort: str  # "hot" | "cold" | "neutral"  — heuristic
    clip_scores: Dict[str, float] = field(default_factory=dict)
    vlm_comfort: Optional[str] = None  # "hot" | "cold" | "neutral"
    vlm_answer: Optional[str] = None
    fused_comfort: str = "neutral"  # heuristic + clip + vlm 융합


# CLIP zero-shot query 셋 — 각 카테고리당 한 줄. 다국어 SigLIP이라 영어 사용.
CLIP_QUERIES_HOT = [
    "a person fanning themselves with their hand",
    "a person wiping sweat from their face",
    "a person taking off their jacket or shirt",
]
CLIP_QUERIES_COLD = [
    "a person hugging their arms across their chest",
    "a person hunched and shivering",
    "a person wrapped in a blanket",
]


class BehaviorAnalyzer:
    HISTORY_SEC = 3.0
    POLL_HZ = 5.0
    HUNCH_BASELINE_SEC = 8.0
    CLIP_HZ = 1.0          # CLIP 1초 1회
    VLM_PERIOD_SEC = 6.0   # VLM 6초 1회

    def __init__(
        self,
        hub,
        gpu_url: Optional[str] = None,
    ) -> None:
        self.hub = hub
        self.gpu_url = gpu_url  # e.g. "http://101.79.21.220:9000"
        self._latest: Optional[BehaviorSnapshot] = None
        self._clip_scores: Dict[str, float] = {}
        self._vlm_comfort: Optional[str] = None
        self._vlm_answer: Optional[str] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._heur_thread: Optional[threading.Thread] = None
        self._clip_thread: Optional[threading.Thread] = None
        self._vlm_thread: Optional[threading.Thread] = None
        self._wrist_l_hist: Deque[tuple[float, float, float]] = deque(maxlen=64)
        self._wrist_r_hist: Deque[tuple[float, float, float]] = deque(maxlen=64)
        self._shoulder_width_hist: Deque[tuple[float, float]] = deque(maxlen=128)
        self._http: Optional[httpx.Client] = None

    def start(self) -> "BehaviorAnalyzer":
        if self.gpu_url is not None:
            self._http = httpx.Client(timeout=15.0)
        self._heur_thread = threading.Thread(target=self._run_heuristic, daemon=True)
        self._heur_thread.start()
        if self.gpu_url is not None:
            self._clip_thread = threading.Thread(target=self._run_clip, daemon=True)
            self._clip_thread.start()
            self._vlm_thread = threading.Thread(target=self._run_vlm, daemon=True)
            self._vlm_thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        for t in (self._heur_thread, self._clip_thread, self._vlm_thread):
            if t is not None:
                t.join(timeout=2.0)
        if self._http is not None:
            self._http.close()

    def latest(self) -> dict:
        with self._lock:
            if self._latest is None:
                return {
                    "timestamp": 0.0,
                    "arms_crossed": 0.0,
                    "hunched": 0.0,
                    "hands_up": 0.0,
                    "fanning": 0.0,
                    "touching_face": 0.0,
                    "comfort": "neutral",
                }
            return asdict(self._latest)

    def _run_heuristic(self) -> None:
        interval = 1.0 / self.POLL_HZ
        while not self._stop.is_set():
            try:
                snap = self._compute()
            except Exception:
                snap = None
            if snap is not None:
                with self._lock:
                    snap.clip_scores = dict(self._clip_scores)
                    snap.vlm_comfort = self._vlm_comfort
                    snap.vlm_answer = self._vlm_answer
                    snap.fused_comfort = self._fuse(
                        snap.comfort, self._clip_scores, self._vlm_comfort
                    )
                    self._latest = snap
            time.sleep(interval)

    def _run_clip(self) -> None:
        interval = 1.0 / self.CLIP_HZ
        queries = CLIP_QUERIES_HOT + CLIP_QUERIES_COLD
        import json as _json
        while not self._stop.is_set():
            jpeg = self._grab_jpeg()
            if jpeg is None:
                time.sleep(interval); continue
            try:
                assert self._http is not None
                r = self._http.post(
                    f"{self.gpu_url}/behavior_clip",
                    data={"queries": _json.dumps(queries)},
                    files={"file": ("frame.jpg", jpeg, "image/jpeg")},
                    timeout=10.0,
                )
                if r.status_code == 200:
                    scores = r.json().get("scores", {})
                    with self._lock:
                        self._clip_scores = scores
            except Exception:
                pass
            time.sleep(interval)

    def _run_vlm(self) -> None:
        while not self._stop.is_set():
            jpeg = self._grab_jpeg()
            if jpeg is None:
                time.sleep(self.VLM_PERIOD_SEC); continue
            try:
                assert self._http is not None
                r = self._http.post(
                    f"{self.gpu_url}/vlm/comfort",
                    files={"file": ("frame.jpg", jpeg, "image/jpeg")},
                    timeout=30.0,
                )
                if r.status_code == 200:
                    data = r.json()
                    with self._lock:
                        self._vlm_comfort = data.get("comfort")
                        self._vlm_answer = data.get("answer")
            except Exception:
                pass
            time.sleep(self.VLM_PERIOD_SEC)

    def _grab_jpeg(self) -> Optional[bytes]:
        if self.hub.rgb is None:
            return None
        try:
            frame = self.hub.rgb.read(timeout=1.0)
        except Exception:
            return None
        ok, buf = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes() if ok else None

    # Fusion weights — 합 1.0. CLIP을 1차 신호로 두고(다양한 동작 의미를 자연어 query로
    # 직접 매칭), 휴리스틱과 VLM은 서로 상보적인 약점(노이즈 vs 보수적 편향)을 가지므로
    # 균형 있게 0.3을 부여.
    W_HEUR = 0.3
    W_CLIP = 0.4
    W_VLM  = 0.3

    @staticmethod
    def _fuse(heur: str, clip_scores: Dict[str, float], vlm: Optional[str]) -> str:
        """heuristic + clip + vlm을 가중합해서 최종 comfort 결정. 가중치 합 1.0."""
        W_H, W_C, W_V = BehaviorAnalyzer.W_HEUR, BehaviorAnalyzer.W_CLIP, BehaviorAnalyzer.W_VLM
        hot = 0.0
        cold = 0.0
        if heur == "hot": hot += W_H
        if heur == "cold": cold += W_H
        if clip_scores:
            for q in CLIP_QUERIES_HOT:
                hot += clip_scores.get(q, 0.0) * W_C
            for q in CLIP_QUERIES_COLD:
                cold += clip_scores.get(q, 0.0) * W_C
        if vlm == "hot": hot += W_V
        elif vlm == "cold": cold += W_V
        # 임계 0.45 — 어느 축도 단독으로는 결정 못 하고 최소 두 축 합의가 필요한 설계
        if hot > 0.45 and hot > cold + 0.1:
            return "hot"
        if cold > 0.45 and cold > hot + 0.1:
            return "cold"
        return "neutral"

    def _compute(self) -> Optional[BehaviorSnapshot]:
        if self.hub.pose is None:
            return None
        pose_data = self.hub.pose.latest()
        if not pose_data.get("poses"):
            return None
        p = pose_data["poses"][0]
        ts = float(pose_data.get("timestamp") or time.monotonic())

        face_bbox = None
        if self.hub.face is not None:
            face_data = self.hub.face.latest()
            if face_data.get("faces"):
                face_bbox = face_data["faces"][0].get("bbox")

        sl, sr = p.get("shoulder_l"), p.get("shoulder_r")
        wl, wr = p.get("wrist_l"), p.get("wrist_r")
        el, er = p.get("elbow_l"), p.get("elbow_r")

        # wrist 시계열 누적
        if wl and wl.get("visible"):
            self._wrist_l_hist.append((ts, float(wl["x"]), float(wl["y"])))
        if wr and wr.get("visible"):
            self._wrist_r_hist.append((ts, float(wr["x"]), float(wr["y"])))

        # shoulder 폭 시계열 (베이스라인 추정용)
        sw = None
        if sl and sr and sl.get("visible") and sr.get("visible"):
            sw = abs(sl["x"] - sr["x"])
            self._shoulder_width_hist.append((ts, float(sw)))

        cut = ts - self.HISTORY_SEC
        self._wrist_l_hist = deque(
            ((t, x, y) for (t, x, y) in self._wrist_l_hist if t >= cut),
            maxlen=64,
        )
        self._wrist_r_hist = deque(
            ((t, x, y) for (t, x, y) in self._wrist_r_hist if t >= cut),
            maxlen=64,
        )
        baseline_cut = ts - self.HUNCH_BASELINE_SEC
        self._shoulder_width_hist = deque(
            ((t, w) for (t, w) in self._shoulder_width_hist if t >= baseline_cut),
            maxlen=128,
        )

        arms_crossed  = self._arms_crossed_score(sl, sr, wl, wr)
        hunched       = self._hunched_score(sw)
        hands_up      = self._hands_up_score(sl, sr, wl, wr, el, er)
        fanning       = self._fanning_score(sw)
        touching_face = self._touching_face_score(wl, wr, face_bbox)

        # 쾌적도 신호 — 가장 큰 쪽으로
        hot_score  = max(fanning, hands_up * 0.6)
        cold_score = max(arms_crossed, hunched * 0.7)
        if hot_score > 0.5 and hot_score > cold_score + 0.1:
            comfort = "hot"
        elif cold_score > 0.5 and cold_score > hot_score + 0.1:
            comfort = "cold"
        else:
            comfort = "neutral"

        return BehaviorSnapshot(
            timestamp=ts,
            arms_crossed=arms_crossed,
            hunched=hunched,
            hands_up=hands_up,
            fanning=fanning,
            touching_face=touching_face,
            comfort=comfort,
        )

    # ---- 개별 신호 계산기 ----

    @staticmethod
    def _arms_crossed_score(sl, sr, wl, wr) -> float:
        """양 wrist가 반대편 어깨 영역으로 X자 교차한 정도."""
        if not (sl and sr and wl and wr
                and sl.get("visible") and sr.get("visible")
                and wl.get("visible") and wr.get("visible")):
            return 0.0
        # 화면상 sl이 더 우측, sr이 더 좌측 (또는 그 반대)일 수 있음. 좌우 정렬.
        if sl["x"] < sr["x"]:
            left_s, right_s = sl, sr
        else:
            left_s, right_s = sr, sl
        if wl["x"] < wr["x"]:
            left_w, right_w = wl, wr
        else:
            left_w, right_w = wr, wl
        # X자: 좌측 wrist가 우측 어깨 쪽, 우측 wrist가 좌측 어깨 쪽
        shoulder_w = max(1.0, right_s["x"] - left_s["x"])
        center = (left_s["x"] + right_s["x"]) / 2
        # 양 wrist가 어깨 중심 근처 + 어깨 y 근처
        cx_l = (left_w["x"] - center) / shoulder_w
        cx_r = (right_w["x"] - center) / shoulder_w
        # 좌측 wrist는 중심 또는 약간 우측(양수), 우측 wrist는 중심 또는 약간 좌측(음수)
        crossed = max(0.0, cx_l) * max(0.0, -cx_r)  # 0~0.25 정도 기대
        crossed = min(1.0, crossed * 8.0)
        # y가 어깨 근처 (가슴 영역)
        shoulder_y = (left_s["y"] + right_s["y"]) / 2
        y_off = (abs(left_w["y"] - shoulder_y) + abs(right_w["y"] - shoulder_y)) / 2 / shoulder_w
        y_score = max(0.0, 1.0 - y_off * 2.0)
        return float(min(1.0, crossed * y_score))

    def _hunched_score(self, current_sw: Optional[float]) -> float:
        """현재 shoulder 폭이 베이스라인보다 좁아진 정도."""
        if current_sw is None or len(self._shoulder_width_hist) < 8:
            return 0.0
        widths = sorted(w for _, w in self._shoulder_width_hist)
        baseline = widths[int(len(widths) * 0.75)]  # 75 percentile
        if baseline <= 1:
            return 0.0
        ratio = current_sw / baseline
        # ratio가 0.85 이하면 점수 증가, 0.7 이하면 1.0
        score = max(0.0, (0.85 - ratio) / 0.15)
        return float(min(1.0, score))

    @staticmethod
    def _hands_up_score(sl, sr, wl, wr, el, er) -> float:
        """wrist가 어깨 y보다 위에 있는 정도."""
        if not (sl and sr and sl.get("visible") and sr.get("visible")):
            return 0.0
        shoulder_y = (sl["y"] + sr["y"]) / 2
        scale = max(1.0, abs(sl["x"] - sr["x"]))
        scores = []
        for w in (wl, wr):
            if w and w.get("visible"):
                # y가 더 작으면 (= 위로 올라감) 점수 증가
                diff = (shoulder_y - w["y"]) / scale
                scores.append(max(0.0, min(1.0, diff)))
        if not scores:
            return 0.0
        return float(max(scores))

    def _fanning_score(self, current_sw: Optional[float]) -> float:
        """wrist 위치의 최근 시간축 분산."""
        scale = current_sw if current_sw and current_sw > 1 else 100.0

        def hist_var(hist: Deque[tuple[float, float, float]]) -> float:
            if len(hist) < 5:
                return 0.0
            xs = [x for _, x, _ in hist]
            ys = [y for _, _, y in hist]
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            vx = sum((x - mx) ** 2 for x in xs) / len(xs)
            vy = sum((y - my) ** 2 for y in ys) / len(ys)
            return (vx + vy) ** 0.5

        std = max(hist_var(self._wrist_l_hist), hist_var(self._wrist_r_hist))
        normalized = std / max(1.0, scale)
        # 0.15 정도면 명확히 흔드는 중. 0.5+면 격렬.
        return float(min(1.0, max(0.0, (normalized - 0.05) / 0.25)))

    @staticmethod
    def _touching_face_score(wl, wr, face_bbox) -> float:
        if not face_bbox:
            return 0.0
        fx, fy = face_bbox["x"], face_bbox["y"]
        fw, fh = face_bbox["w"], face_bbox["h"]
        # 약간 확장
        margin_x, margin_y = fw * 0.15, fh * 0.15
        x0, y0 = fx - margin_x, fy - margin_y
        x1, y1 = fx + fw + margin_x, fy + fh + margin_y
        for w in (wl, wr):
            if w and w.get("visible"):
                if x0 <= w["x"] <= x1 and y0 <= w["y"] <= y1:
                    return 1.0
        return 0.0
