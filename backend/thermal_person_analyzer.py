"""Thermal 영상에서 multi-person 머리 박스 + 안정적 트래킹.

검출 파이프:
  1. BGR → grayscale
  2. Adaptive percentile threshold (상위 N% 따뜻한 픽셀) — Otsu보다
     frame 간 일관성이 높음
  3. Morphology open/close
  4. ConnectedComponentsWithStats
  5. Person 필터 (사람 형태 vs 배경 가구):
     - area ∈ [MIN, MAX]
     - height ≥ MIN_HEIGHT
     - width ≤ height × MAX_W_TO_H (가로로 긴 가구/난방기 거부)
     - ROI 평균 밝기 ≥ MIN_BRIGHTNESS (약한 hot 영역 거부)

트래킹: 2-stage 매칭으로 ID 안정성 확보.
  1. IoU greedy primary
  2. Centroid distance backup (사람 크기에 비례한 threshold)
  - 매칭 안 된 detection → 새 ID
  - 매칭 안 된 track → misses++, MAX_MISSES 초과 폐기
  - age ≥ MIN_AGE인 track만 표시 (1-frame 노이즈 거부)
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int


@dataclass
class _Detection:
    bbox: Box
    head: Box
    area: int


@dataclass
class _Track:
    id: int
    bbox: Box
    head: Box
    area: int
    age: int = 0
    misses: int = 0
    last_seen_ts: float = 0.0


@dataclass
class TrackedPerson:
    id: int
    bbox: Box
    head: Box
    area: int


@dataclass
class ThermalPersonResult:
    timestamp: float
    people: List[TrackedPerson] = field(default_factory=list)


def _iou(a: Box, b: Box) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _smooth(old: Box, new: Box, alpha: float) -> Box:
    return Box(
        int(round(alpha * old.x + (1 - alpha) * new.x)),
        int(round(alpha * old.y + (1 - alpha) * new.y)),
        int(round(alpha * old.w + (1 - alpha) * new.w)),
        int(round(alpha * old.h + (1 - alpha) * new.h)),
    )


def _centroid(b: Box) -> Tuple[float, float]:
    return (b.x + b.w / 2.0, b.y + b.h / 2.0)


class ThermalPersonAnalyzer:
    # ---------- Detection 파라미터 (640×480 기준) ----------
    MIN_AREA = 400
    MAX_AREA = 220000
    MIN_HEIGHT = 20
    MAX_W_TO_H = 1.8
    PCT_THRESHOLD = 72       # 상위 28% 따뜻한 픽셀
    SMALL_BLOB_H = 80

    # Hot 검출 채널 — 차가운 것(커피, 음료)이 흰색/파랑으로 표시되는 colormap에서
    # brightness로 검출하면 차가운 게 잡힌다. R-B는 colormap의 hot=빨강/주황,
    # cold=파랑/흰색 패턴을 직접 반영해 차가운 것을 자동 제외.
    RB_FLOOR = 25            # R-B 절대 floor (Rainbow/Jet/Iron 계열에서)
    RB_BRIGHT_FLOOR = 30     # blob 안 평균 R-B
    GRAY_FLOOR = 110         # grayscale colormap fallback
    GRAY_BRIGHT_FLOOR = 120
    COLORMAP_STD_THR = 8     # R-B std가 이보다 작으면 grayscale colormap으로 간주

    # 한 사람이 안경/옷 주름 때문에 여러 blob으로 쪼개졌을 때 합치는 임계값.
    MERGE_X_OVERLAP = 0.4    # x 범위 겹침 비율 ≥ 이 값이면 merge 후보
    MERGE_Y_GAP_RATIO = 0.5  # 수직 gap ≤ blob 높이 × 이 비율이면 merge

    # ---------- Tracking 파라미터 ----------
    IOU_MATCH = 0.05         # 작은 blob도 매칭되도록 매우 낮게
    DIST_MATCH_RATIO = 1.0   # bbox 크기에 비례한 거리
    DIST_MATCH_ABS_MIN = 80  # 작은 blob도 80px 이동까지는 같은 사람
    MAX_MISSES = 25
    HOLD_FRAMES = 8
    MIN_AGE = 2
    SMOOTH_ALPHA = 0.65

    # Temporal mask EMA — 일시적 노이즈를 frame 간 평균으로 제거.
    MASK_EMA_ALPHA = 0.55    # past 비중. 너무 높으면 잔상, 너무 낮으면 노이즈
    MASK_EMA_BINARY_THR = 110  # EMA mask를 다시 binary로 만들 때 임계

    def __init__(self, thermal_source, fps: float = 5.0) -> None:
        self.source = thermal_source
        self.interval = 1.0 / max(fps, 1.0)
        self._latest: Optional[ThermalPersonResult] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Open: 작은 노이즈 제거 (작은 kernel). Close: gap 메우기 (큰 kernel).
        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
        self._tracks: List[_Track] = []
        self._next_id = 1
        # Mask EMA — frame 간 평균으로 일시적 노이즈 제거.
        self._mask_ema: Optional[np.ndarray] = None

    def start(self) -> "ThermalPersonAnalyzer":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def latest(self) -> dict:
        with self._lock:
            if self._latest is None:
                return {"timestamp": 0.0, "people": []}
            return {
                "timestamp": self._latest.timestamp,
                "people": [asdict(p) for p in self._latest.people],
            }

    def _run(self) -> None:
        last_seen = -1.0
        while not self._stop.is_set():
            try:
                frame = self.source.read(timeout=0.5)
            except Exception:
                time.sleep(0.05)
                continue
            if frame.timestamp == last_seen:
                time.sleep(self.interval / 2)
                continue
            last_seen = frame.timestamp

            try:
                detections = self._detect(frame.image)
            except Exception:
                time.sleep(self.interval)
                continue

            visible = self._update_tracks(detections, frame.timestamp)
            with self._lock:
                self._latest = ThermalPersonResult(
                    timestamp=frame.timestamp,
                    people=visible,
                )
            time.sleep(self.interval)

    def _detect(self, bgr_image: np.ndarray) -> List[_Detection]:
        # 차가운 영역(흰 커피, 음료)이 흰색으로 보이는 colormap에서 brightness로
        # 잡으면 차가운 게 잡힘. R-B 채널로 colormap의 색조 자체를 hot 지표로 사용.
        b, _, r = cv2.split(bgr_image)
        rb = cv2.subtract(r, b)
        if float(rb.std()) < self.COLORMAP_STD_THR:
            # Grayscale-like colormap → brightness fallback.
            feature = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            floor = self.GRAY_FLOOR
            blob_bright_floor = self.GRAY_BRIGHT_FLOOR
        else:
            feature = rb
            floor = self.RB_FLOOR
            blob_bright_floor = self.RB_BRIGHT_FLOOR

        pct = float(np.percentile(feature, self.PCT_THRESHOLD))
        thr = max(pct, floor)
        _, mask = cv2.threshold(feature, thr, 255, cv2.THRESH_BINARY)
        # Open(노이즈) → Close × 2(안경/옷 gap 메우기).
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel_close)

        # Temporal EMA — 일시적 깜빡임 제거. 지속적으로 hot인 영역만 살아남음.
        mask_f = mask.astype(np.float32)
        if self._mask_ema is None or self._mask_ema.shape != mask_f.shape:
            self._mask_ema = mask_f
        else:
            a = self.MASK_EMA_ALPHA
            self._mask_ema = a * self._mask_ema + (1.0 - a) * mask_f
        stable = (self._mask_ema >= self.MASK_EMA_BINARY_THR).astype(np.uint8) * 255

        num, _, stats, _ = cv2.connectedComponentsWithStats(stable, connectivity=8)
        raw: List[tuple[int, int, int, int, int]] = []
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < self.MIN_AREA or area > self.MAX_AREA:
                continue
            if h < self.MIN_HEIGHT:
                continue
            if w > h * self.MAX_W_TO_H:
                continue
            roi = feature[y:y + h, x:x + w]
            if roi.size == 0 or float(roi.mean()) < blob_bright_floor:
                continue
            raw.append((int(x), int(y), int(w), int(h), int(area)))

        # 수직 인접 blob 합치기: 한 사람이 안경/옷 때문에 잘려 잡힌 경우 한 박스로.
        merged = self._merge_vertical(raw)

        detections: List[_Detection] = []
        for x, y, w, h, area in merged:
            # 작은 blob → blob 전체를 head, 큰 blob → 상단 1/3.
            if h < self.SMALL_BLOB_H:
                head_x, head_y, head_w, head_h = x, y, w, h
            else:
                head_h = max(24, h // 3)
                head_w = max(24, min(w, int(head_h * 0.85)))
                head_x = x + (w - head_w) // 2
                head_y = y
            detections.append(_Detection(
                bbox=Box(x, y, w, h),
                head=Box(head_x, head_y, head_w, head_h),
                area=area,
            ))
        return detections

    def _merge_vertical(
        self, dets: List[tuple[int, int, int, int, int]],
    ) -> List[tuple[int, int, int, int, int]]:
        """수직으로 가깝고 x 범위가 겹치는 blob들을 한 사람으로 합침.

        한 사람 영역이 안경(차가운 렌즈), 셔츠 칼라 등에 의해 위/아래로
        분리될 때 같은 ID로 인식하기 위함. 합칠 게 없을 때까지 반복.
        """
        current = list(dets)
        while True:
            n = len(current)
            if n < 2:
                return current
            used = [False] * n
            result: List[tuple[int, int, int, int, int]] = []
            merged_any = False
            for i in range(n):
                if used[i]:
                    continue
                x, y, w, h, area = current[i]
                for j in range(i + 1, n):
                    if used[j]:
                        continue
                    x2, y2, w2, h2, area2 = current[j]
                    ox = max(0, min(x + w, x2 + w2) - max(x, x2))
                    min_w = min(w, w2)
                    if min_w <= 0 or ox / min_w < self.MERGE_X_OVERLAP:
                        continue
                    if y2 >= y + h:
                        gap = y2 - (y + h)
                    elif y >= y2 + h2:
                        gap = y - (y2 + h2)
                    else:
                        gap = 0  # 이미 수직으로 겹침
                    max_gap = max(h, h2) * self.MERGE_Y_GAP_RATIO
                    if gap > max_gap:
                        continue
                    nx = min(x, x2)
                    ny = min(y, y2)
                    nw = max(x + w, x2 + w2) - nx
                    nh = max(y + h, y2 + h2) - ny
                    x, y, w, h = nx, ny, nw, nh
                    area += area2
                    used[j] = True
                    merged_any = True
                used[i] = True
                result.append((x, y, w, h, area))
            current = result
            if not merged_any:
                return current

    def _update_tracks(
        self, detections: List[_Detection], ts: float,
    ) -> List[TrackedPerson]:
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        def assign(ti: int, di: int) -> None:
            t = self._tracks[ti]
            d = detections[di]
            t.bbox = _smooth(t.bbox, d.bbox, self.SMOOTH_ALPHA)
            t.head = _smooth(t.head, d.head, self.SMOOTH_ALPHA)
            t.area = d.area
            t.misses = 0
            t.age += 1
            t.last_seen_ts = ts
            matched_tracks.add(ti)
            matched_dets.add(di)

        # 1. IoU primary.
        iou_pairs: List[Tuple[float, int, int]] = []
        for ti, t in enumerate(self._tracks):
            for di, d in enumerate(detections):
                iou = _iou(t.bbox, d.bbox)
                if iou >= self.IOU_MATCH:
                    iou_pairs.append((iou, ti, di))
        iou_pairs.sort(key=lambda p: -p[0])
        for _, ti, di in iou_pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            assign(ti, di)

        # 2. Centroid distance backup — 작은 blob에도 절대 최소 거리 도입.
        dist_pairs: List[Tuple[float, int, int]] = []
        for ti, t in enumerate(self._tracks):
            if ti in matched_tracks:
                continue
            tcx, tcy = _centroid(t.bbox)
            for di, d in enumerate(detections):
                if di in matched_dets:
                    continue
                dcx, dcy = _centroid(d.bbox)
                dist = math.hypot(tcx - dcx, tcy - dcy)
                # 작은 blob도 잘 매칭되도록 절대 최소값을 도입.
                size_based = max(t.bbox.w, t.bbox.h) * self.DIST_MATCH_RATIO
                max_dist = max(self.DIST_MATCH_ABS_MIN, size_based)
                if dist < max_dist:
                    dist_pairs.append((dist, ti, di))
        dist_pairs.sort(key=lambda p: p[0])
        for _, ti, di in dist_pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            assign(ti, di)

        # 3. 매칭 안 된 detection → 새 track.
        for di, d in enumerate(detections):
            if di in matched_dets:
                continue
            self._tracks.append(_Track(
                id=self._next_id,
                bbox=d.bbox, head=d.head, area=d.area,
                age=1, misses=0, last_seen_ts=ts,
            ))
            self._next_id += 1

        # 4. 매칭 안 된 track → misses 증가, 한도 초과 폐기.
        for ti, t in enumerate(self._tracks):
            if ti not in matched_tracks:
                t.misses += 1
        self._tracks = [t for t in self._tracks if t.misses <= self.MAX_MISSES]

        # 5. 표시: 충분히 안정적인 것만.
        return [
            TrackedPerson(id=t.id, bbox=t.bbox, head=t.head, area=t.area)
            for t in self._tracks
            if t.misses <= self.HOLD_FRAMES and t.age >= self.MIN_AGE
        ]
