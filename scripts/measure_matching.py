"""RGB→열화상 인물 매칭 일치율 측정.

RGB 얼굴 검출 박스를 호모그래피 H로 열화상 좌표에 투영하고,
독립적인 열화상 인물 검출(열점 기반) 결과와 교차 비교해 일치율을 산출한다.

판정 기준: 투영된 얼굴 박스 중심이 열화상 인물 머리 영역(1.5배 확장) 내부면 일치.
다인 환경에서는 greedy 일대일 대응(중복 매칭 불허).

사용법 (백엔드 실행 + 캘리브레이션 완료 + 카메라 앞에 사람):
  .venv/bin/python scripts/measure_matching.py            # 유효 샘플 100회
  .venv/bin/python scripts/measure_matching.py --samples 200 --interval 0.5
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def get(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=5) as r:
        return json.load(r)


def project(H, x: float, y: float) -> tuple[float, float]:
    w = H[2][0] * x + H[2][1] * y + H[2][2] or 1e-9
    return (
        (H[0][0] * x + H[0][1] * y + H[0][2]) / w,
        (H[1][0] * x + H[1][1] * y + H[1][2]) / w,
    )


def inside_expanded(px: float, py: float, box: dict, scale: float = 1.5) -> bool:
    cx, cy = box["x"] + box["w"] / 2, box["y"] + box["h"] / 2
    hw, hh = box["w"] * scale / 2, box["h"] * scale / 2
    return (cx - hw) <= px <= (cx + hw) and (cy - hh) <= py <= (cy + hh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--samples", type=int, default=100, help="유효 샘플 수")
    ap.add_argument("--interval", type=float, default=0.5, help="폴링 간격(s)")
    ap.add_argument("--timeout", type=float, default=600, help="전체 제한(s)")
    args = ap.parse_args()

    cal = get(args.url, "/api/calibration")
    if not cal.get("homography_ready"):
        raise SystemExit("캘리브레이션(H)이 없습니다. 먼저 캘리브레이션을 완료하세요.")
    H = cal["homography"]
    print(f"[i] H 로드 (대응점 {cal['pair_count']}쌍). 유효 샘플 {args.samples}회 수집 시작 — 카메라 앞에 머물러 주세요.")

    matched = 0
    attempts = 0          # 얼굴 단위 시도 수
    valid_polls = 0       # 양쪽 모두 검출된 폴링 수
    person_counts: dict[int, int] = {}
    t0 = time.time()

    while valid_polls < args.samples and time.time() - t0 < args.timeout:
        time.sleep(args.interval)
        try:
            faces = get(args.url, "/api/face").get("faces", [])
            people = get(args.url, "/api/thermal/persons").get("people", [])
        except Exception:
            continue
        if not faces or not people:
            continue
        valid_polls += 1
        n = len(faces)
        person_counts[n] = person_counts.get(n, 0) + 1

        heads = [p.get("head") or p.get("bbox") for p in people]
        heads = [h for h in heads if h]
        used: set[int] = set()
        for f in faces:
            b = f["bbox"]
            px, py = project(H, b["x"] + b["w"] / 2, b["y"] + b["h"] / 2)
            attempts += 1
            for i, h in enumerate(heads):
                if i in used:
                    continue
                if inside_expanded(px, py, h):
                    matched += 1
                    used.add(i)
                    break
        if valid_polls % 20 == 0:
            rate = 100 * matched / max(1, attempts)
            print(f"  {valid_polls}/{args.samples} 폴링 · 매칭 {matched}/{attempts} ({rate:.1f}%)")

    rate = 100 * matched / max(1, attempts)
    print("\n===== 결과 =====")
    print(f"유효 폴링        : {valid_polls}회 (양쪽 검출 동시 성립)")
    print(f"동시 인원 분포    : {dict(sorted(person_counts.items()))}")
    print(f"얼굴 매칭 시도    : {attempts}회")
    print(f"매칭 성공        : {matched}회")
    print(f"교차 일치율      : {rate:.1f}%")
    print("\n보고서 문구 예시: RGB 얼굴 검출의 호모그래피 투영과 독립적인 열화상 인물 검출 간")
    print(f"교차 일치율 {rate:.1f}% (유효 {valid_polls}회 폴링, 얼굴 {attempts}건) 측정.")


if __name__ == "__main__":
    main()
