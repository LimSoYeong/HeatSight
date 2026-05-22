"""RGB + 열화상 동시 캡처 검증.

두 소스를 동시에 켜고 N초간 read()를 반복하여
- 양쪽 모두 정상 수신되는지
- 실제 fps가 얼마나 나오는지
- 프레임 도착 시각 차이(동기화 격차)가 어느 정도인지
를 확인한다.
"""

from __future__ import annotations

import time

import cv2

from capture import RGBSource, ThermalSource


def main() -> None:
    duration_s = 5.0
    warmup_s = 1.0

    with RGBSource() as rgb, ThermalSource() as thermal:
        print(f"RGB     resolved index: {rgb.resolved_index} ({rgb.name_substring!r})")
        print(f"Thermal resolved index: {thermal.resolved_index} ({thermal.name_substring!r})")
        print(f"{warmup_s}초 워밍업 중...")
        time.sleep(warmup_s)

        deltas_ms = []
        rgb_seen = thermal_seen = 0
        last_rgb = last_thermal = None

        t0 = time.monotonic()
        while time.monotonic() - t0 < duration_s:
            try:
                rf = rgb.read(timeout=0.5)
                if last_rgb is None or rf.timestamp != last_rgb.timestamp:
                    rgb_seen += 1
                last_rgb = rf
            except TimeoutError as e:
                print(f"  RGB timeout: {e}")
            try:
                tf = thermal.read(timeout=0.5)
                if last_thermal is None or tf.timestamp != last_thermal.timestamp:
                    thermal_seen += 1
                last_thermal = tf
            except TimeoutError as e:
                print(f"  Thermal timeout: {e}")
            if last_rgb and last_thermal:
                deltas_ms.append(abs(last_rgb.timestamp - last_thermal.timestamp) * 1000)
            time.sleep(1 / 60)

    elapsed = time.monotonic() - t0
    print(f"\n측정 구간: {elapsed:.2f}s")
    print(f"RGB     신규 프레임 수: {rgb_seen}  → {rgb_seen / elapsed:.1f} fps")
    print(f"Thermal 신규 프레임 수: {thermal_seen}  → {thermal_seen / elapsed:.1f} fps")
    if deltas_ms:
        avg = sum(deltas_ms) / len(deltas_ms)
        print(f"두 소스 최신 프레임 시각차: 평균 {avg:.1f}ms, 최대 {max(deltas_ms):.1f}ms")

    if last_rgb is not None:
        cv2.imwrite("/tmp/heatsight_rgb_latest.png", last_rgb.image)
        print(f"RGB 마지막: shape={last_rgb.image.shape} → /tmp/heatsight_rgb_latest.png")
    if last_thermal is not None:
        cv2.imwrite("/tmp/heatsight_thermal_latest.png", last_thermal.image)
        print(f"Thermal 마지막: shape={last_thermal.image.shape} → /tmp/heatsight_thermal_latest.png")


if __name__ == "__main__":
    main()
