"""HeatSight 카메라 캡처 유틸.

RGB 카메라(MacBook 내장)와 열화상 카메라(Cellplus/Obsidian Sensors)를
모두 ffmpeg avfoundation pipe로 읽어 동일 인터페이스로 다룬다.

OpenCV cv2.VideoCapture는 AVFoundation 디바이스 인덱싱이 ffmpeg와
일치하지 않아 사용하지 않는다.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float


def list_avfoundation_video_devices() -> list[tuple[int, str]]:
    """ffmpeg가 보는 AVFoundation 비디오 디바이스를 [(index, name), ...]로 반환."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation",
         "-list_devices", "true", "-i", ""],
        capture_output=True, text=True,
    )
    devices: list[tuple[int, str]] = []
    in_video = False
    for line in result.stderr.splitlines():
        if "AVFoundation video devices:" in line:
            in_video = True
            continue
        if "AVFoundation audio devices:" in line:
            in_video = False
            continue
        if not in_video:
            continue
        m = re.search(r"\[(\d+)\]\s+(.+?)\s*$", line)
        if m:
            devices.append((int(m.group(1)), m.group(2)))
    return devices


def find_video_device_index(name_substring: str) -> int:
    devices = list_avfoundation_video_devices()
    for idx, name in devices:
        if name_substring.lower() in name.lower():
            return idx
    available = ", ".join(f"[{i}] {n}" for i, n in devices)
    raise RuntimeError(
        f"'{name_substring}' 디바이스를 찾지 못함. 사용 가능: {available}"
    )


class FrameSource:
    def __init__(self) -> None:
        self._latest: Optional[Frame] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "FrameSource":
        self._open()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def read(self, timeout: float = 2.0) -> Frame:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest is not None:
                    return self._latest
            time.sleep(0.01)
        raise TimeoutError(f"{type(self).__name__}: 프레임 수신 타임아웃 ({timeout}s)")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._close()

    def __enter__(self) -> "FrameSource":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def _open(self) -> None: ...
    def _close(self) -> None: ...
    def _run(self) -> None: ...


class FfmpegAvfoundationSource(FrameSource):
    """ffmpeg avfoundation pipe로 BGR24 raw 프레임을 받는 공통 소스.

    macOS는 Continuity Camera 전환 등 카메라 중재 이벤트로, 장시간 열려 있던
    세션의 피드를 조용히 검정 프레임으로 바꾸거나(ffmpeg 프로세스는 살아있음)
    ffmpeg를 종료시키곤 한다. 이를 대비해 워치독 스레드가 프레임 정체를
    감지하면 ffmpeg를 죽이고 새로 열어 자가복구한다.
    """

    STALE_TIMEOUT = 3.0       # 마지막 유효 프레임 이후 이 시간(s)을 넘기면 재연결
    WATCHDOG_PERIOD = 1.0     # 워치독 점검 주기(s)

    def __init__(self, name_substring: str, width: int, height: int, fps: int,
                 debug_ffmpeg: bool = False) -> None:
        super().__init__()
        self.name_substring = name_substring
        self.width = width
        self.height = height
        self.fps = fps
        self.debug_ffmpeg = debug_ffmpeg
        self.resolved_index: Optional[int] = None
        self.resolved_name: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None
        self._last_accept = 0.0          # 마지막 유효 프레임 수신 시각(monotonic)
        self._reopen_lock = threading.Lock()
        self._watchdog: Optional[threading.Thread] = None

    def _resolve_device(self) -> None:
        """디바이스 이름을 매번 새로 해석 — Continuity 등으로 바뀔 수 있으므로."""
        devices = list_avfoundation_video_devices()
        self.resolved_index = None
        self.resolved_name = None
        for idx, name in devices:
            if self.name_substring.lower() in name.lower():
                self.resolved_index = idx
                self.resolved_name = name
                break
        if self.resolved_name is None:
            available = ", ".join(f"[{i}] {n}" for i, n in devices)
            raise RuntimeError(
                f"'{self.name_substring}' 디바이스를 찾지 못함. 사용 가능: {available}"
            )

    def _spawn_proc(self) -> None:
        # 핵심: ffmpeg `-list_devices` 인덱스와 실제 `-i <idx>` 인덱스가
        # macOS Continuity Camera 등으로 동적으로 어긋날 수 있다.
        # ffmpeg는 -i에 정확한 디바이스 이름을 직접 받으므로 그쪽이 견고하다.
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "avfoundation",
            "-framerate", str(self.fps),
            "-video_size", f"{self.width}x{self.height}",
            "-i", self.resolved_name,
            # 출력 fps 고정. avfoundation이 입력 -framerate를 무시하고 동일 프레임을
            # 수백 fps로 복제·폭주시키는 경우가 있어(FaceTime HD에서 533fps 관찰됨)
            # 출력단에서 한 번 더 못박는다.
            "-r", str(self.fps),
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-",
        ]
        stderr = None if self.debug_ffmpeg else subprocess.DEVNULL
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=stderr, bufsize=10 ** 8,
        )

    def _open(self) -> None:
        self._resolve_device()
        self._spawn_proc()
        self._last_accept = time.monotonic()

    def start(self) -> "FrameSource":
        super().start()  # _open() + _run 스레드 기동
        self._watchdog = threading.Thread(target=self._watch, daemon=True)
        self._watchdog.start()
        return self

    def _kill_proc(self) -> None:
        proc = self._proc
        self._proc = None  # _run이 재연결 도중 죽은 stdout을 읽지 않도록 먼저 비운다
        if proc is not None:
            proc.kill()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass

    def _reopen(self, reason: str) -> None:
        """ffmpeg 세션을 죽이고 새로 연다. 동시 호출은 락으로 직렬화."""
        if self._stop.is_set():
            return
        if not self._reopen_lock.acquire(blocking=False):
            return  # 이미 다른 스레드가 재연결 중
        try:
            print(f"[{type(self).__name__}] 재연결: {reason}", flush=True)
            self._kill_proc()
            self._resolve_device()
            self._spawn_proc()
            self._last_accept = time.monotonic()
            print(f"[{type(self).__name__}] 재연결 완료 → {self.resolved_name}", flush=True)
        except Exception as e:  # 디바이스 일시적 부재 등 — 다음 주기에 재시도
            print(f"[{type(self).__name__}] 재연결 실패: {e}", flush=True)
            time.sleep(1.0)
        finally:
            self._reopen_lock.release()

    def _run(self) -> None:
        frame_bytes = self.width * self.height * 3
        while not self._stop.is_set():
            proc = self._proc
            stdout = proc.stdout if proc is not None else None
            if stdout is None:
                time.sleep(0.05)  # 재연결 도중 — 잠깐 대기
                continue
            raw = stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                # ffmpeg 종료/짧은 read → 즉시 재연결 (워치독 staleness 기다리지 않음)
                self._reopen("ffmpeg 종료 또는 짧은 read")
                time.sleep(0.05)
                continue
            image = (
                np.frombuffer(raw, dtype=np.uint8)
                .reshape(self.height, self.width, 3)
                .copy()
            )
            if image.mean() < 1.0:
                continue  # 검정 프레임 스킵 — 지속되면 워치독이 staleness로 재연결
            with self._lock:
                self._latest = Frame(image=image, timestamp=time.monotonic())
            self._last_accept = time.monotonic()

    def _watch(self) -> None:
        """프레임이 STALE_TIMEOUT 넘게 정체되면 재연결(검정 프레임/피드 stall 대비)."""
        while not self._stop.is_set():
            time.sleep(self.WATCHDOG_PERIOD)
            if self._proc is None:
                continue
            if time.monotonic() - self._last_accept > self.STALE_TIMEOUT:
                self._reopen(f"{self.STALE_TIMEOUT:.0f}s 이상 프레임 정체")

    def _close(self) -> None:
        self._kill_proc()


class ThermalSource(FfmpegAvfoundationSource):
    """Cellplus (Obsidian Sensors) 열화상. 640x480@30fps 단일 모드."""

    def __init__(self, name_substring: str = "Camera Contol I/F") -> None:
        super().__init__(name_substring, width=640, height=480, fps=30)


class RGBSource(FfmpegAvfoundationSource):
    """MacBook 내장 RGB 카메라 (FaceTime HD)."""

    def __init__(
        self,
        name_substring: str = "FaceTime",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        super().__init__(name_substring, width=width, height=height, fps=fps)
