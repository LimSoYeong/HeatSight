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
    """ffmpeg avfoundation pipe로 BGR24 raw 프레임을 받는 공통 소스."""

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

    def _open(self) -> None:
        devices = list_avfoundation_video_devices()
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

        # 핵심: ffmpeg `-list_devices` 인덱스와 실제 `-i <idx>` 인덱스가
        # macOS Continuity Camera 등으로 동적으로 어긋날 수 있다.
        # ffmpeg는 -i에 정확한 디바이스 이름을 직접 받으므로 그쪽이 견고하다.
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "avfoundation",
            "-framerate", str(self.fps),
            "-video_size", f"{self.width}x{self.height}",
            "-i", self.resolved_name,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-",
        ]
        stderr = None if self.debug_ffmpeg else subprocess.DEVNULL
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=stderr, bufsize=10 ** 8,
        )

    def _run(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        frame_bytes = self.width * self.height * 3
        while not self._stop.is_set():
            raw = self._proc.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                break
            image = (
                np.frombuffer(raw, dtype=np.uint8)
                .reshape(self.height, self.width, 3)
                .copy()
            )
            if image.mean() < 1.0:
                continue
            with self._lock:
                self._latest = Frame(image=image, timestamp=time.monotonic())

    def _close(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass


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
