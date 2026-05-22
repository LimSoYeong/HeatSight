"""Cellplus (CTC-VGA-USB) 시리얼 컨트롤.

USB CDC ACM 시리얼 포트를 통해 카메라 레지스터를 read/write.
프로토콜은 cellplus-korea/CTC-VGA-USB 저장소의 SerialWorker 구현 참조.
"""

from __future__ import annotations

import glob
import struct
from typing import Optional

import serial


# 레지스터 주소 (RegisterMap.h)
M_FPGA_MAJOR_VER_ADDR  = 0x00001010
M_COLORMAP_IDX_ADDR    = 0x0000200A   # 1 byte: 컬러맵 인덱스
F_COLORMAP_COUNT_ADDR  = 0x609C0000   # 4 byte: 지원되는 컬러맵 수

# 카메라가 출력하는 영상 해상도 (좌표 변환의 기준)
IMAGE_WIDTH  = 640
IMAGE_HEIGHT = 480

# 카메라의 native 온도 좌표계 ↔ 화면 영상 좌표계 사이 관계.
# 실측 결과: 좌우 반전 (flip horizontal).
def _native_to_display(x: int, y: int) -> tuple[int, int]:
    return (IMAGE_WIDTH - 1 - x, y)

def _display_to_native(x: int, y: int) -> tuple[int, int]:
    return (IMAGE_WIDTH - 1 - x, y)


# 온도 측정 (RegisterMap.h T_* 그룹)
T_COORD_ADDR       = 0x2D000008   # ROI 좌표 write: (Y<<16)|X
T_DATA_ADDR        = 0x2D00002C   # 현재 ROI의 raw int16 온도값
T_MINMAX_FLAG_ADDR = 0x2D000000   # min/max 부호 플래그
T_MINMAX_DATA_ADDR = 0x2D000054   # min/max raw 값 (LE: min[0:2], max[2:4])
T_MIN_COORD_ADDR   = 0x2D000058   # min 좌표 (LE: Y[0:2], X[2:4])
T_MAX_COORD_ADDR   = 0x2D00005C   # max 좌표 (LE: Y[0:2], X[2:4])

# 프로토콜 상수 (SerialWorker.h)
_CMD_READ_HEADER  = bytes([0x00, 0x40, 0x00, 0x08, 0x0C, 0x00])
_CMD_WRITE_HEADER = bytes([0x00, 0x40, 0x02, 0x08])
_RESP_READ_TYPE   = bytes([0x01, 0x08])
_RESP_WRITE_TYPE  = bytes([0x03, 0x08])
_RESP_BODY_OFFSET = 8
_WRITE_RESP_LEN   = 12
_WRITE_RESP_WRITTEN_OFFSET = 10   # 헤더 파일의 kWriteRespWrittenOffset 기준


# 24개 컬러맵 중 사용할 프리셋 (HeatSight 서비스 화면이 노출하는 후보)
COLORMAP_PRESETS: list[int] = [0, 6, 13, 16, 17]


def next_preset(current: int, delta: int) -> int:
    """현재 인덱스에서 ±delta만큼 프리셋 안에서 순환한 다음 인덱스 반환."""
    if not COLORMAP_PRESETS:
        return current
    if current in COLORMAP_PRESETS:
        pos = COLORMAP_PRESETS.index(current)
    else:
        pos = 0
    return COLORMAP_PRESETS[(pos + delta) % len(COLORMAP_PRESETS)]


class CellplusControl:
    """CTC-VGA-USB 시리얼 컨트롤 채널 클라이언트."""

    BAUDRATE = 115200

    def __init__(self, port_path: str) -> None:
        self.port_path = port_path
        self.ser: Optional[serial.Serial] = None
        self._req_id = 0

    def open(self) -> "CellplusControl":
        self.ser = serial.Serial(
            port=self.port_path, baudrate=self.BAUDRATE,
            bytesize=8, parity="N", stopbits=1,
            rtscts=True, timeout=2.0,
        )
        return self

    def close(self) -> None:
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def __enter__(self) -> "CellplusControl":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def _next_rid(self) -> int:
        self._req_id = (self._req_id + 1) & 0xFFFF
        return self._req_id

    def read_mem(self, address: int, length: int) -> bytes:
        assert self.ser is not None
        packet = (
            _CMD_READ_HEADER
            + struct.pack("<H", self._next_rid())
            + address.to_bytes(8, "little")
            + bytes([0x00, 0x00])
            + struct.pack("<H", length)
        )
        self.ser.reset_input_buffer()
        self.ser.write(packet)
        total = _RESP_BODY_OFFSET + length
        resp = self.ser.read(total)
        if len(resp) != total:
            raise RuntimeError(
                f"READMEM 타임아웃 ({address:#010x}): {len(resp)}/{total} bytes"
            )
        if resp[2:4] != _RESP_READ_TYPE:
            raise RuntimeError(
                f"READMEM type 불일치 ({address:#010x}): header={resp[:4].hex()}"
            )
        return resp[_RESP_BODY_OFFSET:]

    def write_mem(self, address: int, length: int, value: int) -> bool:
        assert self.ser is not None
        packet = (
            _CMD_WRITE_HEADER
            + struct.pack("<H", 8 + length)
            + struct.pack("<H", self._next_rid())
            + address.to_bytes(8, "little")
            + value.to_bytes(length, "little")
        )
        self.ser.reset_input_buffer()
        self.ser.write(packet)
        resp = self.ser.read(_WRITE_RESP_LEN)
        if len(resp) != _WRITE_RESP_LEN:
            raise RuntimeError(
                f"WRITEMEM 응답 짧음 ({address:#010x}): {len(resp)}/{_WRITE_RESP_LEN}"
            )
        if resp[2:4] != _RESP_WRITE_TYPE:
            raise RuntimeError(
                f"WRITEMEM type 불일치 ({address:#010x}): header={resp[:4].hex()}"
            )
        written = int.from_bytes(
            resp[_WRITE_RESP_WRITTEN_OFFSET:_WRITE_RESP_WRITTEN_OFFSET + 2], "little"
        )
        return written == length

    # 고수준 API

    def get_fpga_major(self) -> int:
        return self.read_mem(M_FPGA_MAJOR_VER_ADDR, 1)[0]

    def get_colormap_count(self) -> int:
        return int.from_bytes(self.read_mem(F_COLORMAP_COUNT_ADDR, 4)[:4], "little")

    def get_colormap(self) -> int:
        return self.read_mem(M_COLORMAP_IDX_ADDR, 1)[0]

    def set_colormap(self, index: int) -> bool:
        return self.write_mem(M_COLORMAP_IDX_ADDR, 1, index)

    # --- 온도 측정 ---

    def set_temperature_coord(self, x: int, y: int) -> bool:
        """ROI 좌표를 카메라에 설정.

        인자 (x, y)는 화면(display) 좌표계 — 사용자가 보는 영상의 좌표.
        내부에서 카메라 native 좌표계로 변환(좌우 반전)한 뒤 (Y<<16)|X로 패킹해 write.
        """
        nx, ny = _display_to_native(int(x), int(y))
        nx = max(0, min(0xFFFF, nx))
        ny = max(0, min(0xFFFF, ny))
        packed = (ny << 16) | nx
        return self.write_mem(T_COORD_ADDR, 4, packed)

    def get_temperature_pixel_raw(self) -> int:
        """현재 ROI 픽셀의 raw int16 온도값."""
        data = self.read_mem(T_DATA_ADDR, 4)
        return int.from_bytes(data[:2], "little", signed=True)

    def measure_pixel_raw(self, x: int, y: int, settle_s: float = 0.02) -> int:
        """좌표를 설정하고 raw 온도값을 한 번에 읽는 편의 함수."""
        self.set_temperature_coord(x, y)
        if settle_s > 0:
            import time
            time.sleep(settle_s)
        return self.get_temperature_pixel_raw()

    def get_minmax(self) -> dict:
        """프레임 전체 min/max raw 값과 좌표를 화면 좌표계로 반환."""
        flag_bytes = self.read_mem(T_MINMAX_FLAG_ADDR, 4)
        minmax = self.read_mem(T_MINMAX_DATA_ADDR, 4)
        min_coord = self.read_mem(T_MIN_COORD_ADDR, 4)
        max_coord = self.read_mem(T_MAX_COORD_ADDR, 4)

        n_min_y = int.from_bytes(min_coord[0:2], "little")
        n_min_x = int.from_bytes(min_coord[2:4], "little")
        n_max_y = int.from_bytes(max_coord[0:2], "little")
        n_max_x = int.from_bytes(max_coord[2:4], "little")

        min_x, min_y = _native_to_display(n_min_x, n_min_y)
        max_x, max_y = _native_to_display(n_max_x, n_max_y)

        return {
            "min_raw": int.from_bytes(minmax[0:2], "little", signed=True),
            "max_raw": int.from_bytes(minmax[2:4], "little", signed=True),
            "min_x":   min_x,
            "min_y":   min_y,
            "max_x":   max_x,
            "max_y":   max_y,
            "flag":    flag_bytes[0],
        }


def find_control_port() -> Optional[str]:
    """카메라의 두 시리얼 포트 중 컨트롤 응답을 주는 쪽을 찾는다.

    각 후보 포트를 짧게 열고 FPGA Major 버전 read를 시도해 정상 응답하는 포트 반환.
    """
    candidates = sorted(glob.glob("/dev/cu.usbmodemMirVG*"))
    for path in candidates:
        try:
            with CellplusControl(path) as ctl:
                _ = ctl.get_fpga_major()
                return path
        except Exception:
            continue
    return None
