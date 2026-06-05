"""libtemperature_calculator.so ctypes wrapper.

C API (TemperatureCalculator.h):
    TempCalcHandle CTC_Create(void);
    void           CTC_Destroy(TempCalcHandle h);
    int            CTC_SetParameter(TempCalcHandle h, const float* params, int count);
    int            CTC_ComputeTemperature(TempCalcHandle h,
                                          int16_t raw, float boardTempC,
                                          float radiometricGain, float pwmTarget,
                                          double* out);
"""
from __future__ import annotations

import ctypes
import threading
from pathlib import Path
from typing import Sequence

_LIB_PATH = Path(__file__).resolve().parent / "lib" / "libtemperature_calculator.so"

_lib = ctypes.CDLL(str(_LIB_PATH))

# 시그니처 등록
_lib.CTC_Create.restype = ctypes.c_void_p
_lib.CTC_Create.argtypes = []

_lib.CTC_Destroy.restype = None
_lib.CTC_Destroy.argtypes = [ctypes.c_void_p]

_lib.CTC_SetParameter.restype = ctypes.c_int
_lib.CTC_SetParameter.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]

_lib.CTC_ComputeTemperature.restype = ctypes.c_int
_lib.CTC_ComputeTemperature.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int16,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_double),
]


class TemperatureCalculator:
    """C API를 감싼 thread-safe wrapper. 한 프로세스에 하나만."""

    def __init__(self) -> None:
        self._handle = _lib.CTC_Create()
        if not self._handle:
            raise RuntimeError("CTC_Create 실패")
        self._lock = threading.Lock()
        self._param_set = False

    def __del__(self) -> None:
        if getattr(self, "_handle", None):
            _lib.CTC_Destroy(self._handle)
            self._handle = None

    def set_parameters(self, params: Sequence[float]) -> None:
        if len(params) < 5:
            raise ValueError(f"params는 최소 5개 필요, 받은 개수: {len(params)}")
        arr = (ctypes.c_float * len(params))(*[float(p) for p in params])
        with self._lock:
            rc = _lib.CTC_SetParameter(self._handle, arr, len(params))
            if rc != 0:
                raise RuntimeError(f"CTC_SetParameter 실패: rc={rc}")
            self._param_set = True

    def compute(self, raw: int, board_temp_c: float,
                radiometric_gain: float, pwm_target: float) -> float:
        if not self._param_set:
            raise RuntimeError("set_parameters를 먼저 호출해야 함")
        out = ctypes.c_double(0.0)
        with self._lock:
            rc = _lib.CTC_ComputeTemperature(
                self._handle,
                ctypes.c_int16(int(raw)),
                ctypes.c_float(float(board_temp_c)),
                ctypes.c_float(float(radiometric_gain)),
                ctypes.c_float(float(pwm_target)),
                ctypes.byref(out),
            )
            if rc != 0:
                raise RuntimeError(f"CTC_ComputeTemperature 실패: rc={rc}")
        return float(out.value)
