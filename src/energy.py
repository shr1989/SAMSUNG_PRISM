import ctypes
import os
import subprocess
import time
from dataclasses import dataclass

_FIELDS = [
    "user_time",
    "system_time",
    "pkg_idle_wkups",
    "interrupt_wkups",
    "pageins",
    "wired_size",
    "resident_size",
    "phys_footprint",
    "proc_start_abstime",
    "proc_exit_abstime",
    "child_user_time",
    "child_system_time",
    "child_pkg_idle_wkups",
    "child_interrupt_wkups",
    "child_pageins",
    "child_elapsed_abstime",
    "diskio_bytesread",
    "diskio_byteswritten",
    "cpu_time_qos_default",
    "cpu_time_qos_maintenance",
    "cpu_time_qos_background",
    "cpu_time_qos_utility",
    "cpu_time_qos_legacy",
    "cpu_time_qos_user_initiated",
    "cpu_time_qos_user_interactive",
    "billed_system_time",
    "serviced_system_time",
    "logical_writes",
    "lifetime_max_phys_footprint",
    "instructions",
    "cycles",
    "billed_energy",
    "serviced_energy",
    "interval_max_phys_footprint",
    "runnable_time",
    "flags",
    "user_ptime",
    "system_ptime",
    "pinstructions",
    "pcycles",
    "energy_nj",
    "penergy_nj",
    "secure_time_in_system",
    "secure_ptime_in_system",
    "neural_footprint",
    "lifetime_max_neural_footprint",
    "interval_max_neural_footprint",
]
RUSAGE_INFO_V6 = 6


class _RusageInfoV6(ctypes.Structure):
    _fields_ = (
        [("uuid", ctypes.c_uint8 * 16)]
        + [(name, ctypes.c_uint64) for name in _FIELDS]
        + [("reserved", ctypes.c_uint64 * 9)]
    )


_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")


def process_energy_joules(pid: int | None = None) -> float:
    """CPU energy the macOS kernel attributes to a process (rusage_info_v6.ri_energy_nj)."""
    info = _RusageInfoV6()
    if _libproc.proc_pid_rusage(pid or os.getpid(), RUSAGE_INFO_V6, ctypes.byref(info)) != 0:
        raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
    return info.energy_nj / 1e9


@dataclass
class EnergyMeter:
    pid: int | None = None

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.start_joules = process_energy_joules(self.pid)
        self.start_time = time.perf_counter()

    def average_watts(self) -> float:
        elapsed = time.perf_counter() - self.start_time
        return (process_energy_joules(self.pid) - self.start_joules) / max(elapsed, 1e-9)


def thermal_state() -> str:
    try:
        output = subprocess.run(
            ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    warnings = [
        line.strip() for line in output.splitlines() if "level" in line and "No " not in line
    ]
    return "; ".join(warnings) or "no thermal warning"
