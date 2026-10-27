"""Print GPU free memory and Windows commit headroom. Does not load a model."""

import ctypes
from ctypes import wintypes

import torch


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


stat = MEMORYSTATUSEX()
stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
print(
    f"phys_free_mb {stat.ullAvailPhys // (1024 * 1024)} "
    f"commit_free_mb {stat.ullAvailPageFile // (1024 * 1024)}"
)
for index in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(index)
    print(f"cuda:{index} free_mib {free // (1024 * 1024)} total_mib {total // (1024 * 1024)}")
