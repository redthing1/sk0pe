from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import msgpack

from ..core.arch import Architecture, parse_arch
from ..core.memory import MemoryMap, MemoryProvider, MemoryRegion
from ..core.permissions import MemoryPermissions
from ..core.state import RegisterFile
from .base import Snapshot


class SnapshotMemoryProvider:
    """Memory provider backed by snapshot regions."""

    def __init__(self, memory_map: MemoryMap):
        self._memory_map = memory_map

    def get_region(self, address: int) -> Optional[MemoryRegion]:
        return self._memory_map.find(address)

    def read(self, address: int, size: int) -> Optional[bytes]:
        region = self.get_region(address)
        if not region or region.data is None:
            return None
        if not region.contains(address, size):
            return None
        offset = address - region.start
        if offset + size > len(region.data):
            return None
        return region.data[offset : offset + size]


class W1DumpSnapshot:
    """Loader for w1dump snapshot files."""

    @staticmethod
    def load(path: str) -> Snapshot:
        dump_path = Path(path)
        if not dump_path.exists():
            raise FileNotFoundError(f"w1dump file not found: {dump_path}")

        data = dump_path.read_bytes()
        if not data:
            raise ValueError("w1dump file is empty")

        try:
            payload = msgpack.unpackb(data, raw=False, strict_map_key=False)
        except msgpack.exceptions.ExtraData as exc:
            raise ValueError(f"w1dump contains extra data: {exc}") from exc
        except Exception as exc:
            raise ValueError(f"w1dump parse error: {exc}") from exc

        required_keys = {"metadata", "thread", "regions", "modules"}
        missing = required_keys - set(payload.keys())
        if missing:
            raise ValueError(f"w1dump missing required keys: {sorted(missing)}")

        metadata = dict(payload["metadata"])
        thread = dict(payload["thread"])
        modules = list(payload.get("modules", []))

        arch = parse_arch(str(metadata.get("arch", "")))
        platform = str(metadata.get("os", ""))

        registers = RegisterFile(arch)
        gpr_values = list(thread.get("gpr_values", []))
        registers.update(_parse_gpr_values(arch, gpr_values))

        regions = _parse_regions(payload.get("regions", []))
        memory_map = MemoryMap(regions)
        memory = SnapshotMemoryProvider(memory_map)

        return Snapshot(
            arch=arch,
            platform=platform,
            metadata=metadata,
            registers=registers,
            memory_map=memory_map,
            modules=modules,
            thread=thread,
            memory=memory,
        )


def _parse_regions(raw_regions: Iterable[Dict[str, Any]]) -> List[MemoryRegion]:
    regions: List[MemoryRegion] = []
    for entry in raw_regions:
        data = entry.get("data")
        if data is not None:
            data = bytes(data)
        permissions = MemoryPermissions(entry.get("permissions", 0))
        region = MemoryRegion(
            start=int(entry["start"]),
            end=int(entry["end"]),
            permissions=permissions,
            module_name=str(entry.get("module_name", "")),
            data=data,
        )
        regions.append(region)
    return regions


def _parse_gpr_values(arch: Architecture, values: List[int]) -> Dict[str, int]:
    if arch == Architecture.X64:
        order = [
            "rax",
            "rbx",
            "rcx",
            "rdx",
            "rsi",
            "rdi",
            "r8",
            "r9",
            "r10",
            "r11",
            "r12",
            "r13",
            "r14",
            "r15",
            "rbp",
            "rsp",
            "rip",
            "rflags",
            "fs",
            "gs",
        ]
        if len(values) < 18:
            raise ValueError(f"expected at least 18 x64 GPR values, got {len(values)}")
        return {
            name: int(values[idx])
            for idx, name in enumerate(order)
            if idx < len(values)
        }

    if arch == Architecture.X86:
        order = [
            "eax",
            "ebx",
            "ecx",
            "edx",
            "esi",
            "edi",
            "ebp",
            "esp",
            "eip",
            "eflags",
        ]
        if len(values) < 10:
            raise ValueError(f"expected at least 10 x86 GPR values, got {len(values)}")
        return {
            name: int(values[idx])
            for idx, name in enumerate(order)
            if idx < len(values)
        }

    if arch == Architecture.ARM64:
        order = [f"x{i}" for i in range(29)] + ["x29", "lr", "sp", "nzcv", "pc"]
        if len(values) < 34:
            raise ValueError(
                f"expected at least 34 arm64 GPR values, got {len(values)}"
            )
        return {
            name: int(values[idx])
            for idx, name in enumerate(order)
            if idx < len(values)
        }

    raise ValueError(f"unsupported architecture for GPR parsing: {arch}")
