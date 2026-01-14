from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ..core.arch import Architecture
from ..core.memory import MemoryMap, MemoryProvider
from ..core.state import RegisterFile


@dataclass
class Snapshot:
    arch: Architecture
    platform: str
    metadata: Dict[str, Any]
    registers: RegisterFile
    memory_map: MemoryMap
    modules: List[Dict[str, Any]]
    thread: Dict[str, Any]
    memory: MemoryProvider

    def __repr__(self) -> str:
        return (
            f"Snapshot(arch={self.arch.value}, regions={len(self.memory_map)}, "
            f"modules={len(self.modules)})"
        )

    def emulator(self, backend: str = "unicorn", config=None):
        from ..backends import MaatEmulator, TritonEmulator, UnicornEmulator
        from ..backends.base import EmulatorConfig

        backend_key = backend.lower()
        config = config or EmulatorConfig()
        if backend_key == "maat" and config.map_zero_page is None:
            config.map_zero_page = True

        if backend_key == "unicorn":
            return UnicornEmulator(self, config)
        if backend_key == "triton":
            return TritonEmulator(self, config)
        if backend_key == "maat":
            return MaatEmulator(self, config)
        raise ValueError(f"unsupported backend: {backend}")
