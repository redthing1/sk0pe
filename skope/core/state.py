from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from .arch import Architecture


_ARM64_ALIASES = {
    "lr": "x30",
    "fp": "x29",
    "cpsr": "nzcv",
}

_X64_ALIASES = {
    "eflags": "rflags",
}

_X86_ALIASES = {
    "rflags": "eflags",
}


class RegisterFile:
    """Architecture-aware register map with alias normalization."""

    def __init__(self, arch: Architecture, registers: Optional[Dict[str, int]] = None):
        self.arch = arch
        self._values: Dict[str, int] = {}
        if registers:
            for name, value in registers.items():
                self.set(name, value)

    def _normalize(self, name: str) -> str:
        normalized = name.lower()
        if normalized == "pc":
            return self.arch.pc_name
        if normalized == "sp":
            return self.arch.sp_name
        if self.arch == Architecture.ARM64:
            return _ARM64_ALIASES.get(normalized, normalized)
        if self.arch == Architecture.X64:
            return _X64_ALIASES.get(normalized, normalized)
        if self.arch == Architecture.X86:
            return _X86_ALIASES.get(normalized, normalized)
        return normalized

    def set(self, name: str, value: int) -> None:
        self._values[self._normalize(name)] = int(value)

    def get(self, name: str, default: Optional[int] = None) -> int:
        key = self._normalize(name)
        if default is None:
            return self._values.get(key, 0)
        return self._values.get(key, default)

    def has(self, name: str) -> bool:
        return self._normalize(name) in self._values

    def as_dict(self) -> Dict[str, int]:
        return dict(self._values)

    def update(self, values: Dict[str, int]) -> None:
        for name, value in values.items():
            self.set(name, value)


@dataclass
class MachineState:
    """Snapshot of register state with cached PC/SP values."""

    registers: RegisterFile
    pc: Optional[int] = None
    sp: Optional[int] = None

    def __post_init__(self) -> None:
        if self.pc is None:
            self.pc = self.registers.get(self.registers.arch.pc_name)
        if self.sp is None:
            self.sp = self.registers.get(self.registers.arch.sp_name)
