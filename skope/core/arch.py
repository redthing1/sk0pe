from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from .errors import ArchitectureError


class Architecture(Enum):
    X86 = "x86"
    X64 = "x64"
    ARM64 = "arm64"

    @property
    def bits(self) -> int:
        return 64 if self in (Architecture.X64, Architecture.ARM64) else 32

    @property
    def pointer_size(self) -> int:
        return self.bits // 8

    @property
    def pc_name(self) -> str:
        if self == Architecture.X86:
            return "eip"
        if self == Architecture.X64:
            return "rip"
        if self == Architecture.ARM64:
            return "pc"
        raise ArchitectureError(f"unknown architecture: {self}")

    @property
    def sp_name(self) -> str:
        if self == Architecture.X86:
            return "esp"
        if self == Architecture.X64:
            return "rsp"
        if self == Architecture.ARM64:
            return "sp"
        raise ArchitectureError(f"unknown architecture: {self}")

    @property
    def frame_pointer_name(self) -> Optional[str]:
        if self == Architecture.X86:
            return "ebp"
        if self == Architecture.X64:
            return "rbp"
        if self == Architecture.ARM64:
            return "x29"
        return None

    @property
    def return_name(self) -> str:
        if self == Architecture.X86:
            return "eax"
        if self == Architecture.X64:
            return "rax"
        if self == Architecture.ARM64:
            return "x0"
        raise ArchitectureError(f"unknown architecture: {self}")

    @property
    def stack_alignment(self) -> int:
        if self in (Architecture.X64, Architecture.ARM64):
            return 16
        if self == Architecture.X86:
            return 4
        raise ArchitectureError(f"unknown architecture: {self}")

    def register_names(self, category: str = "all") -> List[str]:
        return list(REGISTER_NAMES.get(self, {}).get(category, []))


REGISTER_NAMES: Dict[Architecture, Dict[str, List[str]]] = {
    Architecture.ARM64: {
        "gp": [f"x{i}" for i in range(31)],
        "special": ["sp", "pc", "nzcv"],
        "flags": ["n", "z", "c", "v"],
        "all": [f"x{i}" for i in range(31)] + ["sp", "pc", "nzcv"],
    },
    Architecture.X64: {
        "gp": [
            "rax",
            "rbx",
            "rcx",
            "rdx",
            "rsi",
            "rdi",
            "rbp",
            "rsp",
            "r8",
            "r9",
            "r10",
            "r11",
            "r12",
            "r13",
            "r14",
            "r15",
        ],
        "special": ["rip", "rflags"],
        "flags": ["cf", "pf", "af", "zf", "sf", "of", "df"],
        "all": [
            "rax",
            "rbx",
            "rcx",
            "rdx",
            "rsi",
            "rdi",
            "rbp",
            "rsp",
            "r8",
            "r9",
            "r10",
            "r11",
            "r12",
            "r13",
            "r14",
            "r15",
            "rip",
            "rflags",
        ],
    },
    Architecture.X86: {
        "gp": ["eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp"],
        "special": ["eip", "eflags"],
        "flags": ["cf", "pf", "af", "zf", "sf", "of", "df"],
        "all": [
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
        ],
    },
}


_ARCH_ALIASES = {
    "x86": Architecture.X86,
    "i386": Architecture.X86,
    "x64": Architecture.X64,
    "x86_64": Architecture.X64,
    "amd64": Architecture.X64,
    "arm64": Architecture.ARM64,
    "aarch64": Architecture.ARM64,
}


def parse_arch(value: str) -> Architecture:
    """Parse an architecture string into an Architecture enum."""

    key = value.strip().lower()
    arch = _ARCH_ALIASES.get(key)
    if not arch:
        raise ArchitectureError(f"unsupported architecture: {value}")
    return arch
