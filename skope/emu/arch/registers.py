#!/usr/bin/env python3
"""Architecture register metadata and helpers."""

from __future__ import annotations

from typing import Dict, List

from ..base import Arch


# Architecture-specific register groups
REGISTER_NAMES: Dict[Arch, Dict[str, List[str]]] = {
    Arch.ARM64: {
        "gp": [f"x{i}" for i in range(31)],  # x0-x30
        "special": ["sp", "pc", "xzr", "wzr", "nzcv"],
        "flags": ["n", "z", "c", "v"],
        "all": [f"x{i}" for i in range(31)]
        + ["sp", "pc", "xzr", "wzr", "nzcv", "n", "z", "c", "v"],
    },
    Arch.X64: {
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
    Arch.X86: {
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


def get_register_names(arch: Arch, category: str = "all") -> List[str]:
    """Return register names for the requested architecture and category."""

    return REGISTER_NAMES.get(arch, {}).get(category, [])


def get_pc_register(arch: Arch) -> str:
    """Return the canonical program-counter register name."""

    if arch == Arch.ARM64:
        return "pc"
    if arch == Arch.X64:
        return "rip"
    if arch == Arch.X86:
        return "eip"
    raise ValueError(f"unknown architecture: {arch}")


def get_sp_register(arch: Arch) -> str:
    """Return the canonical stack-pointer register name."""

    if arch == Arch.ARM64:
        return "sp"
    if arch == Arch.X64:
        return "rsp"
    if arch == Arch.X86:
        return "esp"
    raise ValueError(f"unknown architecture: {arch}")


def get_return_register(arch: Arch) -> str:
    """Return the register used for function return values."""

    if arch == Arch.ARM64:
        return "x0"
    if arch == Arch.X64:
        return "rax"
    if arch == Arch.X86:
        return "eax"
    raise ValueError(f"unknown architecture: {arch}")


def get_word_size(arch: Arch) -> int:
    """Return the natural word size in bytes for the architecture."""

    return arch.pointer_size


def get_stack_alignment(arch: Arch) -> int:
    """Return ABI-required stack alignment in bytes."""

    if arch in (Arch.ARM64, Arch.X64):
        return 16
    if arch == Arch.X86:
        return 4
    raise ValueError(f"unknown architecture: {arch}")


# Conventional stack bases for bare-metal setups.
DEFAULT_STACK_BASE: Dict[Arch, int] = {
    Arch.ARM64: 0x300000000,
    Arch.X64: 0x300000000,
    Arch.X86: 0x30000000,
}


__all__ = [
    "REGISTER_NAMES",
    "DEFAULT_STACK_BASE",
    "get_register_names",
    "get_pc_register",
    "get_sp_register",
    "get_return_register",
    "get_word_size",
    "get_stack_alignment",
]
