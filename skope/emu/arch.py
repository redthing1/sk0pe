#!/usr/bin/env python3
"""
architecture-specific constants and utilities
"""

from typing import Dict, List, Optional, Any
from .base import Arch


# register names for each architecture
REGISTER_NAMES = {
    Arch.ARM64: {
        "gp": [f"x{i}" for i in range(31)],  # x0-x30
        "special": ["sp", "pc", "xzr", "wzr"],
        "flags": ["n", "z", "c", "v"],  # condition flags
        "all": [f"x{i}" for i in range(31)]
        + ["sp", "pc", "xzr", "wzr", "n", "z", "c", "v"],
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
    """get register names for an architecture"""
    return REGISTER_NAMES.get(arch, {}).get(category, [])


def get_pc_register(arch: Arch) -> str:
    """get the program counter register name"""
    if arch == Arch.ARM64:
        return "pc"
    elif arch == Arch.X64:
        return "rip"
    elif arch == Arch.X86:
        return "eip"
    else:
        raise ValueError(f"unknown architecture: {arch}")


def get_sp_register(arch: Arch) -> str:
    """get the stack pointer register name"""
    if arch == Arch.ARM64:
        return "sp"
    elif arch == Arch.X64:
        return "rsp"
    elif arch == Arch.X86:
        return "esp"
    else:
        raise ValueError(f"unknown architecture: {arch}")


def decode_arm64_cpsr(cpsr: int) -> Dict[str, bool]:
    """decode arm64 cpsr/pstate into individual flags"""
    return {
        "n": bool(cpsr & (1 << 31)),  # negative
        "z": bool(cpsr & (1 << 30)),  # zero
        "c": bool(cpsr & (1 << 29)),  # carry
        "v": bool(cpsr & (1 << 28)),  # overflow
        # other flags can be added as needed
    }


def encode_arm64_cpsr(flags: Dict[str, bool]) -> int:
    """encode individual flags into arm64 cpsr/pstate"""
    cpsr = 0
    if flags.get("n", False):
        cpsr |= 1 << 31
    if flags.get("z", False):
        cpsr |= 1 << 30
    if flags.get("c", False):
        cpsr |= 1 << 29
    if flags.get("v", False):
        cpsr |= 1 << 28
    return cpsr


def decode_x86_flags(eflags: int) -> Dict[str, bool]:
    """decode x86/x64 eflags into individual flags"""
    return {
        "cf": bool(eflags & (1 << 0)),  # carry
        "pf": bool(eflags & (1 << 2)),  # parity
        "af": bool(eflags & (1 << 4)),  # auxiliary carry
        "zf": bool(eflags & (1 << 6)),  # zero
        "sf": bool(eflags & (1 << 7)),  # sign
        "of": bool(eflags & (1 << 11)),  # overflow
        "df": bool(eflags & (1 << 10)),  # direction
    }


def encode_x86_flags(flags: Dict[str, bool]) -> int:
    """encode individual flags into x86/x64 eflags"""
    eflags = 0
    if flags.get("cf", False):
        eflags |= 1 << 0
    if flags.get("pf", False):
        eflags |= 1 << 2
    if flags.get("af", False):
        eflags |= 1 << 4
    if flags.get("zf", False):
        eflags |= 1 << 6
    if flags.get("sf", False):
        eflags |= 1 << 7
    if flags.get("df", False):
        eflags |= 1 << 10
    if flags.get("of", False):
        eflags |= 1 << 11
    return eflags


def get_return_register(arch: Arch) -> str:
    """get the return value register name"""
    if arch == Arch.ARM64:
        return "x0"
    elif arch == Arch.X64:
        return "rax"
    elif arch == Arch.X86:
        return "eax"
    else:
        raise ValueError(f"unknown architecture: {arch}")


def get_word_size(arch: Arch) -> int:
    """get the word size for architecture"""
    return arch.pointer_size


def get_stack_alignment(arch: Arch) -> int:
    """get required stack alignment for architecture"""
    if arch == Arch.ARM64:
        return 16
    elif arch == Arch.X64:
        return 16
    elif arch == Arch.X86:
        return 4
    else:
        raise ValueError(f"unknown architecture: {arch}")


# default stack addresses for different architectures
DEFAULT_STACK_BASE = {
    Arch.ARM64: 0x300000000,
    Arch.X64: 0x300000000,
    Arch.X86: 0x30000000,
}
