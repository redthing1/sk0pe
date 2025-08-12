#!/usr/bin/env python3
"""
cross-platform calling convention helper
"""

from typing import List, Optional, Dict, Tuple
from enum import Enum
from redlog import get_logger
from .base import Arch, Permission
from .arch import (
    get_sp_register,
    get_return_register,
    get_word_size,
    get_stack_alignment,
    DEFAULT_STACK_BASE,
)


class Convention(Enum):
    """supported calling conventions"""

    # x86 conventions
    CDECL = "cdecl"  # default c convention
    STDCALL = "stdcall"  # windows api
    FASTCALL = "fastcall"  # first 2 args in ecx, edx
    THISCALL = "thiscall"  # this in ecx

    # x64 conventions
    SYSV64 = "sysv64"  # linux/mac x64
    WIN64 = "win64"  # windows x64

    # arm64 conventions
    AAPCS64 = "aapcs64"  # standard arm64
    WIN_ARM64 = "win_arm64"  # windows arm64


class CallingConvention:
    """handles cross-platform function calling conventions"""

    def __init__(self, emulator, stack_base: Optional[int] = None):
        self.emu = emulator
        self.arch = emulator.exe.arch  # Use the proper Arch enum
        self.log = get_logger("calling_convention")

        # use architecture-specific functions from arch.py
        self.sp_reg = get_sp_register(self.arch)
        self.ret_reg = get_return_register(self.arch)
        self.word_size = get_word_size(self.arch)
        self.alignment = get_stack_alignment(self.arch)

        # configurable stack base
        self.default_stack_base = stack_base or DEFAULT_STACK_BASE[self.arch]

    def setup_stack(self, stack_base=None, stack_size=0x100000):
        """set up stack for function calls"""
        if stack_base is None:
            stack_base = self.default_stack_base

        # allocate stack memory
        self.log.dbg(f"setting up stack at 0x{stack_base:x} (size: 0x{stack_size:x})")

        # use the generic map_memory method from base Emulator
        try:
            self.emu.map_memory(stack_base, stack_size, Permission.RW)
        except Exception as e:
            self.log.dbg(f"stack mapping failed: {e}, may already be mapped")

        # initialize stack pointer to top of stack
        self.stack_base = stack_base
        self.stack_size = stack_size
        self.stack_top = stack_base + stack_size

        # align stack for architecture
        self.stack_top &= ~(self.alignment - 1)

        self.emu.set_reg_by_name(self.sp_reg, self.stack_top)
        return stack_base, stack_size

    def call_function(
        self,
        func_addr: int,
        args: List[int],
        return_addr: int = None,
        convention: Convention = None,
    ):
        """call a function with arguments using specified calling convention"""
        if return_addr is None:
            return_addr = 0xDEADBEEF  # default stop address

        # determine convention
        if convention is None:
            convention = self._default_convention()

        self.log.dbg(
            f"calling 0x{func_addr:x} with {len(args)} args using {convention.value}"
        )

        # dispatch to appropriate handler
        if self.arch == Arch.X86:
            self._call_x86(func_addr, args, return_addr, convention)
        elif self.arch == Arch.X64:
            self._call_x64(func_addr, args, return_addr, convention)
        elif self.arch == Arch.ARM64:
            self._call_arm64(func_addr, args, return_addr, convention)

    def _default_convention(self) -> Convention:
        """get default convention for current architecture"""
        if self.arch == Arch.X86:
            return Convention.CDECL
        elif self.arch == Arch.X64:
            # check if windows (would need platform detection)
            return Convention.SYSV64  # default to linux/mac
        elif self.arch == Arch.ARM64:
            return Convention.AAPCS64

    def _call_x86(
        self, func_addr: int, args: List[int], return_addr: int, convention: Convention
    ):
        """handle x86 calling conventions"""
        sp = self.emu.get_reg_by_name(self.sp_reg)

        if convention == Convention.FASTCALL and len(args) >= 1:
            # first 2 args in ecx, edx
            if len(args) >= 1:
                self.emu.set_reg_by_name("ecx", args[0])
            if len(args) >= 2:
                self.emu.set_reg_by_name("edx", args[1])
            stack_args = args[2:]
        elif convention == Convention.THISCALL and len(args) >= 1:
            # this pointer in ecx
            self.emu.set_reg_by_name("ecx", args[0])
            stack_args = args[1:]
        else:
            # cdecl, stdcall - all args on stack
            stack_args = args

        # push return address
        sp -= 4
        self._write_memory(sp, return_addr, 4)

        # push arguments in reverse order
        for arg in reversed(stack_args):
            sp -= 4
            self._write_memory(sp, arg & 0xFFFFFFFF, 4)

        self.emu.set_reg_by_name(self.sp_reg, sp)

    def _call_x64(
        self, func_addr: int, args: List[int], return_addr: int, convention: Convention
    ):
        """handle x64 calling conventions"""
        sp = self.emu.get_reg_by_name(self.sp_reg)

        # push return address first
        sp -= 8
        self._write_memory(sp, return_addr, 8)

        if convention == Convention.SYSV64:
            # system v: rdi, rsi, rdx, rcx, r8, r9
            reg_args = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
        else:  # WIN64
            # windows: rcx, rdx, r8, r9
            reg_args = ["rcx", "rdx", "r8", "r9"]
            # also need shadow space (32 bytes) even for register args
            sp -= 32

        # set register arguments
        for i, arg in enumerate(args[: len(reg_args)]):
            self.emu.set_reg_by_name(reg_args[i], arg)

        # push remaining arguments on stack (right to left)
        stack_args = args[len(reg_args) :]
        for arg in reversed(stack_args):
            sp -= 8
            self._write_memory(sp, arg, 8)

        # ensure 16-byte alignment
        if sp % 16 != 0:
            sp -= 8

        self.emu.set_reg_by_name(self.sp_reg, sp)

    def _call_arm64(
        self, func_addr: int, args: List[int], return_addr: int, convention: Convention
    ):
        """handle arm64 calling conventions"""
        # both aapcs64 and windows arm64 use x0-x7 for args
        for i, arg in enumerate(args[:8]):
            self.emu.set_reg_by_name(f"x{i}", arg)

        # return address in x30 (lr)
        self.emu.set_reg_by_name("x30", return_addr)

        # set frame pointer
        sp = self.emu.get_reg_by_name(self.sp_reg)
        self.emu.set_reg_by_name("x29", sp)

        # handle stack arguments if needed
        if len(args) > 8:
            # push additional args on stack
            stack_args = args[8:]
            for arg in reversed(stack_args):
                sp -= 8
                self._write_memory(sp, arg, 8)
            self.emu.set_reg_by_name(self.sp_reg, sp)

    def get_return_value(self) -> int:
        """get function return value from appropriate register"""
        return self.emu.get_reg_by_name(self.ret_reg)

    def _write_memory(self, addr: int, value: int, size: int):
        """write value to memory"""
        data = value.to_bytes(size, "little")
        self.emu.mem_write(addr, data)
