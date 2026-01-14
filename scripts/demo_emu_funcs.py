#!/usr/bin/env python3
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import lief
import typer
from redlog import Level, get_logger, set_level

import skope
from skope.backends import EmulatorConfig, HookType
from skope.core.arch import Architecture
from skope.core.permissions import MemoryPermissions

app = typer.Typer(help="demo of emulated function calling")

# stop address for function returns (we can ret to garbage)
STOP_ADDRESS = 0x8000000000000000


class Convention(Enum):
    CDECL = "cdecl"
    STDCALL = "stdcall"
    FASTCALL = "fastcall"
    THISCALL = "thiscall"
    SYSV64 = "sysv64"
    WIN64 = "win64"
    AAPCS64 = "aapcs64"
    WIN_ARM64 = "win_arm64"


class CallingConvention:
    def __init__(self, emu):
        self.emu = emu
        self.arch = emu.arch
        self.sp_reg = self.arch.sp_name
        self.ret_reg = self.arch.return_name
        self.word_size = self.arch.pointer_size
        self.alignment = self.arch.stack_alignment

    def setup_stack(self, stack_base: Optional[int] = None, stack_size: int = 0x100000):
        if stack_base is None:
            if self.arch in (Architecture.X64, Architecture.ARM64):
                stack_base = 0x7FFF00000000
            else:
                stack_base = 0xBFFF0000

        try:
            self.emu.map(stack_base, stack_size, int(MemoryPermissions.RW))
        except Exception:
            pass

        stack_top = stack_base + stack_size
        stack_top &= ~(self.alignment - 1)
        self.emu.reg_write(self.sp_reg, stack_top)
        return stack_base, stack_size

    def call_function(
        self,
        func_addr: int,
        args: list,
        return_addr: int = STOP_ADDRESS,
        convention: Optional[Convention] = None,
    ) -> None:
        if convention is None:
            convention = self._default_convention()

        if self.arch == Architecture.X86:
            self._call_x86(func_addr, args, return_addr, convention)
        elif self.arch == Architecture.X64:
            self._call_x64(func_addr, args, return_addr, convention)
        elif self.arch == Architecture.ARM64:
            self._call_arm64(func_addr, args, return_addr, convention)

    def get_return_value(self) -> int:
        return self.emu.reg_read(self.ret_reg)

    def _default_convention(self) -> Convention:
        if self.arch == Architecture.X86:
            return Convention.CDECL
        if self.arch == Architecture.X64:
            return Convention.SYSV64
        return Convention.AAPCS64

    def _write_int(self, addr: int, value: int, size: int) -> None:
        self.emu.write(addr, value.to_bytes(size, "little"))

    def _call_x86(
        self, func_addr: int, args: list, return_addr: int, convention: Convention
    ) -> None:
        sp = self.emu.reg_read(self.sp_reg)

        if convention == Convention.FASTCALL and len(args) >= 1:
            if len(args) >= 1:
                self.emu.reg_write("ecx", args[0])
            if len(args) >= 2:
                self.emu.reg_write("edx", args[1])
            stack_args = args[2:]
        elif convention == Convention.THISCALL and len(args) >= 1:
            self.emu.reg_write("ecx", args[0])
            stack_args = args[1:]
        else:
            stack_args = args

        sp -= 4
        self._write_int(sp, return_addr, 4)

        for arg in reversed(stack_args):
            sp -= 4
            self._write_int(sp, arg & 0xFFFFFFFF, 4)

        self.emu.reg_write(self.sp_reg, sp)

    def _call_x64(
        self, func_addr: int, args: list, return_addr: int, convention: Convention
    ) -> None:
        sp = self.emu.reg_read(self.sp_reg)

        sp -= 8
        self._write_int(sp, return_addr, 8)

        if convention == Convention.SYSV64:
            reg_args = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
        else:
            reg_args = ["rcx", "rdx", "r8", "r9"]
            sp -= 32

        for i, arg in enumerate(args[: len(reg_args)]):
            self.emu.reg_write(reg_args[i], arg)

        stack_args = args[len(reg_args) :]
        for arg in reversed(stack_args):
            sp -= 8
            self._write_int(sp, arg, 8)

        if sp % 16 != 0:
            sp -= 8

        self.emu.reg_write(self.sp_reg, sp)

    def _call_arm64(
        self, func_addr: int, args: list, return_addr: int, convention: Convention
    ) -> None:
        for i, arg in enumerate(args[:8]):
            self.emu.reg_write(f"x{i}", arg)

        self.emu.reg_write("x30", return_addr)
        sp = self.emu.reg_read(self.sp_reg)
        self.emu.reg_write("x29", sp)

        if len(args) > 8:
            stack_args = args[8:]
            for arg in reversed(stack_args):
                sp -= 8
                self._write_int(sp, arg, 8)
            self.emu.reg_write(self.sp_reg, sp)


def _make_disassembler(arch: Architecture):
    import capstone as cs

    if arch == Architecture.X86:
        return cs.Cs(cs.CS_ARCH_X86, cs.CS_MODE_32)
    if arch == Architecture.X64:
        return cs.Cs(cs.CS_ARCH_X86, cs.CS_MODE_64)
    if arch == Architecture.ARM64:
        return cs.Cs(cs.CS_ARCH_ARM64, cs.CS_MODE_ARM)
    raise ValueError(f"unsupported architecture: {arch}")


def setup_trace_hook(emu, log):
    disasm = _make_disassembler(emu.arch)

    def trace_code(address, size):
        try:
            code = emu.read(address, min(size, 16))
            insns = list(disasm.disasm(code, address, 1))
            if insns:
                insn = insns[0]
                print(f"  0x{address:08x}: {insn.mnemonic:<8} {insn.op_str}")
            else:
                print(f"  0x{address:08x}: <{code[:size].hex()}>")
        except Exception as exc:
            log.err(f"trace error at 0x{address:x}: {exc}")
        return True

    emu.hooks.add(HookType.CODE, trace_code)


def run_function_test(
    emu,
    cc: CallingConvention,
    name: str,
    addr: int,
    args: list,
    expected: int,
    max_instructions: int = 100,
) -> bool:
    log = get_logger("test")

    log.info(f"\n=== test: {name}({', '.join(str(a) for a in args)}) ===")

    cc.call_function(addr, args, return_addr=STOP_ADDRESS)
    emu.pc = addr

    result = emu.run(start=addr, end=STOP_ADDRESS, count=max_instructions)
    if result.error:
        log.dbg(f"emulation stopped: {result.error}")

    result_val = cc.get_return_value()
    success = result_val == expected

    log.info(
        f"{name} returned: {result_val} ({'correct' if success else f'wrong, expected {expected}'})"
    )
    return success


def _find_function(binary: lief.Binary, name: str) -> Optional[int]:
    for func in binary.abstract.exported_functions:
        func_name = str(func.name) if func.name else ""
        if name in func_name:
            return int(func.address)

    for sym in binary.abstract.symbols:
        sym_name = str(sym.name) if sym.name else ""
        if name in sym_name and bool(getattr(sym, "is_function", False)):
            return int(sym.value)

    return None


@app.command()
def main(
    binary_path: Path = typer.Argument(
        Path("./build-macos/bin/function_calls"), help="path to binary file"
    ),
    backend: str = typer.Option(
        "unicorn", "--backend", "-b", help="emulation backend (unicorn or triton)"
    ),
    trace: bool = typer.Option(
        False, "--trace", "-t", help="trace execution (print instructions)"
    ),
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="increase verbosity (-v, -vv, -vvv)",
    ),
    convention: Optional[str] = typer.Option(
        None,
        "--convention",
        "-c",
        help="force specific calling convention (cdecl, stdcall, fastcall, sysv64, win64, etc)",
    ),
):
    log = get_logger("demo")
    level = Level(min(Level.INFO + verbose, Level.ANNOYING))
    set_level(level)

    log.inf(f"loading binary: {binary_path}")
    binary = lief.parse(str(binary_path))
    if not binary:
        log.err(f"failed to parse binary: {binary_path}")
        raise typer.Exit(1)

    session = skope.open_binary(
        str(binary_path),
        backend=backend,
        config=EmulatorConfig(load_strategy="all_regions"),
    )
    emu = session.emulator
    log.info(f"arch: {emu.arch.value}")

    test_cases = [
        ("no_args", [], 42),
        ("one_arg", [21], 42),
        ("two_args", [10, 20], 30),
        ("many_args", [1, 2, 3, 4, 5, 6, 7, 8], 36),
        ("factorial", [5], 120),
        ("nested_calc", [10], 62),
        ("return_64bit", [0x12345678, 0x9ABCDEF0], 0x9ABCDEF012345678),
        ("array_sum", [0x40000000, 5], 15),
        ("memory_test", [0x40001000, 42], 0),
        ("read_global", [], 100),
        ("string_length", [0x40002000], 11),
    ]

    functions = {}
    for test_name, _, _ in test_cases:
        addr = _find_function(binary, test_name)
        if addr:
            functions[test_name] = addr
            log.dbg(f"found {test_name} at 0x{addr:x}")

    struct_functions = ["sum_point", "make_point", "sum_large_struct", "modify_point"]
    for func_name in struct_functions:
        addr = _find_function(binary, func_name)
        if addr:
            functions[func_name] = addr
            log.dbg(f"found {func_name} at 0x{addr:x}")

    global_functions = ["write_global", "modify_global"]
    for func_name in global_functions:
        addr = _find_function(binary, func_name)
        if addr:
            functions[func_name] = addr
            log.dbg(f"found {func_name} at 0x{addr:x}")

    if not functions:
        log.err("no test functions found in binary")
        raise typer.Exit(1)

    log.info(f"found {len(functions)} test functions")

    if trace:
        setup_trace_hook(emu, log)

    cc = CallingConvention(emu)

    conv_override = None
    if convention:
        try:
            conv_override = Convention(convention.lower())
            log.info(f"using forced convention: {conv_override.value}")
        except ValueError:
            log.err(f"unknown convention: {convention}")
            raise typer.Exit(1)

    stack_base, stack_size = cc.setup_stack()
    log.info(f"stack ready at 0x{stack_base:x}")

    test_memory_base = 0x40000000
    try:
        emu.map(test_memory_base, 0x10000, int(MemoryPermissions.RWX))
    except Exception:
        pass

    array_data = [1, 2, 3, 4, 5]
    for i, val in enumerate(array_data):
        emu.write(test_memory_base + i * 4, val.to_bytes(4, "little"))

    emu.write(0x40001000, (0).to_bytes(4, "little"))

    test_string = b"hello world\x00"
    emu.write(0x40002000, test_string)

    passed = 0
    total = 0

    for test_name, args, expected in test_cases:
        if test_name not in functions:
            continue

        if conv_override:
            test_conv = conv_override
        elif test_name == "two_args" and emu.arch == Architecture.X86:
            test_conv = Convention.FASTCALL
            log.dbg(f"using {test_conv.value} convention for {test_name}")
        else:
            test_conv = None

        if test_conv:
            log.info(
                f"\n=== test: {test_name}({', '.join(str(a) for a in args)}) [conv: {test_conv.value}] ==="
            )
            cc.call_function(
                functions[test_name],
                args,
                return_addr=STOP_ADDRESS,
                convention=test_conv,
            )
            emu.pc = functions[test_name]
            result = emu.run(
                start=functions[test_name],
                end=STOP_ADDRESS,
                count=(500 if test_name in ("factorial", "string_length") else 100),
            )
            if result.error:
                log.dbg(f"emulation stopped: {result.error}")
            result_val = cc.get_return_value()
            success = result_val == expected
            log.info(
                f"{test_name} returned: {result_val} ({'correct' if success else f'wrong, expected {expected}'})"
            )
        else:
            success = run_function_test(
                emu,
                cc,
                test_name,
                functions[test_name],
                args,
                expected,
                max_instructions=(
                    500 if test_name in ("factorial", "string_length") else 100
                ),
            )

        total += 1
        if success:
            passed += 1

    if "sum_point" in functions:
        log.info("\n=== test: struct passing (Point) ===")
        point_packed = (20 << 32) | 10
        success = run_function_test(
            emu,
            cc,
            "sum_point",
            functions["sum_point"],
            [point_packed],
            30,
            max_instructions=50,
        )
        total += 1
        if success:
            passed += 1

    if "increment_counter" in functions and "get_counter" in functions:
        log.info("\n=== test: stateful functions ===")
        for i in range(3):
            cc.call_function(
                functions["increment_counter"], [], return_addr=STOP_ADDRESS
            )
            emu.pc = functions["increment_counter"]
            emu.run(start=functions["increment_counter"], end=STOP_ADDRESS, count=50)
            result_val = cc.get_return_value()
            log.dbg(f"increment_counter() call {i+1} returned: {result_val}")

        cc.call_function(functions["get_counter"], [], return_addr=STOP_ADDRESS)
        emu.pc = functions["get_counter"]
        emu.run(start=functions["get_counter"], end=STOP_ADDRESS, count=50)

        result_val = cc.get_return_value()
        success = result_val == 3
        log.info(
            f"get_counter() returned: {result_val} ({'correct' if success else 'wrong, expected 3'})"
        )

        total += 1
        if success:
            passed += 1

    if (
        "write_global" in functions
        and "read_global" in functions
        and "modify_global" in functions
    ):
        log.info("\n=== test: global variable operations ===")

        cc.call_function(functions["write_global"], [200], return_addr=STOP_ADDRESS)
        emu.pc = functions["write_global"]
        emu.run(start=functions["write_global"], end=STOP_ADDRESS, count=50)

        cc.call_function(functions["read_global"], [], return_addr=STOP_ADDRESS)
        emu.pc = functions["read_global"]
        emu.run(start=functions["read_global"], end=STOP_ADDRESS, count=50)

        result_val = cc.get_return_value()
        success = result_val == 200
        log.info(
            f"read_global() after write returned: {result_val} ({'correct' if success else 'wrong, expected 200'})"
        )

        cc.call_function(functions["modify_global"], [50], return_addr=STOP_ADDRESS)
        emu.pc = functions["modify_global"]
        emu.run(start=functions["modify_global"], end=STOP_ADDRESS, count=50)

        result_val = cc.get_return_value()
        success = result_val == 250
        log.info(
            f"modify_global(50) returned: {result_val} ({'correct' if success else 'wrong, expected 250'})"
        )

        total += 2
        if success:
            passed += 2

    log.info(f"\n✓ passed {passed}/{total} tests")

    if passed < total:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
