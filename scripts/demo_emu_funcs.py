#!/usr/bin/env python3
"""
demo: emulate function calls in a test binary
works with both unicorn and triton
"""

from pathlib import Path
from typing import Optional, Dict, Any
import typer
from redlog import get_logger, set_level, Level
from skope.load.lief_loader import (
    load_binary,
    create_lief_triton_emulator_from_executable,
    create_lief_unicorn_emulator_from_executable,
)
from skope.emu import Hook
from skope.emu.calling_convention import CallingConvention, Convention

app = typer.Typer(help="demo of emulated function calling")

# stop address for function returns (we can ret to garbage)
STOP_ADDRESS = 0x8000000000000000


def setup_trace_hook(emu, log):
    """set up proper instruction trace hook"""

    # get disassembler for emu
    disasm = emu.disassembler()

    def trace_code(address, size):
        # read and disassemble instruction
        try:
            code = emu.mem_read(address, min(size, 16))
            insns = list(disasm.disasm(code, address, 1))
            if insns:
                insn = insns[0]
                print(f"  0x{address:08x}: {insn.mnemonic:<8} {insn.op_str}")
            else:
                print(f"  0x{address:08x}: <{code[:size].hex()}>")
        except Exception as e:
            log.err(f"trace error at 0x{address:x}: {e}")
        return True

    emu.hook_code_execute = trace_code


def run_function_test(
    emu,
    cc,
    name: str,
    addr: int,
    args: list,
    expected: int,
    max_instructions: int = 100,
) -> bool:
    """run a single function test"""
    log = get_logger("test")

    log.info(f"\n=== test: {name}({', '.join(str(a) for a in args)}) ===")

    # set up function call with stop address
    cc.call_function(addr, args, return_addr=STOP_ADDRESS)

    # set PC to function start (triton needs this)
    emu.set_pc(addr)

    try:
        # emulate until we hit the stop address
        emu.emulate(start=addr, end=STOP_ADDRESS, count=max_instructions)
    except Exception as e:
        log.dbg(f"emulation stopped: {e}")

    # get return value
    result = cc.get_return_value()
    success = result == expected

    log.info(
        f"{name} returned: {result} ({'correct' if success else f'wrong, expected {expected}'})"
    )
    return success


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
    """demonstrate calling functions"""

    # configure logging
    log = get_logger("demo")
    level = Level(min(Level.INFO + verbose, Level.ANNOYING))
    set_level(level)

    # load binary
    log.inf(f"loading binary: {binary_path}")
    exe = load_binary(str(binary_path))
    log.info(f"format: {exe._format}, arch: {exe.arch.name}")

    # functions to call in binary
    test_cases = [
        # (name, args, expected_result)
        ("no_args", [], 42),
        ("one_arg", [21], 42),
        ("two_args", [10, 20], 30),
        ("many_args", [1, 2, 3, 4, 5, 6, 7, 8], 36),
        ("factorial", [5], 120),
        ("nested_calc", [10], 62),
        ("return_64bit", [0x12345678, 0x9ABCDEF0], 0x9ABCDEF012345678),
        ("array_sum", [0x40000000, 5], 15),  # will need to set up array in memory
        ("memory_test", [0x40001000, 42], 0),  # will need to set up memory
        ("read_global", [], 100),  # initial global value
        ("string_length", [0x40002000], 11),  # will need to set up string
    ]

    # find function addresses
    functions = {}
    for test_name, _, _ in test_cases:
        addr = exe.find_function(test_name)
        if addr:
            functions[test_name] = addr
            log.dbg(f"found {test_name} at 0x{addr:x}")

    # also look for struct test functions
    struct_functions = ["sum_point", "make_point", "sum_large_struct", "modify_point"]
    for func_name in struct_functions:
        addr = exe.find_function(func_name)
        if addr:
            functions[func_name] = addr
            log.dbg(f"found {func_name} at 0x{addr:x}")

    # also look for global test functions
    global_functions = ["write_global", "modify_global"]
    for func_name in global_functions:
        addr = exe.find_function(func_name)
        if addr:
            functions[func_name] = addr
            log.dbg(f"found {func_name} at 0x{addr:x}")

    if not functions:
        log.err("no test functions found in binary")
        raise typer.Exit(1)

    log.info(f"found {len(functions)} test functions")

    # create emulator
    log.info(f"creating {backend} emulator")

    # determine hooks
    hooks = Hook.DEFAULT
    if trace:
        hooks |= Hook.CODE_EXECUTE

    if backend == "triton":
        emu = create_lief_triton_emulator_from_executable(exe, hooks)
    else:
        emu = create_lief_unicorn_emulator_from_executable(exe, hooks)

    # ensure data sections are properly loaded for globals
    # the emulator should already map these, but let's verify
    log.dbg("checking data section mapping for globals")

    # set up trace hook if requested
    if trace:
        setup_trace_hook(emu, log)

    # create calling convention helper
    cc = CallingConvention(emu)

    # parse convention override if provided
    conv_override = None
    if convention:
        try:
            conv_override = Convention(convention.lower())
            log.info(f"using forced convention: {conv_override.value}")
        except ValueError:
            log.err(f"unknown convention: {convention}")
            raise typer.Exit(1)

    # set up stack
    stack_base, stack_size = cc.setup_stack()
    log.info(f"stack ready at 0x{stack_base:x}")

    # set up memory for tests that need it
    # allocate some test memory regions
    test_memory_base = 0x40000000
    if hasattr(emu, "_uc"):  # unicorn
        try:
            emu._map(test_memory_base, 0x10000, 7)  # rwx
        except:
            pass  # might already be mapped

    # set up array for array_sum test
    array_data = [1, 2, 3, 4, 5]
    for i, val in enumerate(array_data):
        emu.mem_write(test_memory_base + i * 4, val.to_bytes(4, "little"))

    # set up memory location for memory_test
    emu.mem_write(0x40001000, (0).to_bytes(4, "little"))

    # set up string for string_length test
    test_string = b"hello world\x00"
    emu.mem_write(0x40002000, test_string)

    # run tests
    passed = 0
    total = 0

    for test_name, args, expected in test_cases:
        if test_name not in functions:
            continue

        # for mixed convention demo, use different conventions for different functions
        if conv_override:
            test_conv = conv_override
        elif test_name == "two_args" and exe.arch.name == "X86":
            # demo: use fastcall for two_args on x86
            test_conv = Convention.FASTCALL
            log.dbg(f"using {test_conv.value} convention for {test_name}")
        else:
            test_conv = None  # use default

        # run the test with appropriate convention
        if test_conv:
            # need to recreate cc.call_function call with convention parameter
            log.info(
                f"\n=== test: {test_name}({', '.join(str(a) for a in args)}) [conv: {test_conv.value}] ==="
            )
            cc.call_function(
                functions[test_name],
                args,
                return_addr=STOP_ADDRESS,
                convention=test_conv,
            )

            # set PC to function start (triton needs this)
            emu.set_pc(functions[test_name])

            try:
                emu.emulate(
                    start=functions[test_name],
                    end=STOP_ADDRESS,
                    count=500 if test_name in ("factorial", "string_length") else 100,
                )
            except Exception as e:
                log.dbg(f"emulation stopped: {e}")

            result = cc.get_return_value()
            success = result == expected
            log.info(
                f"{test_name} returned: {result} ({'correct' if success else f'wrong, expected {expected}'})"
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

    # test struct passing: Point struct gets passed as packed value in ARM64
    if "sum_point" in functions:
        log.info("\n=== test: struct passing (Point) ===")

        # Point struct with x=10, y=20: pack into single 64-bit argument
        # ARM64 packs small structs (<= 16 bytes) into registers
        point_packed = (20 << 32) | 10  # y in high bits, x in low bits (little endian)

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

    # test stateful functions
    if "increment_counter" in functions and "get_counter" in functions:
        log.info("\n=== test: stateful functions ===")

        # call increment_counter three times
        for i in range(3):
            cc.call_function(
                functions["increment_counter"], [], return_addr=STOP_ADDRESS
            )
            emu.set_pc(functions["increment_counter"])
            try:
                emu.emulate(
                    start=functions["increment_counter"], end=STOP_ADDRESS, count=50
                )
            except Exception as e:
                log.dbg(f"emulation stopped: {e}")
            result = cc.get_return_value()
            log.dbg(f"increment_counter() call {i+1} returned: {result}")

        # check final counter value
        cc.call_function(functions["get_counter"], [], return_addr=STOP_ADDRESS)
        emu.set_pc(functions["get_counter"])
        try:
            emu.emulate(start=functions["get_counter"], end=STOP_ADDRESS, count=50)
        except Exception as e:
            log.dbg(f"emulation stopped: {e}")

        result = cc.get_return_value()
        success = result == 3
        log.info(
            f"get_counter() returned: {result} ({'correct' if success else 'wrong, expected 3'})"
        )

        total += 1
        if success:
            passed += 1

    # test global variable modification
    if (
        "write_global" in functions
        and "read_global" in functions
        and "modify_global" in functions
    ):
        log.info("\n=== test: global variable operations ===")

        # write a new value
        cc.call_function(functions["write_global"], [200], return_addr=STOP_ADDRESS)
        emu.set_pc(functions["write_global"])
        try:
            emu.emulate(start=functions["write_global"], end=STOP_ADDRESS, count=50)
        except Exception as e:
            log.dbg(f"emulation stopped: {e}")

        # read it back
        cc.call_function(functions["read_global"], [], return_addr=STOP_ADDRESS)
        emu.set_pc(functions["read_global"])
        try:
            emu.emulate(start=functions["read_global"], end=STOP_ADDRESS, count=50)
        except Exception as e:
            log.dbg(f"emulation stopped: {e}")

        result = cc.get_return_value()
        success = result == 200
        log.info(
            f"read_global() after write returned: {result} ({'correct' if success else 'wrong, expected 200'})"
        )

        # modify it
        cc.call_function(functions["modify_global"], [50], return_addr=STOP_ADDRESS)
        emu.set_pc(functions["modify_global"])
        try:
            emu.emulate(start=functions["modify_global"], end=STOP_ADDRESS, count=50)
        except Exception as e:
            log.dbg(f"emulation stopped: {e}")

        result = cc.get_return_value()
        success = result == 250
        log.info(
            f"modify_global(50) returned: {result} ({'correct' if success else 'wrong, expected 250'})"
        )

        total += 2
        if success:
            passed += 2

    log.info(f"\n✓ passed {passed}/{total} tests")

    if passed < total:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
