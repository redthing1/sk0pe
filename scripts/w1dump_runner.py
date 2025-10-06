#!/usr/bin/env python3
"""Emulate instructions captured inside a w1dump snapshot."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from redlog import Level, field, get_logger, set_level

from skope.emu import Hook
from skope.load.w1dump_loader import (
    create_w1dump_unicorn_emulator,
    create_w1dump_triton_emulator,
    create_w1dump_maat_emulator,
)


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
APP_NAME = "w1dump-runner"
app = typer.Typer(
    name=APP_NAME,
    help=f"{APP_NAME}: emulate a recorded process snapshot",
    no_args_is_help=True,
    context_settings=CONTEXT_SETTINGS,
    pretty_exceptions_short=True,
    pretty_exceptions_show_locals=False,
)


class Backend(str, Enum):
    """Supported emulator backends for w1dump execution."""

    UNICORN = "unicorn"
    TRITON = "triton"
    MAAT = "maat"


BACKEND_FACTORIES = {
    Backend.UNICORN: create_w1dump_unicorn_emulator,
    Backend.TRITON: create_w1dump_triton_emulator,
    Backend.MAAT: create_w1dump_maat_emulator,
}


def configure_logging(verbosity: int) -> None:
    level = Level(min(Level.INFO + verbosity, Level.ANNOYING))
    set_level(level)


def pick_factory(backend: Backend):
    try:
        return BACKEND_FACTORIES[backend]
    except KeyError as exc:  # pragma: no cover
        raise typer.BadParameter(f"unsupported backend: {backend}") from exc


def install_trace_hook(emu, dump):
    disasm = emu.disassembler()
    trace_log = get_logger("w1dump.trace")

    def trace_code(address, size):
        try:
            code = emu.mem_read(address, min(size, 16))
            insns = list(disasm.disasm(code, address, count=1))
            insn = insns[0] if insns else None
            if insn:
                text = f"{insn.mnemonic:8} {insn.op_str}".rstrip()
            else:
                text = f"<{code[:size].hex()}>"

            module = dump.get_module_at(address)
            module_name = module.name if module else "<unknown>"
            print(f"  0x{address:016x}: {text:<40} [{module_name}]")
        except Exception as exc:  # pylint: disable=broad-except
            trace_log.err(f"trace failed @0x{address:016x}: {exc}")
        return True

    emu.hook_code_execute = trace_code


def install_memory_error_logger(emu):
    mem_log = get_logger("w1dump.mem")

    try:
        import unicorn as uc  # type: ignore

        access_map = {
            getattr(uc, "UC_MEM_READ", None): "read",
            getattr(uc, "UC_MEM_WRITE", None): "write",
            getattr(uc, "UC_MEM_FETCH", None): "fetch",
        }
    except ImportError:  # pragma: no cover - backend may not be Unicorn
        access_map = {}

    def describe(access: int) -> str:
        return access_map.get(access, f"unknown({access})")

    def memory_error(access: int, address: int, size: int, value: int) -> bool:
        mem_log.err(
            "memory access error",
            field("type", describe(access)),
            field("address", f"0x{address:x}"),
            field("size", size),
            field("value", value),
        )
        return False

    emu.hook_memory_error = memory_error


@app.command()
def run(
    dump_file: Path = typer.Option(
        ..., "--dump", "-d", help="Path to the .w1dump file"
    ),
    backend: Backend = typer.Option(Backend.UNICORN, "--backend", "-b"),
    module: Optional[str] = typer.Option(
        None, "--module", "-m", help="Module name to load (default: main module)"
    ),
    start: Optional[str] = typer.Option(
        None,
        "--start",
        "-s",
        help="Override start address (hex string, default: recorded PC)",
    ),
    count: int = typer.Option(
        1000,
        "--count",
        "-c",
        min=0,
        help="Maximum number of instructions to execute",
    ),
    trace: bool = typer.Option(
        False, "--trace", "-t", help="Print executed instructions"
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase log verbosity"
    ),
):
    """Emulate instructions captured inside a w1dump snapshot."""

    configure_logging(verbose)
    log = get_logger("w1dump.runner")

    dump_path = dump_file.expanduser()
    if not dump_path.exists():
        raise typer.BadParameter(f"dump file not found: {dump_path}")

    hooks = Hook.DEFAULT
    if trace:
        hooks |= Hook.CODE_EXECUTE

    factory = pick_factory(backend)
    log.inf(
        "loading dump", field("path", str(dump_path)), field("backend", backend.value)
    )

    emu, dump = factory(str(dump_path), module, hooks)

    log.info(
        "snapshot",
        field("process", dump.metadata.process_name),
        field("pid", dump.metadata.pid),
        field("arch", dump.metadata.arch),
        field("os", dump.metadata.os),
    )
    log.info(
        "contents",
        field("modules", len(dump.modules)),
        field("regions", len(dump.regions)),
    )

    if trace:
        install_trace_hook(emu, dump)

    install_memory_error_logger(emu)

    start_addr = int(start, 16) if start else emu.pc

    log.info(
        "initial state",
        field("pc", f"0x{emu.pc:016x}"),
        field("sp", f"0x{emu.sp:016x}"),
    )
    log.info(
        "starting emulation",
        field("start", f"0x{start_addr:016x}"),
        field("max_insns", count),
    )

    try:
        emu.emulate(start=start_addr, count=count)
        log.info("emulation completed")
    except Exception as exc:  # pylint: disable=broad-except
        log.err(f"emulation stopped: {exc}")

    module_at_end = dump.get_module_at(emu.pc)
    log.info(
        "final state",
        field("pc", f"0x{emu.pc:016x}"),
        field("sp", f"0x{emu.sp:016x}"),
        field("module", module_at_end.name if module_at_end else "<unknown>"),
    )


if __name__ == "__main__":
    app()
