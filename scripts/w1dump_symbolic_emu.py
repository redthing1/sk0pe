#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import skope
from skope.backends import EmulatorConfig, HookType
from skope.core.arch import Architecture


def _parse_address(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = value.strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    return int(text, 16)


def _parse_reg_assignment(text: str) -> Tuple[str, int]:
    if "=" not in text:
        raise ValueError("register assignment must be name=value")
    name, value = text.split("=", 1)
    return name.strip(), _parse_address(value.strip())


def _parse_symbolic_mem(text: str) -> Tuple[int, int, int]:
    parts = text.split(":")
    if len(parts) < 2:
        raise ValueError("symbolic memory format: addr:size[:word]")
    addr = _parse_address(parts[0])
    size = int(parts[1], 0)
    word = int(parts[2], 0) if len(parts) > 2 else 1
    if addr is None:
        raise ValueError("symbolic memory requires an address")
    if size <= 0:
        raise ValueError("symbolic memory size must be positive")
    if word not in (1, 2, 4, 8):
        raise ValueError("symbolic memory word size must be 1, 2, 4, or 8")
    if size % word != 0:
        raise ValueError("symbolic memory size must be divisible by word size")
    return addr, size, word


def _dedupe_regs(regs: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for reg in regs:
        reg_name = reg.lower()
        if reg_name in seen:
            continue
        seen.add(reg_name)
        ordered.append(reg_name)
    return ordered


def _filter_pointer_regs(arch: Architecture, regs: Iterable[str]) -> List[str]:
    skip = {arch.sp_name}
    frame = getattr(arch, "frame_pointer_name", None)
    if frame:
        skip.add(frame)
    return [reg for reg in regs if reg not in skip]


def _make_disassembler(arch: Architecture):
    import capstone as cs

    if arch == Architecture.X86:
        return cs.Cs(cs.CS_ARCH_X86, cs.CS_MODE_32)
    if arch == Architecture.X64:
        return cs.Cs(cs.CS_ARCH_X86, cs.CS_MODE_64)
    if arch == Architecture.ARM64:
        return cs.Cs(cs.CS_ARCH_ARM64, cs.CS_MODE_ARM)
    raise ValueError(f"unsupported architecture: {arch}")


def _trace_hook(emu, arch: Architecture):
    disasm = _make_disassembler(arch)
    state = {"count": 0}

    def on_code(address: int, size: int) -> bool:
        state["count"] += 1
        try:
            blob = emu.read(address, min(size, 16))
            insn = next(disasm.disasm(blob, address, count=1), None)
            if insn:
                text = f"{insn.mnemonic} {insn.op_str}".strip()
                bytes_text = " ".join(f"{b:02x}" for b in insn.bytes)
                print(
                    f"[{state['count']:06d}] 0x{address:016x}: {bytes_text:<20} {text}"
                )
            else:
                print(f"[{state['count']:06d}] 0x{address:016x}: <{blob[:size].hex()}>")
        except Exception as exc:
            print(f"[{state['count']:06d}] 0x{address:016x}: <trace failed: {exc}>")
        return True

    return on_code


def _symbolize_triton_registers(ctx, regs: Iterable[str]) -> None:
    for reg_name in regs:
        reg = getattr(ctx.registers, reg_name, None)
        if reg is None:
            continue
        sym = ctx.symbolizeRegister(reg, f"sym_{reg_name}")
        if sym is not None:
            sym.setAlias(f"sym_{reg_name}")


def _symbolize_triton_memory(ctx, entries: List[Tuple[int, int, int]]) -> None:
    from triton import CPUSIZE, MemoryAccess

    size_map = {1: CPUSIZE.BYTE, 2: CPUSIZE.WORD, 4: CPUSIZE.DWORD, 8: CPUSIZE.QWORD}
    for addr, size, word in entries:
        cpu_size = size_map[word]
        count = size // word
        for i in range(count):
            offset = addr + i * word
            mem = MemoryAccess(offset, cpu_size)
            sym = ctx.symbolizeMemory(mem)
            sym.setAlias(f"mem_{offset:x}")


def _format_triton_ast(ctx, ast) -> str:
    ast_ctx = ctx.getAstContext()
    try:
        simplified = ctx.simplify(ast, solver=False, llvm=False)
    except TypeError:
        simplified = ctx.simplify(ast)
    unrolled = ast_ctx.unroll(simplified)
    size = unrolled.getBitvectorSize()
    raw_text = str(simplified)
    unrolled_text = str(unrolled)
    if raw_text != unrolled_text:
        return f"{unrolled_text} (bv{size})"
    return f"{raw_text} (bv{size})"


def _dump_triton_registers(ctx, arch: Architecture) -> None:
    print("\n[skope] symbolic register expressions")
    reg_names = arch.register_names("gp") + [arch.pc_name, arch.sp_name]
    for reg_name in reg_names:
        reg = getattr(ctx.registers, reg_name, None)
        if not reg:
            continue
        sym_expr = ctx.getSymbolicRegister(reg)
        concrete = ctx.getConcreteRegisterValue(reg)
        if sym_expr:
            ast_text = _format_triton_ast(ctx, sym_expr.getAst())
            print(f"  {reg_name:<6} = {ast_text}  (concrete=0x{concrete:x})")
        else:
            print(f"  {reg_name:<6} = 0x{concrete:x}")


def _dump_triton_symbolic_memory(ctx, limit: int) -> None:
    memory = ctx.getSymbolicMemory()
    if not memory:
        print("\n[skope] no symbolic memory entries")
        return
    print("\n[skope] symbolic memory entries")
    for addr in sorted(memory.keys())[:limit]:
        ast_text = _format_triton_ast(ctx, memory[addr].getAst())
        print(f"  0x{addr:x} = {ast_text}")


def _symbolize_maat_registers(engine, arch: Architecture, regs: Iterable[str]) -> None:
    import maat

    bits = arch.bits
    for reg_name in regs:
        try:
            setattr(engine.cpu, reg_name, maat.Var(bits, f"sym_{reg_name}"))
        except Exception:
            continue


def _symbolize_maat_memory(engine, entries: List[Tuple[int, int, int]]) -> None:
    if not hasattr(engine.mem, "make_concolic"):
        print(
            "[skope] maat engine does not expose make_concolic; skipping memory symbols"
        )
        return
    for addr, size, word in entries:
        engine.mem.make_concolic(addr, size, word, f"mem_{addr:x}")


def _maat_value_info(value, varctx=None) -> Tuple[str, Optional[int], bool]:
    expr_text = str(value)
    is_symbolic = False
    is_concolic = False
    is_symbolic_fn = getattr(value, "is_symbolic", None)
    if callable(is_symbolic_fn):
        try:
            is_symbolic = (
                is_symbolic_fn(varctx) if varctx is not None else is_symbolic_fn()
            )
        except Exception:
            pass
    is_concolic_fn = getattr(value, "is_concolic", None)
    if callable(is_concolic_fn):
        try:
            is_concolic = (
                is_concolic_fn(varctx) if varctx is not None else is_concolic_fn()
            )
        except Exception:
            pass
    concrete = None
    if hasattr(value, "as_uint"):
        try:
            concrete = int(
                value.as_uint(varctx) if varctx is not None else value.as_uint()
            )
        except Exception:
            concrete = None
    return expr_text, concrete, is_symbolic or is_concolic


def _dump_maat_registers(engine, arch: Architecture) -> None:
    print("\n[skope] register values")
    reg_names = arch.register_names("gp") + [arch.pc_name, arch.sp_name]
    for reg_name in reg_names:
        if not hasattr(engine.cpu, reg_name):
            continue
        value = getattr(engine.cpu, reg_name)
        expr_text, concrete, is_symbolic = _maat_value_info(value, engine.vars)
        if is_symbolic and concrete is not None:
            print(f"  {reg_name:<6} = {expr_text}  (concrete=0x{concrete:x})")
        else:
            print(f"  {reg_name:<6} = {expr_text}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Symbolic emulation demo for w1dump snapshots."
    )
    parser.add_argument("dump", help="Path to the .w1dump file")
    parser.add_argument(
        "--backend",
        "-b",
        choices=["triton", "maat"],
        default="triton",
        help="Backend to use (default: triton)",
    )
    parser.add_argument(
        "--start-addr",
        "-s",
        help="Start address in hex (default: snapshot PC)",
    )
    parser.add_argument(
        "--count",
        "-c",
        type=int,
        default=200,
        help="Maximum instruction count to execute",
    )
    parser.add_argument(
        "--trace",
        "-t",
        action="store_true",
        help="Print each executed instruction",
    )
    parser.add_argument(
        "--set-reg",
        action="append",
        default=[],
        help="Set a register (name=value); can be repeated",
    )
    parser.add_argument(
        "--symbolize-reg",
        action="append",
        default=[],
        help="Symbolize a register by name; can be repeated",
    )
    parser.add_argument(
        "--symbolize-all-gpr",
        action="store_true",
        help="Symbolize all general-purpose registers (excludes sp/fp)",
    )
    parser.add_argument(
        "--symbolize-mem",
        action="append",
        default=[],
        help="Symbolize memory: addr:size[:word] (word defaults to 1)",
    )
    parser.add_argument(
        "--symbolize-loads",
        action="store_true",
        help="Symbolize bytes read by load instructions",
    )
    parser.add_argument(
        "--show-symbolic-memory",
        action="store_true",
        help="Print symbolic memory entries (Triton only)",
    )
    parser.add_argument(
        "--symbolic-memory-limit",
        type=int,
        default=32,
        help="Maximum symbolic memory entries to print",
    )
    parser.add_argument(
        "--map-zero-page",
        action="store_true",
        default=None,
        help="Map a zero page to tolerate null-pointer reads",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v, -vv, -vvv)",
    )
    args = parser.parse_args()

    dump_path = Path(args.dump).expanduser()
    if not dump_path.exists():
        raise SystemExit(f"dump file not found: {dump_path}")

    config = EmulatorConfig(load_strategy="lazy", map_zero_page=args.map_zero_page)
    session = skope.open_w1dump(str(dump_path), backend=args.backend, config=config)
    emu = session.emulator

    if args.set_reg:
        for item in args.set_reg:
            name, value = _parse_reg_assignment(item)
            emu.reg_write(name, value)

    start_addr = _parse_address(args.start_addr) if args.start_addr else emu.pc
    emu.pc = start_addr

    if args.trace:
        emu.hooks.add(HookType.CODE, _trace_hook(emu, session.snapshot.arch))

    if args.verbose:
        print(
            f"[skope] backend={args.backend} dump={dump_path.name} arch={session.snapshot.arch.value}"
        )
        print(
            f"[skope] start=0x{start_addr:x} count={args.count} trace={'on' if args.trace else 'off'}"
        )
        print(f"[skope] initial pc=0x{emu.pc:x} sp=0x{emu.sp:x}")

    symbolic_regs: List[str] = []
    if args.symbolize_all_gpr:
        regs = session.snapshot.arch.register_names("gp")
        symbolic_regs.extend(_filter_pointer_regs(session.snapshot.arch, regs))
    if args.symbolize_reg:
        symbolic_regs.extend([name.lower() for name in args.symbolize_reg])
    symbolic_regs = _dedupe_regs(symbolic_regs)

    symbolic_mem = [_parse_symbolic_mem(item) for item in args.symbolize_mem]

    if args.backend == "triton":
        from triton import AST_REPRESENTATION

        ctx = emu.backend
        ctx.setAstRepresentationMode(AST_REPRESENTATION.PYTHON)
        emu.enable_snapshot_memory(
            symbolize_loads=args.symbolize_loads, verbose=args.verbose
        )
        if symbolic_regs:
            _symbolize_triton_registers(ctx, symbolic_regs)
        if symbolic_mem:
            _symbolize_triton_memory(ctx, symbolic_mem)
    else:
        engine = emu.backend
        emu.enable_snapshot_memory(
            symbolize_loads=args.symbolize_loads, verbose=args.verbose
        )
        if symbolic_regs:
            _symbolize_maat_registers(engine, session.snapshot.arch, symbolic_regs)
        if symbolic_mem:
            _symbolize_maat_memory(engine, symbolic_mem)

    result = emu.run(start=start_addr, count=args.count)

    print(
        f"[skope] executed={result.executed} stop_reason={result.stop_reason} pc=0x{result.pc:x}"
    )
    if result.error:
        print(f"[skope] error={result.error}")

    if args.backend == "triton":
        ctx = emu.backend
        _dump_triton_registers(ctx, session.snapshot.arch)
        if args.show_symbolic_memory:
            _dump_triton_symbolic_memory(ctx, args.symbolic_memory_limit)
    else:
        _dump_maat_registers(emu.backend, session.snapshot.arch)


if __name__ == "__main__":
    main()
