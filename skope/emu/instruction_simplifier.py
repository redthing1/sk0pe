#!/usr/bin/env python3
"""
instruction simplifier - transforms complex instructions to simpler equivalents

clean pipeline: capstone → simplify → keystone
"""

from typing import Optional, Tuple, Dict
import capstone as cs
import keystone as ks
from redlog import get_logger

log = get_logger("simplifier")

# simplification rules
SIMPLIFICATION_RULES = {
    "arm64": {
        "ldapr": ("ldr", "load-acquire → load"),
        "ldaprb": ("ldrb", "load-acquire byte → load byte"),
        "ldaprh": ("ldrh", "load-acquire halfword → load halfword"),
        "stlr": ("str", "store-release → store"),
        "stlrb": ("strb", "store-release byte → store byte"),
        "stlrh": ("strh", "store-release halfword → store halfword"),
    },
    "x64": {},
    "x86": {},
}


def get_arch_config(arch: str) -> Dict:
    """get capstone and keystone arch/mode config"""
    configs = {
        "arm64": {
            "cs_arch": cs.CS_ARCH_ARM64,
            "cs_mode": cs.CS_MODE_ARM,
            "ks_arch": ks.KS_ARCH_ARM64,
            "ks_mode": ks.KS_MODE_LITTLE_ENDIAN,
        },
        "x64": {
            "cs_arch": cs.CS_ARCH_X86,
            "cs_mode": cs.CS_MODE_64,
            "ks_arch": ks.KS_ARCH_X86,
            "ks_mode": ks.KS_MODE_64,
        },
        "x86": {
            "cs_arch": cs.CS_ARCH_X86,
            "cs_mode": cs.CS_MODE_32,
            "ks_arch": ks.KS_ARCH_X86,
            "ks_mode": ks.KS_MODE_32,
        },
    }
    return configs.get(arch, configs["arm64"])


def find_basic_block_end(data: bytes, start_addr: int, arch: str = "arm64") -> int:
    """find end address of basic block (address after last instruction)"""
    config = get_arch_config(arch)
    md = cs.Cs(config["cs_arch"], config["cs_mode"])
    md.detail = True

    last_addr = start_addr
    for insn in md.disasm(data, start_addr):
        last_addr = insn.address + insn.size
        # check control flow groups
        if any(
            g in insn.groups
            for g in [
                cs.CS_GRP_JUMP,
                cs.CS_GRP_CALL,
                cs.CS_GRP_RET,
                cs.CS_GRP_IRET,
                cs.CS_GRP_BRANCH_RELATIVE,
            ]
        ):
            return last_addr

    return last_addr


def simplify_instruction(
    mnemonic: str, operands: str, arch: str = "arm64"
) -> Optional[Tuple[str, str]]:
    """check if instruction can be simplified. returns (new_mnemonic, new_operands) or None"""
    rules = SIMPLIFICATION_RULES.get(arch, {})
    if mnemonic in rules:
        new_mnemonic, comment = rules[mnemonic]
        log.dbg(f"simplifying {mnemonic} → {new_mnemonic} ({comment})")
        return (new_mnemonic, operands)
    return None


def disassemble_single(
    data: bytes, address: int, arch: str = "arm64"
) -> Optional[cs.CsInsn]:
    """disassemble a single instruction"""
    config = get_arch_config(arch)
    md = cs.Cs(config["cs_arch"], config["cs_mode"])
    return next(md.disasm(data, address, 1), None)


def assemble_single(asm_str: str, address: int, arch: str = "arm64") -> Optional[bytes]:
    """assemble a single instruction"""
    config = get_arch_config(arch)
    try:
        ks_engine = ks.Ks(config["ks_arch"], config["ks_mode"])
        encoding, _ = ks_engine.asm(asm_str, address)
        return bytes(encoding) if encoding else None
    except Exception as e:
        log.warn(f"keystone error for '{asm_str}': {e}")
        return None


def patch_instruction(
    data: bytes, offset: int, address: int, arch: str = "arm64"
) -> Optional[bytes]:
    """attempt to patch a single instruction. returns patched bytes or None if skipped"""
    # disassemble
    insn = disassemble_single(data[offset:], address, arch)
    if not insn:
        log.warn(f"failed to disassemble at 0x{address:x}")
        return None

    # check if we can simplify
    simplified = simplify_instruction(insn.mnemonic, insn.op_str, arch)
    if not simplified:
        return None  # no simplification needed

    # assemble new instruction
    new_asm = f"{simplified[0]} {simplified[1]}"
    new_bytes = assemble_single(new_asm, address, arch)
    if not new_bytes:
        log.warn(f"failed to assemble '{new_asm}' at 0x{address:x}")
        return None

    # check size match
    if len(new_bytes) != insn.size:
        log.warn(
            f"size mismatch at 0x{address:x}: {insn.mnemonic} ({insn.size}B) → {simplified[0]} ({len(new_bytes)}B), skipping"
        )
        return None

    # patch the bytes
    result = bytearray(data)
    result[offset : offset + insn.size] = new_bytes
    log.dbg(f"patched 0x{address:x}: {insn.mnemonic} → {simplified[0]}")
    return bytes(result)


def patch_basic_block(data: bytes, start_addr: int, arch: str = "arm64") -> bytes:
    """patch all simplifiable instructions in a basic block"""
    end_addr = find_basic_block_end(data, start_addr, arch)
    bb_size = end_addr - start_addr

    log.dbg(f"patching basic block 0x{start_addr:x} - 0x{end_addr:x} ({bb_size} bytes)")

    # work on a copy
    result = data

    # disassemble and patch each instruction
    config = get_arch_config(arch)
    md = cs.Cs(config["cs_arch"], config["cs_mode"])

    for insn in md.disasm(result[:bb_size], start_addr):
        offset = insn.address - start_addr
        patched = patch_instruction(result, offset, insn.address, arch)
        if patched:
            result = patched

        # stop at control flow
        if insn.address + insn.size >= end_addr:
            break

    return result


# emulator integration (minimal coupling)
def patch_single_instruction_in_emulator(emu, address: int, arch: str = None) -> bool:
    """patch a single instruction in emulator memory if it can be simplified"""
    if not arch:
        arch = emu.exe.arch.name.lower()

    # read just enough for one instruction
    try:
        data = emu.mem_read(address, 16)  # max instruction size
    except Exception as e:
        log.err(f"failed to read memory at 0x{address:x}: {e}")
        return False

    # try to patch
    patched = patch_instruction(data, 0, address, arch)
    if not patched:
        return True  # nothing to patch is still success

    # get instruction size and write back
    insn = disassemble_single(data, address, arch)
    if insn:
        try:
            emu.mem_write(address, patched[: insn.size])
            return True
        except Exception as e:
            log.err(f"failed to write memory at 0x{address:x}: {e}")

    return False


def patch_basic_block_in_emulator(emu, address: int, arch: str = None) -> bool:
    """patch a basic block in emulator memory"""
    if not arch:
        arch = emu.exe.arch.name.lower()

    # read enough for a basic block
    try:
        data = emu.mem_read(address, 1024)  # conservative size
    except Exception as e:
        log.err(f"failed to read memory at 0x{address:x}: {e}")
        return False

    # patch and write back if changed
    patched = patch_basic_block(data, address, arch)
    end_addr = find_basic_block_end(data, address, arch)
    bb_size = end_addr - address

    if patched[:bb_size] != data[:bb_size]:
        try:
            emu.mem_write(address, patched[:bb_size])
            log.dbg(f"wrote patched basic block to 0x{address:x}")
        except Exception as e:
            log.err(f"failed to write memory at 0x{address:x}: {e}")
            return False

    return True
