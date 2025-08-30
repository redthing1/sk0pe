#!/usr/bin/env python3
"""
w1dump loader - common functionality for loading w1dump files into emulators
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from redlog import get_logger, field
from ..emu.base import Executable, Arch, Segment, Hook
from ..emu.unicorn import UnicornEmulator
from ..emu.triton import TritonEmulator
from ..emu.maat import MaatEmulator
from ..formats.w1dump import W1Dump, load_dump


class W1DumpExecutable(Executable):
    """executable backed by a w1dump"""

    def __init__(self, dump: W1Dump, module_name: Optional[str] = None):
        self.dump = dump
        self.module_name = module_name or (
            dump.main_module.name if dump.main_module else None
        )
        self._segments_cache: Optional[List[Segment]] = None
        self.log = get_logger("w1dump.executable")

        self.log.dbg(
            "initializing W1DumpExecutable",
            field("module_name", self.module_name),
            field("available_modules", [m.name for m in dump.modules[:5]]),
        )

    @property
    def arch(self) -> Arch:
        """get architecture from dump metadata"""
        arch_str = self.dump.metadata.arch.lower()

        if arch_str in ("x86_64", "amd64"):
            return Arch.X64
        elif arch_str == "x86":
            return Arch.X86
        elif arch_str == "arm64":
            return Arch.ARM64
        else:
            raise ValueError(f"unsupported architecture: {arch_str}")

    @property
    def base_address(self) -> int:
        """get base address of the main module"""
        if self.module_name:
            for module in self.dump.modules:
                if module.name == self.module_name:
                    return module.base_address
        return self.dump.main_module.base_address if self.dump.main_module else 0

    def get_segments(self) -> List[Segment]:
        """get segments for the specified module"""
        if self._segments_cache is not None:
            return self._segments_cache

        segments = []

        # add regions belonging to this module
        for region in self.dump.regions:
            if region.module_name == self.module_name and region.data:
                segments.append(Segment(region.start, region.data, region.permissions))
                self.log.dbg(
                    f"added segment from module {self.module_name} at 0x{region.start:x} (size: {len(region.data)})"
                )

        # also include the stack region if we have register state
        if self.dump.thread.gpr_state:
            sp = self.dump.thread.gpr_state.sp
            stack_region = self.dump.get_region_at(sp)
            if stack_region and stack_region.data:
                # avoid duplicates
                if not any(seg.address == stack_region.start for seg in segments):
                    segments.append(
                        Segment(
                            stack_region.start,
                            stack_region.data,
                            stack_region.permissions,
                        )
                    )
                    self.log.dbg(
                        f"added stack segment at 0x{stack_region.start:x} (size: {len(stack_region.data)})"
                    )

        self._segments_cache = segments
        self.log.dbg(
            f"found {len(segments)} segments for module {self.module_name or '???'}"
        )
        return segments

    def get_memory_region(self, address: int) -> Optional[Tuple[int, bytes, int]]:
        """get memory region containing address, for lazy loading"""
        region = self.dump.get_region_at(address)
        if region and region.data:
            return (region.start, region.data, region.permissions)
        return None

    def get_initial_state(self) -> Optional[Dict[str, Any]]:
        """get initial cpu state from dump"""
        if not self.dump.thread.gpr_state:
            return None

        state = {
            "arch": self.arch,
            "registers": {},
        }

        gpr = self.dump.thread.gpr_state

        # extract registers based on architecture
        if self.arch == Arch.ARM64:
            # general purpose registers
            for i in range(31):
                reg_name = f"x{i}"
                if hasattr(gpr, reg_name):
                    state["registers"][reg_name] = getattr(gpr, reg_name)
            # special registers
            state["registers"]["sp"] = gpr.sp
            state["registers"]["pc"] = gpr.pc
            if hasattr(gpr, "cpsr"):
                state["registers"]["cpsr"] = gpr.cpsr

        elif self.arch == Arch.X64:
            # standard x64 registers
            for name in [
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
            ]:
                if hasattr(gpr, name):
                    state["registers"][name] = getattr(gpr, name)

        elif self.arch == Arch.X86:
            # standard x86 registers
            for name in ["eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp", "eip"]:
                if hasattr(gpr, name):
                    state["registers"][name] = getattr(gpr, name)

        # add stack info if available
        if hasattr(gpr, "sp"):
            sp = gpr.sp
            stack_region = self.dump.get_region_at(sp)
            if stack_region:
                state["stack_base"] = stack_region.start
                state["stack_size"] = stack_region.size

        return state


class W1DumpLoader:
    """common functionality for loading w1dump files"""

    def __init__(self, dump: W1Dump):
        self.dump = dump
        self.log = get_logger("w1dump.loader")

    def setup_memory_layout(self, emulator, stack_base: int, stack_size: int) -> None:
        """update emulator's stack info from dump if available"""
        if self.dump.thread.gpr_state:
            sp = self.dump.thread.gpr_state.sp
            stack_region = self.dump.get_region_at(sp)

            if stack_region:
                # update stack info to match the dump
                emulator.stack_base = stack_region.start
                emulator.stack_size = stack_region.size
                self.log.trc(
                    f"using stack from dump at 0x{emulator.stack_base:x} (size: 0x{emulator.stack_size:x})"
                )

    def load_registers(self, emulator, gpr_state) -> None:
        """load register state using architecture-independent interface"""
        from ..emu.arch import get_register_names, get_sp_register, get_pc_register
        from ..emu.base import Arch
        
        # load general purpose registers using arch.py
        gp_regs = get_register_names(emulator.exe.arch, "gp")
        for reg_name in gp_regs:
            if hasattr(gpr_state, reg_name):
                value = getattr(gpr_state, reg_name)
                emulator.set_reg_by_name(reg_name, value)  # clean interface!
                if value != 0:
                    self.log.dbg(f"{reg_name} = 0x{value:x}")

        # load special registers
        sp_reg = get_sp_register(emulator.exe.arch)
        pc_reg = get_pc_register(emulator.exe.arch) 
        
        if hasattr(gpr_state, sp_reg):
            sp_value = getattr(gpr_state, sp_reg)
            emulator.set_reg_by_name(sp_reg, sp_value)
            self.log.dbg(f"{sp_reg} = 0x{sp_value:x}")
            
        if hasattr(gpr_state, pc_reg):
            pc_value = getattr(gpr_state, pc_reg)
            emulator.set_reg_by_name(pc_reg, pc_value)
            self.log.dbg(f"{pc_reg} = 0x{pc_value:x}")

        # handle architecture-specific flags
        if emulator.exe.arch == Arch.ARM64 and hasattr(gpr_state, "cpsr"):
            # arm64 status register  
            from ..emu.arch import decode_arm64_cpsr
            flags = decode_arm64_cpsr(gpr_state.cpsr)
            for flag_name, flag_value in flags.items():
                if flag_value:  # only set non-zero flags
                    emulator.set_reg_by_name(flag_name, 1)
            self.log.dbg(f"cpsr = 0x{gpr_state.cpsr:x}")
            
        elif emulator.exe.arch in (Arch.X64, Arch.X86) and hasattr(gpr_state, "eflags"):
            # x86/x64 flags register
            from ..emu.arch import decode_x86_flags
            flags = decode_x86_flags(gpr_state.eflags)
            for flag_name, flag_value in flags.items():
                if flag_value:  # only set non-zero flags
                    emulator.set_reg_by_name(flag_name, 1)
            self.log.dbg(f"eflags = 0x{gpr_state.eflags:x}")

    def load_registers_from_dump(self, emulator) -> None:
        """load register state from dump into emulator"""
        if not self.dump.thread.gpr_state:
            self.log.wrn("no register state in dump")
            return

        gpr = self.dump.thread.gpr_state
        self.log.trc("loading register state from dump")

        # use unified register loading method
        self.load_registers(emulator, gpr)



def load_w1dump(dump_path: str) -> W1Dump:
    """load a w1dump file"""
    log = get_logger("w1dump.loader")
    log.trc(f"loading dump from [{dump_path}]")

    dump = load_dump(dump_path)
    log.dbg(f"loaded dump: {dump.metadata.process_name} (pid: {dump.metadata.pid})")
    log.trc(
        "dump details",
        field("modules", len(dump.modules)),
        field("regions", len(dump.regions)),
        field("main_module", dump.main_module.name if dump.main_module else None),
    )

    return dump


def create_w1dump_unicorn_emulator(
    dump_path: str, module_name: Optional[str] = None, hooks: Hook = Hook.DEFAULT
) -> Tuple[UnicornEmulator, W1Dump]:
    """create unicorn emulator from w1dump file"""
    log = get_logger("w1dump.loader")
    log.dbg(f"creating unicorn emulator from {dump_path}")

    # load dump and create executable
    dump = load_w1dump(dump_path)
    exe = W1DumpExecutable(dump, module_name)

    # create emulator with built-in lazy loading
    emu = UnicornEmulator(exe, hooks)

    # load initial register state
    loader = W1DumpLoader(dump)
    loader.load_registers_from_dump(emu)

    log.dbg("unicorn emulator created successfully")
    return emu, dump


def create_w1dump_triton_emulator(
    dump_path: str, module_name: Optional[str] = None, hooks: Hook = Hook.DEFAULT
) -> Tuple[TritonEmulator, W1Dump]:
    """create triton emulator from w1dump file"""
    log = get_logger("w1dump.loader")
    log.dbg(f"creating triton emulator from {dump_path}")

    # load dump and create executable
    dump = load_w1dump(dump_path)
    exe = W1DumpExecutable(dump, module_name)

    # create emulator with built-in lazy loading
    emu = TritonEmulator(exe, hooks)

    # load initial register state
    loader = W1DumpLoader(dump)
    loader.load_registers_from_dump(emu)

    log.dbg("triton emulator created successfully")
    return emu, dump


def create_w1dump_maat_emulator(
    dump_path: str, module_name: Optional[str] = None, hooks: Hook = Hook.DEFAULT
) -> Tuple[MaatEmulator, W1Dump]:
    log = get_logger("w1dump.loader")
    log.dbg(f"creating maat emulator from {dump_path}")

    # load dump and create executable
    dump = load_w1dump(dump_path)
    exe = W1DumpExecutable(dump, module_name)

    # create emulator with built-in lazy loading
    emu = MaatEmulator(exe, hooks)

    # load initial register state
    loader = W1DumpLoader(dump)
    loader.load_registers_from_dump(emu)

    log.dbg("maat emulator created successfully")
    return emu, dump
