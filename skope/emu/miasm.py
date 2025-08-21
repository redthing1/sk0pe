#!/usr/bin/env python3
"""
Miasm symbolic execution engine implementation of the emulator interface
"""

from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass
from functools import lru_cache
from enum import Enum
from redlog import get_logger, field

from .base import (
    BareMetalEmulator,
    Executable,
    Hook,
    Arch,
    Segment,
    Permission,
    EmulationError,
)
from .arch import get_pc_register, get_sp_register

import miasm

try:
    from miasm.core.locationdb import LocationDB
    from miasm.arch.aarch64.arch import mn_aarch64
    from miasm.arch.aarch64.sem import Lifter_Aarch64l
    from miasm.arch.x86.arch import mn_x86
    from miasm.arch.x86.sem import Lifter_X86_32, Lifter_X86_64
    from miasm.ir.symbexec import SymbolicExecutionEngine
    from miasm.expression.expression import *
    from miasm.core.utils import decode_hex
    HAS_MIASM = True
except ImportError as e:
    HAS_MIASM = False
    LocationDB = None
    mn_aarch64 = None
    Lifter_Aarch64l = None
    mn_x86 = None
    Lifter_X86_32 = None
    Lifter_X86_64 = None
    SymbolicExecutionEngine = None


@dataclass
class RegisterSymbolicInfo:
    """information about a register's symbolic state"""
    name: str
    concrete_value: Optional[int]
    is_symbolic: bool
    symbolic_expr: Optional[str] = None
    comment: Optional[str] = None


class MiasmEmulator(BareMetalEmulator):
    """miasm symbolic execution engine implementation"""

    def __init__(self, executable: Executable, hooks: Hook = Hook.DEFAULT):
        if not HAS_MIASM:
            raise RuntimeError("miasm is not installed")

        self._loc_db: Optional[LocationDB] = None
        self._lifter = None
        self._symb_engine: Optional[SymbolicExecutionEngine] = None
        self._mn = None  # machine module
        self._regs = None  # register module
        self._pc_reg = None
        self._sp_reg = None
        self._memory_lazy_load_callback: Optional[Callable] = None
        self._instruction_cache: Dict[int, Any] = {}
        
        super().__init__(executable, hooks)

    def _setup(self) -> None:
        """initialize Miasm components"""
        self.log.dbg("setting up miasm emulator", field("arch", self.exe.arch))
        
        # create location database
        self._loc_db = LocationDB()
        
        # initialize architecture-specific components
        if self.exe.arch == Arch.ARM64:
            self._mn = mn_aarch64
            self._lifter = Lifter_Aarch64l(self._loc_db)
            self._regs = mn_aarch64.regs
            self._pc_reg = self._regs.PC
            self._sp_reg = self._regs.SP
        elif self.exe.arch == Arch.X64:
            self._mn = mn_x86
            self._lifter = Lifter_X86_64(self._loc_db)
            self._regs = mn_x86.regs
            self._pc_reg = self._regs.RIP
            self._sp_reg = self._regs.RSP
        elif self.exe.arch == Arch.X86:
            self._mn = mn_x86
            self._lifter = Lifter_X86_32(self._loc_db)
            self._regs = mn_x86.regs
            self._pc_reg = self._regs.EIP
            self._sp_reg = self._regs.ESP
        else:
            raise ValueError(f"unsupported architecture: {self.exe.arch}")

        # create symbolic execution engine
        self._symb_engine = SymbolicExecutionEngine(self._lifter)
        
        self.log.dbg("miasm emulator setup complete")

    def mem_read(self, address: int, size: int) -> bytes:
        """read memory from emulator"""
        try:
            # try to get concrete memory from symbolic engine
            data = bytearray()
            for i in range(size):
                addr = address + i
                mem_expr = ExprMem(ExprInt(addr, self.exe.arch.bits), 8)
                
                if mem_expr in self._symb_engine.symbols:
                    val_expr = self._symb_engine.symbols[mem_expr]
                    if isinstance(val_expr, ExprInt):
                        data.append(val_expr.arg)
                    else:
                        # symbolic memory, try lazy load
                        if self._memory_lazy_load_callback:
                            self._memory_lazy_load_callback(addr)
                            # retry after callback
                            if mem_expr in self._symb_engine.symbols:
                                val_expr = self._symb_engine.symbols[mem_expr]
                                if isinstance(val_expr, ExprInt):
                                    data.append(val_expr.arg)
                                else:
                                    data.append(0)  # default for symbolic
                            else:
                                data.append(0)
                        else:
                            data.append(0)  # default for symbolic
                else:
                    # memory not defined, try lazy load
                    if self._memory_lazy_load_callback:
                        self._memory_lazy_load_callback(addr)
                        # retry after callback
                        mem_expr = ExprMem(ExprInt(addr, self.exe.arch.bits), 8)
                        if mem_expr in self._symb_engine.symbols:
                            val_expr = self._symb_engine.symbols[mem_expr]
                            if isinstance(val_expr, ExprInt):
                                data.append(val_expr.arg)
                            else:
                                data.append(0)
                        else:
                            data.append(0)
                    else:
                        data.append(0)
            
            return bytes(data)
        except Exception as e:
            self.log.err(f"memory read failed at 0x{address:x}: {e}")
            raise EmulationError(f"memory read failed: {e}")

    def mem_write(self, address: int, data: bytes) -> None:
        """write memory to emulator"""
        try:
            for i, byte_val in enumerate(data):
                addr = address + i
                mem_expr = ExprMem(ExprInt(addr, self.exe.arch.bits), 8)
                val_expr = ExprInt(byte_val, 8)
                self._symb_engine.symbols[mem_expr] = val_expr
        except Exception as e:
            self.log.err(f"memory write failed at 0x{address:x}: {e}")
            raise EmulationError(f"memory write failed: {e}")

    def reg_read(self, reg_id: Any) -> int:
        """read register from emulator"""
        try:
            # convert reg_id to miasm register if needed
            if isinstance(reg_id, str):
                # handle different register name formats
                reg_name = reg_id.upper()
                if hasattr(self._regs, reg_name):
                    reg = getattr(self._regs, reg_name)
                else:
                    # try without prefix for arm64 (e.g., "X8" -> "8")
                    if reg_name.startswith('X') and reg_name[1:].isdigit():
                        reg = getattr(self._regs, reg_name[1:])
                    else:
                        raise AttributeError(f"register {reg_name} not found")
            else:
                reg = reg_id
            
            if reg in self._symb_engine.symbols:
                val_expr = self._symb_engine.symbols[reg]
                if isinstance(val_expr, ExprInt):
                    return val_expr.arg
                else:
                    # symbolic register - return 0 for now
                    # TODO: could evaluate with concrete model
                    return 0
            else:
                return 0
        except Exception as e:
            self.log.err(f"register read failed for {reg_id}: {e}")
            raise EmulationError(f"register read failed: {e}")

    def reg_write(self, reg_id: Any, value: int) -> None:
        """write register to emulator"""
        try:
            # convert reg_id to miasm register if needed
            if isinstance(reg_id, str):
                # handle different register name formats  
                reg_name = reg_id.upper()
                if hasattr(self._regs, reg_name):
                    reg = getattr(self._regs, reg_name)
                else:
                    # try without prefix for arm64 (e.g., "X8" -> "8") 
                    if reg_name.startswith('X') and reg_name[1:].isdigit():
                        reg_num = reg_name[1:]
                        reg = getattr(self._regs, reg_num)
                    elif reg_name == 'SP':
                        reg = getattr(self._regs, 'SP')
                    elif reg_name == 'PC':
                        reg = getattr(self._regs, 'PC')
                    else:
                        raise AttributeError(f"register {reg_name} not found")
            else:
                reg = reg_id
            
            val_expr = ExprInt(value, reg.size)
            self._symb_engine.symbols[reg] = val_expr
        except Exception as e:
            self.log.err(f"register write failed for {reg_id}: {e}")
            raise EmulationError(f"register write failed: {e}")

    def emulate(self, start: Optional[int] = None, end: Optional[int] = None, 
                count: int = 0, timeout: int = 0) -> Any:
        """emulate instructions using miasm symbolic execution"""
        if start is None:
            start = self.reg_read(self._pc_reg)
        
        self.log.dbg(f"starting emulation from 0x{start:x}")
        
        current_pc = start
        step_count = 0
        
        try:
            while True:
                # check termination conditions
                if end is not None and current_pc >= end:
                    break
                if count > 0 and step_count >= count:
                    break
                
                # read and lift instruction
                instruction_bytes = self.mem_read(current_pc, 4)  # assume 4-byte instructions
                
                # lift single instruction
                ircfg, next_pc = self._lift_and_execute_instruction(current_pc, instruction_bytes)
                
                if next_pc is None:
                    self.log.dbg("emulation stopped - no next PC")
                    break
                
                current_pc = next_pc
                step_count += 1
                
                self.log.dbg(f"emulation step {step_count}: pc=0x{current_pc:x}")
                
        except Exception as e:
            self.log.err(f"emulation failed at pc 0x{current_pc:x}: {e}")
            raise EmulationError(f"emulation failed: {e}")
        
        return current_pc

    def _lift_and_execute_instruction(self, pc: int, instruction_bytes: bytes) -> Tuple[Any, Optional[int]]:
        """lift and execute a single instruction"""
        
        # disassemble instruction
        if len(instruction_bytes) < 4:
            return None, None
            
        instr = self._mn.dis(instruction_bytes[:4], 'l' if self.exe.arch == Arch.ARM64 else 'b')
        if not instr:
            self.log.warn(f"failed to disassemble instruction at 0x{pc:x}: {instruction_bytes.hex()}")
            return None, None
        
        self.log.dbg(f"disassembled at 0x{pc:x}: {instr}")
        
        # create IR for this instruction
        ircfg = self._lifter.new_ircfg()
        
        # set instruction properties
        instr.offset = pc
        instr.l = 4  # ARM64 instructions are 4 bytes
        self._lifter.add_instr_to_ircfg(instr, ircfg)
        
        # execute the IR symbolically
        # run_at returns the address it stopped at (or None)
        stop_addr = self._symb_engine.run_at(ircfg, pc)
        
        # get next PC - after execution, PC should have advanced
        next_pc = None
        if self._pc_reg in self._symb_engine.symbols:
            pc_expr = self._symb_engine.symbols[self._pc_reg]
            if isinstance(pc_expr, ExprInt):
                next_pc = pc_expr.arg
            else:
                # PC is symbolic - could be a conditional jump
                self.log.dbg(f"PC is symbolic: {pc_expr}")
        
        # if PC hasn't changed, it means we need to advance it by instruction size
        if next_pc == pc:
            next_pc = pc + 4  # ARM64 instructions are 4 bytes
            self._symb_engine.symbols[self._pc_reg] = ExprInt(next_pc, self._pc_reg.size)
        
        return ircfg, next_pc

    def symbolize_register(self, reg_id: Any, name: str) -> None:
        """symbolize a register with given name"""
        try:
            if isinstance(reg_id, str):
                reg = getattr(self._regs, reg_id.upper())
            else:
                reg = reg_id
            
            sym_expr = ExprId(name, reg.size)
            self._symb_engine.symbols[reg] = sym_expr
            self.log.dbg(f"symbolized register {reg} as {name}")
        except Exception as e:
            self.log.err(f"register symbolization failed for {reg_id}: {e}")
            raise EmulationError(f"register symbolization failed: {e}")

    def symbolize_memory(self, address: int, size: int, name: str) -> None:
        """symbolize memory region with given name"""
        try:
            for i in range(size):
                addr = address + i
                mem_expr = ExprMem(ExprInt(addr, self.exe.arch.bits), 8)
                sym_expr = ExprId(f"{name}_{i}", 8)
                self._symb_engine.symbols[mem_expr] = sym_expr
            self.log.dbg(f"symbolized memory 0x{address:x}[{size}] as {name}")
        except Exception as e:
            self.log.err(f"memory symbolization failed at 0x{address:x}: {e}")
            raise EmulationError(f"memory symbolization failed: {e}")

    def get_symbolic_register(self, reg_id: Any) -> Optional[str]:
        """get symbolic expression for register"""
        try:
            if isinstance(reg_id, str):
                reg = getattr(self._regs, reg_id.upper())
            else:
                reg = reg_id
            
            if reg in self._symb_engine.symbols:
                return str(self._symb_engine.symbols[reg])
            return None
        except Exception as e:
            self.log.err(f"get symbolic register failed for {reg_id}: {e}")
            return None

    def get_symbolic_memory(self, address: int) -> Optional[str]:
        """get symbolic expression for memory"""
        try:
            mem_expr = ExprMem(ExprInt(address, self.exe.arch.bits), 8)
            if mem_expr in self._symb_engine.symbols:
                return str(self._symb_engine.symbols[mem_expr])
            return None
        except Exception as e:
            self.log.err(f"get symbolic memory failed at 0x{address:x}: {e}")
            return None

    def get_context(self) -> SymbolicExecutionEngine:
        """get the underlying symbolic execution engine"""
        return self._symb_engine

    def add_memory_callback(self, callback: Callable) -> None:
        """add callback for lazy memory loading"""
        self._memory_lazy_load_callback = callback

    def reset_symbolic_state(self) -> None:
        """reset all symbolic state"""
        # keep concrete memory but clear symbolic registers
        concrete_memory = {}
        for key, val in self._symb_engine.symbols.items():
            if isinstance(key, ExprMem) and isinstance(val, ExprInt):
                concrete_memory[key] = val
        
        self._symb_engine.symbols.clear()
        
        # restore concrete memory
        for dst, src in concrete_memory.items():
            self._symb_engine.symbols[dst] = src
            
        self.log.dbg("reset symbolic state, kept concrete memory")

    def dump_state(self, show_memory: bool = True, show_registers: bool = True) -> None:
        """dump current symbolic state"""
        if show_registers:
            self.log.info("=== Register State ===")
            for reg, expr in self._symb_engine.symbols.items():
                if not isinstance(reg, ExprMem):
                    self.log.info(f"  {reg}: {expr}")
        
        if show_memory:
            self.log.info("=== Memory State ===")
            for mem_expr, val_expr in self._symb_engine.symbols.items():
                if isinstance(mem_expr, ExprMem):
                    addr = mem_expr.ptr
                    if isinstance(addr, ExprInt):
                        self.log.info(f"  [0x{addr.arg:x}]: {val_expr}")
                    else:
                        self.log.info(f"  [{addr}]: {val_expr}")

    # Implement required abstract methods from BareMetalEmulator

    def halt(self) -> None:
        """stop emulation"""
        # miasm doesn't have a specific halt mechanism
        pass

    def set_pc(self, value: int) -> None:
        """set program counter value (architecture-independent)"""
        self.reg_write(self._pc_reg, value)

    def get_pc(self) -> int:
        """get program counter value (architecture-independent)"""
        return self.reg_read(self._pc_reg)

    def get_reg_by_name(self, name: str) -> int:
        """read register by name (architecture-independent)"""
        return self.reg_read(name)

    def set_reg_by_name(self, name: str, value: int) -> None:
        """write register by name (architecture-independent)"""
        self.reg_write(name, value)

    def map_memory(self, address: int, size: int, permissions: Permission) -> None:
        """map memory with generic Permission enum"""
        # miasm doesn't require explicit memory mapping like unicorn
        # memory is mapped on-demand when accessed
        pass

    def _map(self, address: int, size: int, permissions: Permission) -> None:
        """internal method to map memory in the engine"""
        # miasm doesn't require explicit memory mapping
        pass

    def _get_pc_reg(self) -> Any:
        """get the program counter register identifier for this backend"""
        return self._pc_reg

    def _get_sp_reg(self) -> Any:
        """get the stack pointer register identifier for this backend"""
        return self._sp_reg