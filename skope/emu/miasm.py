#!/usr/bin/env python3
"""
Miasm emulator using proper jitter for bare metal dump loading
follows the skope pattern for easy dump loading
"""

from typing import Dict, List, Optional, Any, Tuple, Callable, Union
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

# miasm imports
from miasm.analysis.machine import Machine
from miasm.core.locationdb import LocationDB
from miasm.jitter.csts import PAGE_READ, PAGE_WRITE, PAGE_EXEC


class MiasmJitterEmulator(BareMetalEmulator):
    """miasm emulator using jitter for bare metal dump loading"""
    
    def __init__(self, executable: Executable):
        """initialize the miasm jitter emulator"""
        self.log = get_logger("miasm.jitter")
        
        # setup architecture
        arch_map = {
            Arch.X86: "x86_32",
            Arch.X64: "x86_64",
            Arch.ARM64: "aarch64l",  # little-endian ARM64
        }
        
        if executable.arch not in arch_map:
            raise EmulationError(f"unsupported architecture: {executable.arch}")
        
        arch_name = arch_map[executable.arch]
        
        # create miasm machine and jitter
        self._loc_db = LocationDB()
        self._machine = Machine(arch_name)
        self._jitter = self._machine.jitter(self._loc_db, "python")  # use python jitter for flexibility
        
        # initialize stack
        self._jitter.init_stack()
        
        # memory management for lazy loading
        self._memory_callbacks = []
        self._loaded_segments = set()
        
        # call parent constructor which calls _setup()
        super().__init__(executable)
        
        self.log.dbg("miasm jitter emulator initialized", field("arch", arch_name))
    
    def _setup(self) -> None:
        """initialize the emulator - called by parent constructor"""
        # load initial memory segments from executable
        self._load_initial_memory()
    
    def _load_initial_memory(self):
        """load memory segments from executable"""
        for segment in self.exe.get_segments():
            self.map_memory(segment.address, segment.size, segment.permissions)
            if segment.data:
                self.mem_write(segment.address, segment.data)
            self._loaded_segments.add((segment.address, segment.size))
            self.log.dbg(f"loaded segment @ 0x{segment.address:x} size=0x{segment.size:x}")
    
    def _permission_to_miasm(self, perm: Permission) -> int:
        """convert skope permission to miasm permission"""
        miasm_perm = 0
        if perm & Permission.READ:
            miasm_perm |= PAGE_READ
        if perm & Permission.WRITE:
            miasm_perm |= PAGE_WRITE
        if perm & Permission.EXECUTE:
            miasm_perm |= PAGE_EXEC
        return miasm_perm
    
    # abstract method implementations
    def _map(self, address: int, size: int, permissions: Permission) -> None:
        """internal method to map memory in the engine"""
        miasm_perm = self._permission_to_miasm(permissions)
        
        # align size to page boundary
        page_size = 0x1000
        aligned_size = ((size + page_size - 1) // page_size) * page_size
        
        # allocate memory in jitter
        self._jitter.vm.add_memory_page(address, miasm_perm, b"\x00" * aligned_size)
    
    def _get_pc_reg(self) -> str:
        """get the program counter register name for this backend"""
        return get_pc_register(self.exe.arch)
    
    def _get_sp_reg(self) -> str:
        """get the stack pointer register name for this backend"""
        return get_sp_register(self.exe.arch)
    
    # memory operations
    def map_memory(self, address: int, size: int, permissions: Permission) -> None:
        """map memory with generic Permission enum"""
        self._map(address, size, permissions)
        self.log.dbg(f"mapped memory @ 0x{address:x} size=0x{size:x} perm={permissions}")
    
    def mem_unmap(self, address: int, size: int) -> None:
        """unmap memory region"""
        self._jitter.vm.remove_memory_page(address)
        self.log.dbg(f"unmapped memory @ 0x{address:x}")
    
    def mem_read(self, address: int, size: int) -> bytes:
        """read memory"""
        # check for lazy loading
        if not self._is_memory_mapped(address):
            self._lazy_load_memory(address, size)
        
        try:
            return self._jitter.vm.get_mem(address, size)
        except Exception as e:
            raise EmulationError(f"memory read failed @ 0x{address:x}: {e}")
    
    def mem_write(self, address: int, data: bytes) -> None:
        """write memory"""
        # check for lazy loading
        if not self._is_memory_mapped(address):
            self._lazy_load_memory(address, len(data))
        
        try:
            self._jitter.vm.set_mem(address, data)
        except Exception as e:
            raise EmulationError(f"memory write failed @ 0x{address:x}: {e}")
    
    def _is_memory_mapped(self, address: int) -> bool:
        """check if memory is mapped"""
        try:
            # try to read one byte
            self._jitter.vm.get_mem(address, 1)
            return True
        except:
            return False
    
    def _lazy_load_memory(self, address: int, size: int):
        """lazy load memory from callbacks"""
        for callback in self._memory_callbacks:
            region = callback(address, size)
            if region:
                self.map_memory(region.address, region.size, region.permissions)
                if region.data:
                    self.mem_write(region.address, region.data)
                self.log.dbg(f"lazy loaded region @ 0x{region.address:x} size=0x{region.size:x}")
                return
    
    # register operations
    def reg_read(self, reg_id: Union[str, int]) -> int:
        """read register value"""
        if isinstance(reg_id, str):
            reg_name = self._normalize_register_name(reg_id)
        else:
            # assume it's already a normalized name
            reg_name = reg_id
        
        try:
            return getattr(self._jitter.cpu, reg_name)
        except AttributeError:
            raise EmulationError(f"unknown register: {reg_name}")
    
    def reg_write(self, reg_id: Union[str, int], value: int) -> None:
        """write register value"""
        if isinstance(reg_id, str):
            reg_name = self._normalize_register_name(reg_id)
        else:
            # assume it's already a normalized name
            reg_name = reg_id
        
        try:
            setattr(self._jitter.cpu, reg_name, value)
        except AttributeError:
            raise EmulationError(f"unknown register: {reg_name}")
    
    def get_reg_by_name(self, name: str) -> int:
        """read register by name (architecture-independent)"""
        return self.reg_read(name)
    
    def set_reg_by_name(self, name: str, value: int) -> None:
        """write register by name (architecture-independent)"""
        self.reg_write(name, value)
    
    def _normalize_register_name(self, name: str) -> str:
        """normalize register name for miasm"""
        # miasm uses uppercase for ARM64 registers
        if self.exe.arch == Arch.ARM64:
            # handle special aliases
            if name.lower() == "sp":
                return "SP"
            elif name.lower() == "lr":
                return "LR"
            elif name.lower() == "pc":
                return "PC"
            # general purpose registers
            elif name.lower().startswith("x"):
                return name.upper()
            elif name.lower().startswith("w"):
                return name.upper()
        elif self.exe.arch == Arch.X64:
            # miasm uses uppercase for x64 registers
            return name.upper()
        elif self.exe.arch == Arch.X86:
            # miasm uses uppercase for x86 registers  
            return name.upper()
        return name
    
    # execution
    def emulate(self, start: Optional[int] = None, end: Optional[int] = None, 
                count: int = 0, timeout: int = 0) -> Optional[int]:
        """emulate instructions"""
        if start is not None:
            pc_name = get_pc_register(self.exe.arch)
            setattr(self._jitter.cpu, pc_name, start)
        
        # setup breakpoint at end address if specified
        if end is not None:
            self._jitter.add_breakpoint(end, lambda j: False)
        
        # setup instruction count limit
        if count > 0:
            instr_count = [0]
            def count_callback(jitter):
                instr_count[0] += 1
                return instr_count[0] < count
            self._jitter.exec_cb = count_callback
        
        try:
            # run jitter
            self._jitter.run(start if start is not None else self.get_pc())
            
            # return current pc
            pc_name = get_pc_register(self.exe.arch)
            return getattr(self._jitter.cpu, pc_name)
            
        except Exception as e:
            raise EmulationError(f"emulation failed: {e}")
    
    def halt(self) -> None:
        """stop emulation"""
        self._jitter.running = False
    
    def get_pc(self) -> int:
        """get current program counter"""
        pc_name = get_pc_register(self.exe.arch)
        return getattr(self._jitter.cpu, pc_name)
    
    def set_pc(self, value: int) -> None:
        """set program counter"""
        pc_name = get_pc_register(self.exe.arch)
        setattr(self._jitter.cpu, pc_name, value)
    
    # expose underlying components for external use
    def get_jitter(self):
        """get the underlying jitter for direct access"""
        return self._jitter
    
    def get_machine(self):
        """get the underlying machine for direct access"""
        return self._machine
    
    def get_loc_db(self):
        """get the location database"""
        return self._loc_db
    
    # memory callbacks for lazy loading
    def add_memory_callback(self, callback: Callable) -> None:
        """add memory region callback for lazy loading"""
        self._memory_callbacks.append(callback)
    
    # state management
    def reset(self) -> None:
        """reset emulator state"""
        # reset cpu registers to zero
        for attr in dir(self._jitter.cpu):
            if not attr.startswith("_"):
                try:
                    setattr(self._jitter.cpu, attr, 0)
                except:
                    pass
    
    def dump_state(self) -> Dict[str, Any]:
        """dump current emulator state"""
        state = {
            "pc": self.get_pc(),
            "registers": {},
        }
        
        # dump all registers
        for attr in dir(self._jitter.cpu):
            if attr.startswith("_"):
                continue
            try:
                val = getattr(self._jitter.cpu, attr)
                if isinstance(val, int):
                    state["registers"][attr] = val
            except:
                pass
        
        return state