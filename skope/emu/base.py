#!/usr/bin/env python3
"""
clean, abstract emulator interface with proper architecture support
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, IntFlag
from typing import Dict, List, Optional, Tuple, Callable, Any
from functools import lru_cache
from redlog import get_logger, field


class SkopeError(Exception):
    """base exception for skope library"""
    pass


class EmulationError(SkopeError):
    """emulation operation failed"""
    pass


class MemoryError(SkopeError):
    """memory operation failed"""
    pass


class ArchitectureError(SkopeError):
    """unsupported architecture or architecture-specific error"""
    pass


class Arch(Enum):
    """supported architectures"""

    X86 = "x86"
    X64 = "x64"
    ARM64 = "arm64"

    @property
    def bits(self) -> int:
        """bit width of the architecture"""
        return 64 if self in (Arch.X64, Arch.ARM64) else 32

    @property
    def pointer_size(self) -> int:
        """size of a pointer in bytes"""
        return 8 if self.bits == 64 else 4


class Permission(IntFlag):
    """memory permissions (architecture-independent)"""

    NONE = 0
    READ = 0b001
    WRITE = 0b010
    EXECUTE = 0b100

    # common combinations
    RW = READ | WRITE
    RX = READ | EXECUTE
    RWX = READ | WRITE | EXECUTE

    @classmethod
    def from_rwx(cls, read: bool, write: bool, execute: bool) -> "Permission":
        """create from individual flags"""
        perm = cls.NONE
        if read:
            perm |= cls.READ
        if write:
            perm |= cls.WRITE
        if execute:
            perm |= cls.EXECUTE
        return perm


class Hook(IntFlag):
    """event hooks available during emulation"""

    CODE_EXECUTE = 0b0001
    MEMORY_READ = 0b0010
    MEMORY_WRITE = 0b0100
    MEMORY_ERROR = 0b1000

    DEFAULT = MEMORY_ERROR
    ALL = CODE_EXECUTE | MEMORY_READ | MEMORY_WRITE | MEMORY_ERROR


@dataclass
class MemoryRegion:
    """a mapped memory region"""

    address: int
    size: int
    permissions: Permission

    @property
    def end_address(self) -> int:
        return self.address + self.size

    def contains(self, address: int, size: int = 1) -> bool:
        """check if this region contains the given address range"""
        return self.address <= address and (address + size) <= self.end_address


@dataclass
class Segment:
    """a segment of code/data to be loaded"""

    address: int
    data: bytes
    permissions: Permission = Permission.RWX

    @property
    def size(self) -> int:
        return len(self.data)


class Executable(ABC):
    """abstract base for executable formats"""

    @property
    @abstractmethod
    def arch(self) -> Arch:
        """the architecture of this executable"""
        pass

    @property
    @abstractmethod
    def base_address(self) -> int:
        """the base address where this executable should be loaded"""
        pass

    @abstractmethod
    def get_segments(self) -> List[Segment]:
        """get all segments that need to be loaded"""
        pass

    def get_memory_region(
        self, address: int
    ) -> Optional[Tuple[int, bytes, Permission]]:
        """get memory region containing address, for lazy loading.
        returns (start_address, data, permissions) or None"""
        return None

    def get_initial_state(self) -> Optional[Dict[str, Any]]:
        """get initial cpu state if available"""
        return None


@dataclass
class RawCodeBlob(Executable):
    """simple executable from raw bytes"""

    data: bytes
    arch: Arch
    base_address: int = 0x10000

    def get_segments(self) -> List[Segment]:
        # code is readable and executable
        return [Segment(self.base_address, self.data, Permission.RX)]


class Emulator(ABC):
    """pure abstract emulator interface"""

    @abstractmethod
    def mem_read(self, address: int, size: int) -> bytes:
        """read memory"""
        pass

    @abstractmethod
    def mem_write(self, address: int, data: bytes) -> None:
        """write memory"""
        pass

    @abstractmethod
    def reg_read(self, reg_id: Any) -> int:
        """read a register by its backend-specific identifier"""
        pass

    @abstractmethod
    def reg_write(self, reg_id: Any, value: int) -> None:
        """write a register by its backend-specific identifier"""
        pass

    @abstractmethod
    def emulate(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        count: int = 0,
        timeout: int = 0,
    ) -> Any:
        """run emulation"""
        pass

    @abstractmethod
    def halt(self) -> None:
        """stop emulation"""
        pass

    @abstractmethod
    def set_pc(self, value: int) -> None:
        """set program counter value (architecture-independent)"""
        pass

    @abstractmethod
    def get_pc(self) -> int:
        """get program counter value (architecture-independent)"""
        pass

    @abstractmethod
    def get_reg_by_name(self, name: str) -> int:
        """read register by name (architecture-independent)"""
        pass

    @abstractmethod
    def set_reg_by_name(self, name: str, value: int) -> None:
        """write register by name (architecture-independent)"""
        pass

    @abstractmethod
    def map_memory(self, address: int, size: int, permissions: Permission) -> None:
        """map memory with generic Permission enum"""
        pass


class BareMetalEmulator(Emulator):
    """abstract emulator with common bare-metal functionality"""

    def __init__(self, executable: Executable, hooks: Hook = Hook.DEFAULT):
        self.exe = executable
        self.hooks = hooks
        self.state: Any = None  # user-defined state for hooks
        self._memory_map: List[MemoryRegion] = []
        self.log = get_logger("emulator")

        # memory layout
        self.heap_base: int = 0
        self.heap_size: int = 0
        self.stack_base: int = 0
        self.stack_size: int = 0

        # page size for alignment
        self.page_size: int = 0x1000

        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        """initialize the emulator"""
        pass

    def _align_down(self, value: int) -> int:
        """align value down to page boundary"""
        return value & ~(self.page_size - 1)

    def _align_up(self, value: int) -> int:
        """align value up to page boundary"""
        return (value + self.page_size - 1) & ~(self.page_size - 1)

    @abstractmethod
    def _map(self, address: int, size: int, permissions: Permission) -> None:
        """internal method to map memory in the engine"""
        pass

    def map_memory(self, address: int, size: int, permissions: Permission) -> None:
        """map a memory region with automatic alignment"""
        if size <= 0:
            return

        # align to page boundaries
        aligned_addr = self._align_down(address)
        end_addr = self._align_up(address + size)
        aligned_size = end_addr - aligned_addr

        # check if already mapped
        if self.is_mapped(aligned_addr, aligned_size):
            self.log.dbg(f"region already mapped: 0x{aligned_addr:x}-0x{end_addr:x}")
            return

        # map in engine
        self.log.dbg(
            f"mapping region: 0x{aligned_addr:x}-0x{end_addr:x} (perms: {permissions})"
        )
        self._map(aligned_addr, aligned_size, permissions)

        # track the mapping
        self._memory_map.append(MemoryRegion(aligned_addr, aligned_size, permissions))

    def is_mapped(self, address: int, size: int = 1) -> bool:
        """check if a memory range is fully mapped"""
        end_addr = address + size

        # check if the entire range is covered by mapped regions
        current = address
        while current < end_addr:
            found = False
            for region in self._memory_map:
                if region.contains(current):
                    current = region.end_address
                    found = True
                    break
            if not found:
                return False
        return True

    @abstractmethod
    def _get_pc_reg(self) -> Any:
        """get the program counter register identifier for this backend"""
        pass

    @abstractmethod
    def _get_sp_reg(self) -> Any:
        """get the stack pointer register identifier for this backend"""
        pass

    @property
    def pc(self) -> int:
        """program counter"""
        return self.reg_read(self._get_pc_reg())

    @pc.setter
    def pc(self, value: int) -> None:
        self.reg_write(self._get_pc_reg(), value)

    @property
    def sp(self) -> int:
        """stack pointer"""
        return self.reg_read(self._get_sp_reg())

    @sp.setter
    def sp(self, value: int) -> None:
        self.reg_write(self._get_sp_reg(), value)

    # hook methods - can be overridden
    def hook_code_execute(self, address: int, size: int) -> bool:
        """called when code is executed"""
        return True

    def hook_memory_read(self, address: int, size: int, value: int) -> bool:
        """called when memory is read"""
        return True

    def hook_memory_write(self, address: int, size: int, value: int) -> bool:
        """called when memory is written"""
        return True

    def hook_memory_error(
        self, access: int, address: int, size: int, value: int
    ) -> bool:
        """called on memory access error. return true to retry."""
        return False

    def _get_image_end(self) -> int:
        """get the end address of loaded image segments"""
        segments = self.exe.get_segments()
        if segments:
            image_end = max(seg.address + seg.size for seg in segments)
        else:
            image_end = self.exe.base_address + self.page_size

        return self._align_up(image_end)

    def _handle_memory_fault(self, address: int) -> bool:
        """try to resolve memory fault by asking executable"""
        if region := self.exe.get_memory_region(address):
            start, data, perms = region
            self.map_memory(start, len(data), perms)
            self.mem_write(start, data)
            return True
        return False
