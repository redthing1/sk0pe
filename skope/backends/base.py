from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Set

from ..core.arch import Architecture
from ..core.permissions import MemoryPermissions
from ..snapshot.base import Snapshot


class HookType(Enum):
    CODE = "code"
    MEM_READ = "mem_read"
    MEM_WRITE = "mem_write"
    MEM_ERROR = "mem_error"


HookCallback = Callable[..., bool]


@dataclass(frozen=True)
class HookHandle:
    hook_id: int
    hook_type: HookType


@dataclass
class ExecutionResult:
    executed: int
    stop_reason: str
    pc: int
    error: Optional[str] = None


@dataclass
class EmulatorConfig:
    load_strategy: str = "lazy"
    stack_base: Optional[int] = None
    stack_size: int = 0x100000
    heap_base: Optional[int] = None
    heap_size: int = 0x100000
    map_zero_page: Optional[bool] = None
    zero_page_size: int = 0x1000
    zero_page_permissions: MemoryPermissions = MemoryPermissions.RW


class HookManager:
    def __init__(self, emulator: "Emulator"):
        self._emu = emulator

    def add(self, hook_type: HookType, callback: HookCallback) -> HookHandle:
        return self._emu._add_hook(hook_type, callback)

    def remove(self, handle: HookHandle) -> None:
        self._emu._remove_hook(handle)


class Emulator(ABC):
    """Abstract emulator interface used by backend adapters."""

    def __init__(self, snapshot: Snapshot, config: Optional[EmulatorConfig] = None):
        self.snapshot = snapshot
        self.config = config or EmulatorConfig()
        self.hooks = HookManager(self)

    @property
    def arch(self) -> Architecture:
        return self.snapshot.arch

    @property
    def pc(self) -> int:
        return self.reg_read(self.arch.pc_name)

    @pc.setter
    def pc(self, value: int) -> None:
        self.reg_write(self.arch.pc_name, value)

    @property
    def sp(self) -> int:
        return self.reg_read(self.arch.sp_name)

    @sp.setter
    def sp(self, value: int) -> None:
        self.reg_write(self.arch.sp_name, value)

    @property
    @abstractmethod
    def backend(self) -> Any:
        """Return the underlying backend engine instance."""

    @property
    @abstractmethod
    def capabilities(self) -> Set[str]:
        """Return a set of capability identifiers for this backend."""

    @abstractmethod
    def read(self, address: int, size: int) -> bytes:
        """Read bytes from emulated memory."""

    @abstractmethod
    def write(self, address: int, data: bytes) -> None:
        """Write bytes into emulated memory."""

    @abstractmethod
    def reg_read(self, name: str) -> int:
        """Read a register by name."""

    @abstractmethod
    def reg_write(self, name: str, value: int) -> None:
        """Write a register by name."""

    @abstractmethod
    def map(self, address: int, size: int, permissions: int) -> None:
        """Map a memory region."""

    @abstractmethod
    def is_mapped(self, address: int, size: int = 1) -> bool:
        """Return True if a memory range is mapped."""

    @abstractmethod
    def run(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        count: int = 0,
        timeout: int = 0,
    ) -> ExecutionResult:
        """Run emulation and return an execution summary."""

    @abstractmethod
    def stop(self) -> None:
        """Stop emulation."""

    def enable_snapshot_memory(
        self, *, symbolize_loads: bool = False, verbose: int = 0
    ) -> None:
        """Enable snapshot-backed memory loading for lazy snapshots."""
        return None

    @abstractmethod
    def _add_hook(self, hook_type: HookType, callback: HookCallback) -> HookHandle:
        """Register a hook callback and return its handle."""

    @abstractmethod
    def _remove_hook(self, handle: HookHandle) -> None:
        """Remove a hook callback by handle."""
