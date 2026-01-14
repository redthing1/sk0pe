from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.arch import Architecture
from ..core.errors import EmulationError, MemoryError
from ..core.memory import MemoryRegion
from ..core.permissions import MemoryPermissions
from ..snapshot.base import Snapshot
from .base import (
    Emulator,
    EmulatorConfig,
    ExecutionResult,
    HookHandle,
    HookType,
    HookCallback,
)

try:
    from triton import (
        TritonContext,
        ARCH,
        MODE,
        Instruction,
        EXCEPTION,
    )

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False
    TritonContext = None
    ARCH = None
    MODE = None
    Instruction = None
    EXCEPTION = None


class TritonEmulator(Emulator):
    """Triton-backed emulator implementation."""

    def __init__(self, snapshot: Snapshot, config: Optional[EmulatorConfig] = None):
        if not HAS_TRITON:
            raise RuntimeError("triton is not installed")

        super().__init__(snapshot, config)

        self._ctx = TritonContext()
        self._ctx.setArchitecture(_arch_to_triton(snapshot.arch))
        self._ctx.setMode(MODE.ALIGNED_MEMORY, True)
        self._ctx.setMode(MODE.CONSTANT_FOLDING, True)
        self._ctx.setMode(MODE.AST_OPTIMIZATIONS, True)
        self._ctx.setMode(MODE.ONLY_ON_SYMBOLIZED, True)

        self._mapped: List[MemoryRegion] = []
        self._hooks: Dict[HookType, List[Tuple[int, HookCallback]]] = {
            HookType.CODE: [],
            HookType.MEM_READ: [],
            HookType.MEM_WRITE: [],
            HookType.MEM_ERROR: [],
        }
        self._hook_counter = 1
        self._executed = 0
        self._stop_reason: Optional[str] = None
        self._halted = False
        self._snapshot_loader_installed = False
        self._snapshot_loader_symbolize = False
        self._snapshot_loader_verbose = 0
        self._snapshot_symbolize_handle: Optional[HookHandle] = None

        self._load_snapshot()
        self._load_registers()
        if self.config.load_strategy == "lazy":
            self.enable_snapshot_memory()

    @property
    def backend(self) -> Any:
        return self._ctx

    @property
    def capabilities(self) -> Set[str]:
        return {"emulation", "symbolic"}

    def read(self, address: int, size: int) -> bytes:
        if not self._ensure_memory(
            address, size, eager_region=not self._snapshot_loader_symbolize
        ):
            raise MemoryError(f"memory not available at 0x{address:x}")
        result = bytearray()
        for offset in range(size):
            result.append(self._ctx.getConcreteMemoryValue(address + offset))
        return bytes(result)

    def write(self, address: int, data: bytes) -> None:
        self._ctx.setConcreteMemoryAreaValue(address, list(data))

    def reg_read(self, name: str) -> int:
        reg = self._get_register(name)
        return int(self._ctx.getConcreteRegisterValue(reg))

    def reg_write(self, name: str, value: int) -> None:
        reg = self._get_register(name)
        self._ctx.setConcreteRegisterValue(reg, int(value))

    def map(self, address: int, size: int, permissions: int) -> None:
        if size <= 0:
            return
        perms = MemoryPermissions(permissions)
        self._mapped.append(
            MemoryRegion(start=address, end=address + size, permissions=perms)
        )

    def is_mapped(self, address: int, size: int = 1) -> bool:
        if size <= 0:
            return False
        end = address + size
        current = address
        for region in self._mapped:
            if region.end <= current:
                continue
            if region.start > current:
                return False
            if region.contains(current, end - current):
                return True
            if region.contains(current, 1):
                current = region.end
            if current >= end:
                return True
        return False

    def run(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        count: int = 0,
        timeout: int = 0,
    ) -> ExecutionResult:
        self._executed = 0
        self._stop_reason = None
        self._halted = False

        if timeout:
            raise ValueError("timeout is not supported by the Triton adapter")

        pc = start if start is not None else self.pc

        while True:
            if self._halted:
                break
            if pc == 0:
                self._stop_reason = "pc_zero"
                break
            if end is not None and pc >= end:
                break
            if count and self._executed >= count:
                break

            inst_bytes = self.read(pc, 16)
            inst = Instruction()
            inst.setOpcode(inst_bytes)
            inst.setAddress(pc)

            try:
                fault = self._ctx.processing(inst)
            except Exception as exc:
                raise EmulationError(f"error processing instruction at 0x{pc:x}: {exc}")

            if fault == EXCEPTION.FAULT_UD:
                raise EmulationError(f"undefined instruction at 0x{pc:x}")
            if fault != EXCEPTION.NO_FAULT:
                raise EmulationError(f"emulation fault {fault} at 0x{pc:x}")

            self._executed += 1

            if self._hooks[HookType.CODE]:
                for _, callback in self._hooks[HookType.CODE]:
                    if not callback(pc, inst.getSize()):
                        self._stop_reason = "hook"
                        self._halted = True
                        break

            for access in inst.getLoadAccess():
                mem_access = access[0] if isinstance(access, tuple) else access
                value = (
                    mem_access.getValue()
                    if hasattr(mem_access, "getValue")
                    else access[1]
                    if isinstance(access, tuple) and len(access) > 1
                    else 0
                )
                for _, callback in self._hooks[HookType.MEM_READ]:
                    if not callback(
                        mem_access.getAddress(),
                        mem_access.getSize(),
                        value,
                    ):
                        self._stop_reason = "hook"
                        self._halted = True
                        break

            for access in inst.getStoreAccess():
                mem_access = access[0] if isinstance(access, tuple) else access
                value = (
                    mem_access.getValue()
                    if hasattr(mem_access, "getValue")
                    else access[1]
                    if isinstance(access, tuple) and len(access) > 1
                    else 0
                )
                for _, callback in self._hooks[HookType.MEM_WRITE]:
                    if not callback(
                        mem_access.getAddress(),
                        mem_access.getSize(),
                        value,
                    ):
                        self._stop_reason = "hook"
                        self._halted = True
                        break

            pc = self.pc

        stop_reason = self._stop_reason
        if stop_reason is None:
            if count and self._executed >= count:
                stop_reason = "count"
            elif end is not None and self.pc >= end:
                stop_reason = "end"
            else:
                stop_reason = "halt"

        return ExecutionResult(
            executed=self._executed,
            stop_reason=stop_reason,
            pc=self.pc,
            error=None,
        )

    def stop(self) -> None:
        if self._stop_reason is None:
            self._stop_reason = "halt"
        self._halted = True

    def enable_snapshot_memory(
        self, *, symbolize_loads: bool = False, verbose: int = 0
    ) -> None:
        from triton import CALLBACK

        if not self._snapshot_loader_installed:
            self._ctx.addCallback(
                CALLBACK.GET_CONCRETE_MEMORY_VALUE, self._on_snapshot_memory
            )
            self._snapshot_loader_installed = True

        if symbolize_loads:
            self._snapshot_loader_symbolize = True
            self._ctx.setMode(MODE.ONLY_ON_SYMBOLIZED, False)
        if verbose > self._snapshot_loader_verbose:
            self._snapshot_loader_verbose = verbose

        if symbolize_loads and self._snapshot_symbolize_handle is None:
            self._snapshot_symbolize_handle = self.hooks.add(
                HookType.MEM_READ, self._on_snapshot_symbolic_read
            )

    def _on_snapshot_memory(self, ctx, mem) -> None:
        from triton import MemoryAccess

        addr = mem.getAddress()
        size = mem.getSize()

        if not self._ensure_memory(
            addr, size, eager_region=not self._snapshot_loader_symbolize
        ):
            if self._snapshot_loader_verbose:
                print(
                    f"[skope] triton snapshot loader missing data at 0x{addr:x} ({size} bytes)"
                )
            return

        if not self._snapshot_loader_symbolize:
            return

        for i in range(size):
            addr_i = addr + i
            if not ctx.isMemorySymbolized(addr_i):
                ctx.symbolizeMemory(MemoryAccess(addr_i, 1), f"mem_{addr_i:x}")

    def _on_snapshot_symbolic_read(self, address: int, size: int, _value: int) -> bool:
        from triton import MemoryAccess

        if not self._snapshot_loader_symbolize:
            return True
        for i in range(size):
            addr_i = address + i
            if not self._ctx.isMemorySymbolized(addr_i):
                self._ctx.symbolizeMemory(MemoryAccess(addr_i, 1), f"mem_{addr_i:x}")
        return True

    def _add_hook(self, hook_type: HookType, callback: HookCallback) -> HookHandle:
        handle = HookHandle(self._hook_counter, hook_type)
        self._hook_counter += 1
        self._hooks[hook_type].append((handle.hook_id, callback))
        return handle

    def _remove_hook(self, handle: HookHandle) -> None:
        callbacks = self._hooks.get(handle.hook_type, [])
        self._hooks[handle.hook_type] = [
            entry for entry in callbacks if entry[0] != handle.hook_id
        ]

    def _load_snapshot(self) -> None:
        strategy = self.config.load_strategy
        if strategy == "all_regions":
            for region in self.snapshot.memory_map.iter_regions():
                self._map_region(region)
        elif strategy == "module_only":
            main_module = _find_main_module(self.snapshot.modules)
            for region in self.snapshot.memory_map.iter_regions():
                if not main_module or region.module_name == main_module:
                    self._map_region(region)
        elif strategy == "lazy":
            pass
        else:
            raise ValueError(f"unknown load strategy: {strategy}")

        self._ensure_stack_mapped()
        self._ensure_zero_page()

    def _map_region(self, region: MemoryRegion) -> None:
        self.map(region.start, region.size, int(region.permissions))
        if region.data:
            self.write(region.start, region.data)

    def _ensure_stack_mapped(self) -> None:
        if self.config.stack_base is not None:
            self.map(
                self.config.stack_base,
                self.config.stack_size,
                int(MemoryPermissions.RW),
            )
            return
        sp = self.snapshot.registers.get(self.arch.sp_name)
        if sp == 0:
            return
        region = self.snapshot.memory_map.find(sp)
        if not region:
            return
        if not self.is_mapped(region.start, region.size):
            self._map_region(region)

    def _ensure_zero_page(self) -> None:
        if self.config.map_zero_page is not True:
            return
        size = self.config.zero_page_size
        if size <= 0:
            return
        if self.is_mapped(0, size):
            return
        self.map(0, size, int(self.config.zero_page_permissions))
        self.write(0, bytes(size))

    def _load_registers(self) -> None:
        for name, value in self.snapshot.registers.as_dict().items():
            try:
                self.reg_write(name, value)
            except ValueError:
                continue

    def _ensure_memory(
        self, address: int, size: int, *, eager_region: bool = True
    ) -> bool:
        if size <= 0:
            return True

        if eager_region:
            for offset in range(size):
                addr = address + offset
                if self._ctx.isConcreteMemoryValueDefined(addr, 1):
                    continue
                region = self.snapshot.memory.get_region(addr)
                if not region:
                    return False
                if not self.is_mapped(region.start, region.size):
                    self._map_region(region)
                if region.data:
                    self.write(region.start, region.data)
            return True

        data = self.snapshot.memory.read(address, size)
        if data is not None:
            region = self.snapshot.memory.get_region(address)
            if region and not self.is_mapped(region.start, region.size):
                self.map(region.start, region.size, int(region.permissions))
            self._ctx.setConcreteMemoryAreaValue(address, list(data))
            return True

        for offset in range(size):
            addr = address + offset
            if self._ctx.isConcreteMemoryValueDefined(addr, 1):
                continue
            data = self.snapshot.memory.read(addr, 1)
            if data is None:
                return False
            region = self.snapshot.memory.get_region(addr)
            if region and not self.is_mapped(region.start, region.size):
                self.map(region.start, region.size, int(region.permissions))
            self._ctx.setConcreteMemoryAreaValue(addr, list(data))
        return True

    def _get_register(self, name: str):
        reg_name = name.lower()
        if reg_name == "pc":
            reg_name = self.arch.pc_name
        if reg_name == "sp":
            reg_name = self.arch.sp_name
        reg = getattr(self._ctx.registers, reg_name, None)
        if not reg:
            raise ValueError(f"unknown register: {name}")
        return reg


def _arch_to_triton(arch: Architecture):
    if arch == Architecture.X86:
        return ARCH.X86
    if arch == Architecture.X64:
        return ARCH.X86_64
    if arch == Architecture.ARM64:
        return ARCH.AARCH64
    raise ValueError(f"unsupported architecture: {arch}")


def _find_main_module(modules: List[dict]) -> Optional[str]:
    for module in modules:
        if module.get("type") == "main_executable":
            return module.get("name")
    if len(modules) == 1:
        return modules[0].get("name")
    return None
