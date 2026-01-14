from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.arch import Architecture
from ..core.errors import EmulationError
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
    import maat

    HAS_MAAT = True
except ImportError:  # pragma: no cover
    HAS_MAAT = False
    maat = None


class MaatEmulator(Emulator):
    """Maat-backed emulator implementation."""

    def __init__(self, snapshot: Snapshot, config: Optional[EmulatorConfig] = None):
        if not HAS_MAAT:
            raise RuntimeError("maat is not installed")

        super().__init__(snapshot, config)

        self._engine = self._create_engine(snapshot)
        self._mapped: List[MemoryRegion] = []
        self._hooks: Dict[HookType, List[Tuple[int, HookCallback]]] = {
            HookType.CODE: [],
            HookType.MEM_READ: [],
            HookType.MEM_WRITE: [],
            HookType.MEM_ERROR: [],
        }
        self._hook_counter = 1
        self._hook_installed: Dict[HookType, bool] = {
            HookType.CODE: False,
            HookType.MEM_READ: False,
            HookType.MEM_WRITE: False,
            HookType.MEM_ERROR: False,
        }
        self._executed = 0
        self._stop_reason: Optional[str] = None
        self._snapshot_loader_installed = False
        self._snapshot_loader_symbolize = False
        self._snapshot_loader_verbose = 0
        self._snapshot_loader_symbolized: Set[int] = set()

        self._load_snapshot()
        self._load_registers()
        if self.config.load_strategy == "lazy":
            self.enable_snapshot_memory()

    @property
    def backend(self) -> Any:
        return self._engine

    @property
    def capabilities(self) -> Set[str]:
        return {"emulation", "symbolic"}

    def read(self, address: int, size: int) -> bytes:
        result = bytearray()
        for offset in range(size):
            value = self._engine.mem.read(address + offset, 1)
            result.append(_to_int(value, self._engine.vars) & 0xFF)
        return bytes(result)

    def write(self, address: int, data: bytes) -> None:
        for offset, byte in enumerate(data):
            self._engine.mem.write(address + offset, byte, 1)

    def reg_read(self, name: str) -> int:
        reg_name = self._normalize_reg(name)
        try:
            value = getattr(self._engine.cpu, reg_name)
        except AttributeError as exc:
            raise ValueError(f"unknown register: {name}") from exc
        return _to_int(value, self._engine.vars)

    def reg_write(self, name: str, value: int) -> None:
        reg_name = self._normalize_reg(name)
        try:
            setattr(self._engine.cpu, reg_name, int(value))
        except AttributeError as exc:
            raise ValueError(f"unknown register: {name}") from exc

    def map(self, address: int, size: int, permissions: int) -> None:
        if size <= 0:
            return
        self._engine.mem.map(address, address + size)
        self._mapped.append(
            MemoryRegion(
                start=address,
                end=address + size,
                permissions=MemoryPermissions(permissions),
            )
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
        if timeout:
            raise ValueError("timeout is not supported by the Maat adapter")

        if start is not None:
            self.pc = start

        self._executed = 0
        self._stop_reason = None

        if count > 0:
            self._engine.run(count)
            self._executed = count
        elif end is not None:
            max_insns = 100000
            while self.pc != end and self._executed < max_insns:
                self._engine.run(1)
                self._executed += 1
            if self._executed >= max_insns:
                raise EmulationError("maat run exceeded safety limit")
        else:
            default_count = 1000
            self._engine.run(default_count)
            self._executed = default_count

        stop_reason = self._stop_reason
        if stop_reason is None:
            if count and self._executed >= count:
                stop_reason = "count"
            elif end is not None and self.pc == end:
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
        # Maat does not expose an explicit stop; rely on hooks.

    def enable_snapshot_memory(
        self, *, symbolize_loads: bool = False, verbose: int = 0
    ) -> None:
        if not self._snapshot_loader_installed:
            self._engine.hooks.add(
                maat.EVENT.EXEC,
                maat.WHEN.BEFORE,
                callbacks=[self._on_snapshot_exec],
                data=[],
                name="skope_snapshot_exec",
            )
            self._engine.hooks.add(
                maat.EVENT.MEM_R,
                maat.WHEN.BEFORE,
                callbacks=[self._on_snapshot_mem_access],
                data=[],
                name="skope_snapshot_mem_read",
            )
            if hasattr(maat.EVENT, "MEM_W"):
                self._engine.hooks.add(
                    maat.EVENT.MEM_W,
                    maat.WHEN.BEFORE,
                    callbacks=[self._on_snapshot_mem_access],
                    data=[],
                    name="skope_snapshot_mem_write",
                )
            if hasattr(maat.EVENT, "MEM_RW"):
                self._engine.hooks.add(
                    maat.EVENT.MEM_RW,
                    maat.WHEN.BEFORE,
                    callbacks=[self._on_snapshot_mem_access],
                    data=[],
                    name="skope_snapshot_mem_rw",
                )
            self._snapshot_loader_installed = True

        if symbolize_loads:
            self._snapshot_loader_symbolize = True
        if verbose > self._snapshot_loader_verbose:
            self._snapshot_loader_verbose = verbose

    def _ensure_snapshot_memory(self, address: int, size: int) -> bool:
        if size <= 0:
            return True
        end = address + size
        current = address
        while current < end:
            region = self.snapshot.memory.get_region(current)
            if not region:
                return False
            if not self.is_mapped(region.start, region.size):
                self._map_region(region)
            current = max(current + 1, region.end)
        return True

    def _on_snapshot_exec(self, engine, _data):
        addr = engine.info.addr
        if addr == 0:
            return maat.ACTION.CONTINUE
        if not self._ensure_snapshot_memory(addr, 1):
            if self._snapshot_loader_verbose:
                print(f"[skope] maat snapshot loader missing code at 0x{addr:x}")
        return maat.ACTION.CONTINUE

    def _on_snapshot_mem_access(self, engine, _data):
        if not engine.info.mem_access:
            return maat.ACTION.CONTINUE
        try:
            addr = _to_int(engine.info.mem_access.addr, engine.vars)
        except Exception:
            return maat.ACTION.CONTINUE
        size = int(engine.info.mem_access.size)
        if not self._ensure_snapshot_memory(addr, size):
            if self._snapshot_loader_verbose:
                print(
                    f"[skope] maat snapshot loader missing data at 0x{addr:x} ({size} bytes)"
                )
            return maat.ACTION.CONTINUE

        if self._snapshot_loader_symbolize and hasattr(engine.mem, "make_concolic"):
            for offset in range(size):
                addr_i = addr + offset
                if addr_i in self._snapshot_loader_symbolized:
                    continue
                try:
                    engine.mem.make_concolic(addr_i, 1, 1, f"mem_{addr_i:x}")
                    self._snapshot_loader_symbolized.add(addr_i)
                except Exception:
                    continue
        return maat.ACTION.CONTINUE

    def _add_hook(self, hook_type: HookType, callback: HookCallback) -> HookHandle:
        handle = HookHandle(self._hook_counter, hook_type)
        self._hook_counter += 1
        self._hooks[hook_type].append((handle.hook_id, callback))
        self._ensure_hook_installed(hook_type)
        return handle

    def _remove_hook(self, handle: HookHandle) -> None:
        callbacks = self._hooks.get(handle.hook_type, [])
        self._hooks[handle.hook_type] = [
            entry for entry in callbacks if entry[0] != handle.hook_id
        ]

    def _ensure_hook_installed(self, hook_type: HookType) -> None:
        if self._hook_installed.get(hook_type):
            return

        if hook_type == HookType.CODE:
            self._engine.hooks.add(
                maat.EVENT.EXEC,
                maat.WHEN.BEFORE,
                callbacks=[self._on_code],
                data=[],
                name="skope_code",
            )
            self._hook_installed[hook_type] = True
        elif hook_type == HookType.MEM_READ:
            self._engine.hooks.add(
                maat.EVENT.MEM_R,
                maat.WHEN.BEFORE,
                callbacks=[self._on_mem_read],
                data=[],
                name="skope_mem_read",
            )
            self._hook_installed[hook_type] = True
        elif hook_type == HookType.MEM_WRITE:
            self._engine.hooks.add(
                maat.EVENT.MEM_W,
                maat.WHEN.BEFORE,
                callbacks=[self._on_mem_write],
                data=[],
                name="skope_mem_write",
            )
            self._hook_installed[hook_type] = True
        elif hook_type == HookType.MEM_ERROR:
            self._hook_installed[hook_type] = True

    def _on_code(self, engine, data):
        addr = engine.info.addr
        if addr == 0:
            return maat.ACTION.CONTINUE
        size = 1
        try:
            asm_inst = engine.get_asm_inst(addr)
            size = asm_inst.raw_size()
        except Exception:
            size = 4 if self.arch == Architecture.ARM64 else 1
        for _, callback in self._hooks[HookType.CODE]:
            if not callback(addr, size):
                self._stop_reason = "hook"
                return maat.ACTION.HALT
        return maat.ACTION.CONTINUE

    def _on_mem_read(self, engine, data):
        if engine.info.mem_access:
            try:
                addr = _to_int(engine.info.mem_access.addr, engine.vars)
                size = engine.info.mem_access.size
                value = _to_int(engine.info.mem_access.value, engine.vars)
            except Exception:
                return maat.ACTION.CONTINUE
            for _, callback in self._hooks[HookType.MEM_READ]:
                if not callback(addr, size, value):
                    self._stop_reason = "hook"
                    return maat.ACTION.HALT
        return maat.ACTION.CONTINUE

    def _on_mem_write(self, engine, data):
        if engine.info.mem_access:
            try:
                addr = _to_int(engine.info.mem_access.addr, engine.vars)
                size = engine.info.mem_access.size
                value = _to_int(engine.info.mem_access.value, engine.vars)
            except Exception:
                return maat.ACTION.CONTINUE
            for _, callback in self._hooks[HookType.MEM_WRITE]:
                if not callback(addr, size, value):
                    self._stop_reason = "hook"
                    return maat.ACTION.HALT
        return maat.ACTION.CONTINUE

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
        if region.size <= 0:
            return
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

    def _normalize_reg(self, name: str) -> str:
        reg_name = name.lower()
        if reg_name == "pc":
            return self.arch.pc_name
        if reg_name == "sp":
            return self.arch.sp_name
        return reg_name

    def _create_engine(self, snapshot: Snapshot):
        arch = _arch_to_maat(snapshot.arch)
        os_value = _os_to_maat(snapshot.platform)
        if os_value is None:
            return maat.MaatEngine(arch)
        return maat.MaatEngine(arch, os_value)


def _arch_to_maat(arch: Architecture):
    if arch == Architecture.X86:
        return maat.ARCH.X86
    if arch == Architecture.X64:
        return maat.ARCH.X64
    if arch == Architecture.ARM64:
        return maat.ARCH.ARM64
    raise ValueError(f"unsupported architecture: {arch}")


def _os_to_maat(platform: str):
    if not hasattr(maat, "OS"):
        return None
    os_enum = maat.OS
    if not platform:
        return getattr(os_enum, "LINUX", None)
    value = platform.lower()
    if "win" in value and hasattr(os_enum, "WINDOWS"):
        return os_enum.WINDOWS
    if ("mac" in value or "darwin" in value) and hasattr(os_enum, "MACOS"):
        return os_enum.MACOS
    if hasattr(os_enum, "LINUX"):
        return os_enum.LINUX
    return None


def _find_main_module(modules: List[dict]) -> Optional[str]:
    for module in modules:
        if module.get("type") == "main_executable":
            return module.get("name")
    if len(modules) == 1:
        return modules[0].get("name")
    return None


def _to_int(value: Any, varctx: Optional[Any] = None) -> int:
    if hasattr(value, "as_uint"):
        if varctx is not None:
            return int(value.as_uint(varctx))
        return int(value.as_uint())
    return int(value)
