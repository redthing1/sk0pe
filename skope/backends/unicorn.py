from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.arch import Architecture
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
    import unicorn as uc
    import unicorn.x86_const as uc_x86
    import unicorn.arm64_const as uc_arm64

    HAS_UNICORN = True
except ImportError:  # pragma: no cover
    HAS_UNICORN = False
    uc = None
    uc_x86 = None
    uc_arm64 = None


class UnicornEmulator(Emulator):
    """Unicorn-backed emulator implementation."""

    def __init__(self, snapshot: Snapshot, config: Optional[EmulatorConfig] = None):
        if not HAS_UNICORN:
            raise RuntimeError("unicorn is not installed")

        super().__init__(snapshot, config)

        self._uc: uc.Uc = self._create_engine(snapshot.arch)
        self._mapped: List[MemoryRegion] = []
        self._page_size = 0x1000

        self._hooks: Dict[HookType, List[Tuple[int, HookCallback]]] = {
            HookType.CODE: [],
            HookType.MEM_READ: [],
            HookType.MEM_WRITE: [],
            HookType.MEM_ERROR: [],
        }
        self._hook_counter = 1
        self._installed_hooks: Dict[HookType, Optional[int]] = {
            HookType.CODE: None,
            HookType.MEM_READ: None,
            HookType.MEM_WRITE: None,
            HookType.MEM_ERROR: None,
        }

        self._executed = 0
        self._count_enabled = False
        self._stop_reason: Optional[str] = None

        self._install_core_hooks()
        self._load_snapshot()
        self._load_registers()

    @property
    def backend(self) -> Any:
        return self._uc

    @property
    def capabilities(self) -> Set[str]:
        return {"emulation"}

    def read(self, address: int, size: int) -> bytes:
        return bytes(self._uc.mem_read(address, size))

    def write(self, address: int, data: bytes) -> None:
        self._uc.mem_write(address, data)

    def reg_read(self, name: str) -> int:
        reg_id = self._reg_const(name)
        return int(self._uc.reg_read(reg_id))

    def reg_write(self, name: str, value: int) -> None:
        reg_id = self._reg_const(name)
        self._uc.reg_write(reg_id, int(value))

    def map(self, address: int, size: int, permissions: int) -> None:
        if size <= 0:
            return
        start = self._align_down(address)
        end = self._align_up(address + size)
        if self.is_mapped(start, end - start):
            return
        uc_perms = self._to_uc_perms(MemoryPermissions(permissions))
        self._uc.mem_map(start, end - start, uc_perms)
        self._mapped.append(
            MemoryRegion(
                start=start, end=end, permissions=MemoryPermissions(permissions)
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
        self._executed = 0
        self._count_enabled = True
        self._stop_reason = None

        start_addr = start if start is not None else self.pc
        end_addr = end if end is not None else 0

        try:
            self._uc.emu_start(start_addr, end_addr, timeout * 1000, count)
        except uc.UcError as exc:
            if self._stop_reason is None:
                self._stop_reason = "error"
            return ExecutionResult(
                executed=self._executed,
                stop_reason=self._stop_reason,
                pc=self.pc,
                error=str(exc),
            )
        finally:
            self._count_enabled = False

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
        self._uc.emu_stop()

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

    def _create_engine(self, arch: Architecture) -> uc.Uc:
        if arch == Architecture.X86:
            return uc.Uc(uc.UC_ARCH_X86, uc.UC_MODE_32 | uc.UC_MODE_LITTLE_ENDIAN)
        if arch == Architecture.X64:
            return uc.Uc(uc.UC_ARCH_X86, uc.UC_MODE_64 | uc.UC_MODE_LITTLE_ENDIAN)
        if arch == Architecture.ARM64:
            return uc.Uc(uc.UC_ARCH_ARM64, uc.UC_MODE_ARM, uc_arm64.UC_CPU_ARM64_MAX)
        raise ValueError(f"unsupported architecture: {arch}")

    def _install_core_hooks(self) -> None:
        self._installed_hooks[HookType.CODE] = self._uc.hook_add(
            uc.UC_HOOK_CODE, self._on_code
        )
        self._installed_hooks[HookType.MEM_ERROR] = self._uc.hook_add(
            uc.UC_HOOK_MEM_INVALID, self._on_mem_error
        )

    def _ensure_hook_installed(self, hook_type: HookType) -> None:
        if hook_type in (HookType.CODE, HookType.MEM_ERROR):
            return
        if self._installed_hooks[hook_type] is not None:
            return

        if hook_type == HookType.MEM_READ:
            self._installed_hooks[hook_type] = self._uc.hook_add(
                uc.UC_HOOK_MEM_READ, self._on_mem_read
            )
        elif hook_type == HookType.MEM_WRITE:
            self._installed_hooks[hook_type] = self._uc.hook_add(
                uc.UC_HOOK_MEM_WRITE, self._on_mem_write
            )

    def _on_code(self, _uc, address, size, _user_data):
        if self._count_enabled:
            self._executed += 1
        for _, callback in self._hooks[HookType.CODE]:
            if not callback(address, size):
                self._stop_reason = "hook"
                self.stop()
                break
        return True

    def _on_mem_read(self, _uc, _access, address, size, value, _user_data):
        for _, callback in self._hooks[HookType.MEM_READ]:
            if not callback(address, size, value):
                self._stop_reason = "hook"
                self.stop()
                break
        return True

    def _on_mem_write(self, _uc, _access, address, size, value, _user_data):
        for _, callback in self._hooks[HookType.MEM_WRITE]:
            if not callback(address, size, value):
                self._stop_reason = "hook"
                self.stop()
                break
        return True

    def _on_mem_error(self, _uc, access, address, size, value, _user_data):
        handled = self._handle_memory_fault(address)
        for _, callback in self._hooks[HookType.MEM_ERROR]:
            result = callback(access, address, size, value)
            if result:
                handled = True
            elif result is False:
                handled = handled
        if handled:
            return True
        self._stop_reason = "mem_error"
        return False

    def _handle_memory_fault(self, address: int) -> bool:
        region = self.snapshot.memory.get_region(address)
        if not region:
            return False
        if not self.is_mapped(region.start, region.size):
            self.map(region.start, region.size, int(region.permissions))
        if region.data:
            self.write(region.start, region.data)
        return True

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

    def _load_registers(self) -> None:
        for name, value in self.snapshot.registers.as_dict().items():
            try:
                self.reg_write(name, value)
            except ValueError:
                continue

    def _reg_const(self, name: str) -> int:
        reg_name = name.lower()
        if reg_name == "pc":
            reg_name = self.arch.pc_name
        if reg_name == "sp":
            reg_name = self.arch.sp_name

        if self.arch in (Architecture.X86, Architecture.X64):
            attr = f"UC_X86_REG_{reg_name.upper()}"
            if not hasattr(uc_x86, attr):
                raise ValueError(f"unknown register: {name}")
            return getattr(uc_x86, attr)

        if self.arch == Architecture.ARM64:
            attr = f"UC_ARM64_REG_{reg_name.upper()}"
            if hasattr(uc_arm64, attr):
                return getattr(uc_arm64, attr)
            if reg_name in ("nzcv", "cpsr"):
                if hasattr(uc_arm64, "UC_ARM64_REG_NZCV"):
                    return uc_arm64.UC_ARM64_REG_NZCV
                if hasattr(uc_arm64, "UC_ARM64_REG_PSTATE"):
                    return uc_arm64.UC_ARM64_REG_PSTATE
            raise ValueError(f"unknown register: {name}")

        raise ValueError(f"unsupported architecture: {self.arch}")

    def _to_uc_perms(self, perms: MemoryPermissions) -> int:
        result = 0
        if perms & MemoryPermissions.READ:
            result |= uc.UC_PROT_READ
        if perms & MemoryPermissions.WRITE:
            result |= uc.UC_PROT_WRITE
        if perms & MemoryPermissions.EXECUTE:
            result |= uc.UC_PROT_EXEC
        return result

    def _align_down(self, value: int) -> int:
        return value & ~(self._page_size - 1)

    def _align_up(self, value: int) -> int:
        return (value + self._page_size - 1) & ~(self._page_size - 1)


def _find_main_module(modules: List[dict]) -> Optional[str]:
    for module in modules:
        if module.get("type") == "main_executable":
            return module.get("name")
    if len(modules) == 1:
        return modules[0].get("name")
    return None
