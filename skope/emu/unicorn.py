#!/usr/bin/env python3
"""
unicorn engine implementation of the emulator interface
"""

from typing import Dict, List, Optional, Any
from redlog import get_logger, field

from .base import BareMetalEmulator, Executable, Hook, Arch, Segment, Permission
from .arch import get_pc_register

try:
    import unicorn as uc

    HAS_UNICORN = True
except ImportError:
    HAS_UNICORN = False
    uc = None


class UnicornEmulator(BareMetalEmulator):
    """unicorn engine implementation"""

    def __init__(self, executable: Executable, hooks: Hook = Hook.DEFAULT):
        if not HAS_UNICORN:
            raise RuntimeError("unicorn engine is not installed")

        self._uc: Optional[uc.Uc] = None
        self._arch_consts: Dict[str, Any] = {}

        super().__init__(executable, hooks)

    def _setup(self) -> None:
        """initialize unicorn engine"""
        # map architecture to unicorn constants
        arch_map = {
            Arch.X86: (uc.UC_ARCH_X86, uc.UC_MODE_32),
            Arch.X64: (uc.UC_ARCH_X86, uc.UC_MODE_64),
            Arch.ARM64: (uc.UC_ARCH_ARM64, uc.UC_MODE_ARM),
        }

        uc_arch, uc_mode = arch_map[self.exe.arch]

        # create unicorn instance
        if self.exe.arch == Arch.ARM64:
            # for arm64, use the most capable cpu model
            import unicorn.arm64_const

            self._uc = uc.Uc(uc_arch, uc_mode, unicorn.arm64_const.UC_CPU_ARM64_MAX)
            self._arch_consts = unicorn.arm64_const
        else:
            # add little endian for x86/x64
            uc_mode |= uc.UC_MODE_LITTLE_ENDIAN
            self._uc = uc.Uc(uc_arch, uc_mode)

            if self.exe.arch in (Arch.X86, Arch.X64):
                import unicorn.x86_const

                self._arch_consts = unicorn.x86_const

        # load segments
        self._load_segments()

        # set up memory layout
        self._setup_memory_layout()

        # install hooks
        self._install_hooks()

    def _permission_to_unicorn(self, perm: Permission) -> int:
        """convert our permission flags to unicorn format"""
        result = 0
        if perm & Permission.READ:
            result |= uc.UC_PROT_READ
        if perm & Permission.WRITE:
            result |= uc.UC_PROT_WRITE
        if perm & Permission.EXECUTE:
            result |= uc.UC_PROT_EXEC
        return result

    def _load_segments(self) -> None:
        """load all segments from the executable"""
        segments = self.exe.get_segments()
        self.log.trc(f"loading {len(segments)} segments")
        for i, seg in enumerate(segments):
            self.log.dbg(
                f"loading segment {i} @ addr=0x{seg.address:x} size={seg.size} perms={seg.permissions}"
            )
            self.map_memory(seg.address, seg.size, seg.permissions)
            self.mem_write(seg.address, seg.data)
        self.log.dbg("all segments loaded")

    def _setup_memory_layout(self) -> None:
        """set up heap and stack"""
        self.log.dbg("setting up memory layout")

        # find the end of loaded image
        image_end = self._get_image_end()
        self.log.dbg(f"image end @ 0x{image_end:x} (aligned)")

        # place heap after image
        self.heap_base = image_end
        self.heap_size = 0x100000  # 1mb
        self.log.dbg(
            f"mapping heap @ 0x{self.heap_base:x} (size: 0x{self.heap_size:x})"
        )
        self.map_memory(self.heap_base, self.heap_size, Permission.RW)

        # place stack at conventional location
        if self.exe.arch.bits == 64:
            self.stack_base = 0x7FFFFF000000  # high memory
        else:
            self.stack_base = 0xBFFFF000

        self.stack_size = 0x100000  # 1mb
        self.log.dbg(
            f"mapping stack @ 0x{self.stack_base:x} (size: 0x{self.stack_size:x})"
        )
        self.map_memory(self.stack_base, self.stack_size, Permission.RW)

        # initialize sp to top of stack
        self.sp = self.stack_base + self.stack_size
        self.log.dbg(f"stack pointer initialized to 0x{self.sp:x}")

    def _install_hooks(self) -> None:
        """install requested hooks"""
        if self.hooks & Hook.CODE_EXECUTE:
            self._uc.hook_add(uc.UC_HOOK_CODE, self._hook_code, self.state)

        if self.hooks & Hook.MEMORY_READ:
            self._uc.hook_add(uc.UC_HOOK_MEM_READ, self._hook_mem_read, self.state)

        if self.hooks & Hook.MEMORY_WRITE:
            self._uc.hook_add(uc.UC_HOOK_MEM_WRITE, self._hook_mem_write, self.state)

        if self.hooks & Hook.MEMORY_ERROR:
            self._uc.hook_add(uc.UC_HOOK_MEM_INVALID, self._hook_mem_error, self.state)

    def _hook_code(self, uc, address, size, user_data):
        return self.hook_code_execute(address, size)

    def _hook_mem_read(self, uc, access, address, size, value, user_data):
        return self.hook_memory_read(address, size, value)

    def _hook_mem_write(self, uc, access, address, size, value, user_data):
        return self.hook_memory_write(address, size, value)

    def _hook_mem_error(self, uc, access, address, size, value, user_data):
        return self.hook_memory_error(access, address, size, value)

    def hook_memory_error(
        self, access: int, address: int, size: int, value: int
    ) -> bool:
        """override to support lazy loading"""
        # try lazy loading first
        if self._handle_memory_fault(address):
            return True
        # fall back to base implementation
        return super().hook_memory_error(access, address, size, value)

    def _map(self, address: int, size: int, permissions: Permission) -> None:
        uc_perms = self._permission_to_unicorn(permissions)
        self._uc.mem_map(address, size, uc_perms)

    def mem_read(self, address: int, size: int) -> bytes:
        self.log.ped(f"memory read @ 0x{address:x} (size: {size})")
        return self._uc.mem_read(address, size)

    def mem_write(self, address: int, data: bytes) -> None:
        self.log.ped(f"memory write @ 0x{address:x} (size: {len(data)})")
        self._uc.mem_write(address, data)

    def reg_read(self, reg_id: int) -> int:
        return self._uc.reg_read(reg_id)

    def reg_write(self, reg_id: int, value: int) -> None:
        self._uc.reg_write(reg_id, value)

    def _get_pc_reg(self) -> int:
        """get unicorn register constant for program counter"""
        if self.exe.arch == Arch.ARM64:
            return self._arch_consts.UC_ARM64_REG_PC
        elif self.exe.arch == Arch.X64:
            return self._arch_consts.UC_X86_REG_RIP
        elif self.exe.arch == Arch.X86:
            return self._arch_consts.UC_X86_REG_EIP
        else:
            raise ValueError(f"unsupported architecture: {self.exe.arch}")

    def _get_sp_reg(self) -> int:
        """get unicorn register constant for stack pointer"""
        if self.exe.arch == Arch.ARM64:
            return self._arch_consts.UC_ARM64_REG_SP
        elif self.exe.arch == Arch.X64:
            return self._arch_consts.UC_X86_REG_RSP
        elif self.exe.arch == Arch.X86:
            return self._arch_consts.UC_X86_REG_ESP
        else:
            raise ValueError(f"unsupported architecture: {self.exe.arch}")

    def _get_reg_by_name(self, name: str) -> int:
        """get unicorn register constant by name"""
        name_upper = name.upper()

        if self.exe.arch == Arch.ARM64:
            # handle special cases
            if name == "sp":
                return self._arch_consts.UC_ARM64_REG_SP
            elif name == "pc":
                return self._arch_consts.UC_ARM64_REG_PC
            # general purpose registers
            reg_attr = f"UC_ARM64_REG_{name_upper}"
            if hasattr(self._arch_consts, reg_attr):
                return getattr(self._arch_consts, reg_attr)

        elif self.exe.arch in (Arch.X64, Arch.X86):
            reg_attr = f"UC_X86_REG_{name_upper}"
            if hasattr(self._arch_consts, reg_attr):
                return getattr(self._arch_consts, reg_attr)

        raise ValueError(f"unknown register {name} for {self.exe.arch}")

    def get_reg_by_name(self, name: str) -> int:
        """get register value by name"""
        reg_id = self._get_reg_by_name(name)
        return self.reg_read(reg_id)

    def set_reg_by_name(self, name: str, value: int) -> None:
        """set register value by name"""
        reg_id = self._get_reg_by_name(name)
        self.reg_write(reg_id, value)

    def set_pc(self, value: int) -> None:
        """set program counter value"""
        pc_name = get_pc_register(self.exe.arch)
        self.set_reg_by_name(pc_name, value)

    def get_pc(self) -> int:
        """get program counter value"""
        pc_name = get_pc_register(self.exe.arch)
        return self.get_reg_by_name(pc_name)

    def emulate(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        count: int = 0,
        timeout: int = 0,
    ) -> None:
        """run emulation"""
        if start is None:
            start = self.pc

        self._uc.emu_start(start, end or 0, timeout * 1000, count)

    def halt(self) -> None:
        """stop emulation"""
        self._uc.emu_stop()
