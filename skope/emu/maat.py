#!/usr/bin/env python3
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
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

try:
    import maat

    HAS_MAAT = True
except ImportError:
    HAS_MAAT = False
    maat = None


class MaatEmulator(BareMetalEmulator):

    def __init__(self, executable: Executable, hooks: Hook = Hook.DEFAULT):
        if not HAS_MAAT:
            raise RuntimeError("maat is not installed")

        self._engine: Optional[maat.MaatEngine] = None
        self._symbolic_vars: Dict[str, Any] = {}
        self._memory_hooks: List[Callable] = []
        self._hook_callbacks: Dict[str, Callable] = {}

        super().__init__(executable, hooks)

    def _setup(self) -> None:
        # map architecture to Maat constants
        arch_map = {
            Arch.X86: maat.ARCH.X86,
            Arch.X64: maat.ARCH.X64,
            Arch.ARM64: maat.ARCH.ARM64,
        }

        maat_arch = arch_map.get(self.exe.arch)
        if not maat_arch:
            raise ValueError(f"unsupported architecture: {self.exe.arch}")

        # create maat engine
        self._engine = maat.MaatEngine(maat_arch)
        self.log.dbg(f"created maat engine for {self.exe.arch.name}")

        # load segments
        self._load_segments()

        # set up memory layout
        self._setup_memory_layout()

        # install hooks
        self._install_hooks()

    def _setup_memory_layout(self) -> None:
        self.log.dbg("setting up memory layout")

        # find the end of loaded image
        image_end = self._get_image_end()
        self.log.dbg(f"image end @ 0x{image_end:x} (aligned)")

        # place heap after image
        self.heap_base = image_end
        self.heap_size = 0x100000  # 1mb
        self.log.dbg(f"heap @ 0x{self.heap_base:x} (size: 0x{self.heap_size:x})")

        # place stack at conventional location
        if self.exe.arch.bits == 64:
            self.stack_base = 0x7FFFFF000000  # high memory
        else:
            self.stack_base = 0xBFFFF000

        self.stack_size = 0x100000  # 1mb
        self.log.dbg(f"stack @ 0x{self.stack_base:x} (size: 0x{self.stack_size:x})")

        # map stack region
        self.map_memory(self.stack_base, self.stack_size, Permission.RW)

        # initialize sp to top of stack
        self.sp = self.stack_base + self.stack_size
        self.log.dbg(f"stack pointer initialized to 0x{self.sp:x}")

    def _load_segments(self) -> None:
        # check if this is a W1DumpExecutable so we can access all regions
        if hasattr(self.exe, "dump"):
            # load ALL memory regions from the dump that have data
            regions = [r for r in self.exe.dump.regions if r.data]
            self.log.trc(f"loading {len(regions)} memory regions from dump")

            for i, region in enumerate(regions):
                self.log.dbg(
                    f"loading region {i} @ 0x{region.start:x} size={len(region.data)} "
                    f"perms={region.permissions} module={region.module_name}"
                )
                self.map_memory(region.start, len(region.data), region.permissions)
                self.mem_write(region.start, region.data)

            self.log.dbg(f"loaded all {len(regions)} memory regions from dump")
        else:
            # fallback to loading segments the normal way
            segments = self.exe.get_segments()
            self.log.trc(f"loading {len(segments)} segments")

            for i, seg in enumerate(segments):
                self.log.dbg(
                    f"loading segment {i} @ addr=0x{seg.address:x} size={seg.size} perms={seg.permissions}"
                )
                self.map_memory(seg.address, seg.size, seg.permissions)
                self.mem_write(seg.address, seg.data)

            self.log.dbg("all segments loaded")

    def _make_hook_wrapper(self, method_name):
        """create weakref wrapper to avoid reference cycles"""
        import weakref
        weak_self = weakref.ref(self)
        
        def wrapper(engine, data):
            self_ref = weak_self()
            if self_ref:
                return getattr(self_ref, method_name)(engine, data)
            return maat.ACTION.CONTINUE
        return wrapper

    def _install_hooks(self) -> None:
        self.log.dbg("installing hooks")
        
        if self.hooks & Hook.CODE_EXECUTE:
            self._engine.hooks.add(
                maat.EVENT.EXEC,
                maat.WHEN.BEFORE,
                callbacks=[self._make_hook_wrapper('_hook_code')],
                data=[],
                name="code_execute_hook"
            )
            
        if self.hooks & Hook.MEMORY_READ:
            self._engine.hooks.add(
                maat.EVENT.MEM_R,
                maat.WHEN.BEFORE,
                callbacks=[self._make_hook_wrapper('_hook_mem_read')],
                data=[],
                name="mem_read_hook"
            )
            
        if self.hooks & Hook.MEMORY_WRITE:
            self._engine.hooks.add(
                maat.EVENT.MEM_W,
                maat.WHEN.BEFORE,
                callbacks=[self._make_hook_wrapper('_hook_mem_write')],
                data=[],
                name="mem_write_hook"
            )

    def _hook_code(self, engine, data):
        # get current instruction address from engine
        addr = engine.info.addr
        # skip invalid addresses
        if addr == 0:
            return maat.ACTION.CONTINUE
        # get actual instruction size from maat
        try:
            asm_inst = engine.get_asm_inst(addr)
            size = asm_inst.raw_size()
        except:
            # fallback if instruction not available
            size = 4 if self.exe.arch == Arch.ARM64 else 1
        result = self.hook_code_execute(addr, size)
        return maat.ACTION.CONTINUE if result else maat.ACTION.HALT

    def _hook_mem_read(self, engine, data):
        if engine.info.mem_access:
            # get address (handle both concrete and symbolic)
            addr = engine.info.mem_access.addr.as_uint() if hasattr(engine.info.mem_access.addr, 'as_uint') else engine.info.mem_access.addr
            size = engine.info.mem_access.size
            # get actual value from maat
            try:
                value = engine.info.mem_access.value.as_uint() if hasattr(engine.info.mem_access.value, 'as_uint') else int(engine.info.mem_access.value)
            except:
                value = 0  # fallback if value not available
            result = self.hook_memory_read(addr, size, value)
            return maat.ACTION.CONTINUE if result else maat.ACTION.HALT
        return maat.ACTION.CONTINUE

    def _hook_mem_write(self, engine, data):
        if engine.info.mem_access:
            # get address (handle both concrete and symbolic)
            addr = engine.info.mem_access.addr.as_uint() if hasattr(engine.info.mem_access.addr, 'as_uint') else engine.info.mem_access.addr
            size = engine.info.mem_access.size
            # get actual value being written
            try:
                value = engine.info.mem_access.value.as_uint() if hasattr(engine.info.mem_access.value, 'as_uint') else int(engine.info.mem_access.value)
            except:
                value = 0  # fallback if value not available
            result = self.hook_memory_write(addr, size, value)
            return maat.ACTION.CONTINUE if result else maat.ACTION.HALT
        return maat.ACTION.CONTINUE

    def _map(self, address: int, size: int, permissions: Permission) -> None:
        # maat uses start/end addressing
        end_address = address + size
        self._engine.mem.map(address, end_address)
        self.log.dbg(f"mapped memory region 0x{address:x}-0x{end_address:x}")

    def mem_read(self, address: int, size: int) -> bytes:
        self.log.ped(f"memory read @ 0x{address:x} (size: {size})")
        result = bytearray()
        for i in range(size):
            byte_val = self._engine.mem.read(address + i, 1)
            # convert maat value to concrete byte
            if hasattr(byte_val, "as_uint"):
                result.append(byte_val.as_uint() & 0xFF)
            else:
                result.append(int(byte_val) & 0xFF)
        return bytes(result)

    def mem_write(self, address: int, data: bytes) -> None:
        self.log.ped(f"memory write @ 0x{address:x} (size: {len(data)})")
        for i, byte in enumerate(data):
            self._engine.mem.write(address + i, byte, 1)

    def reg_read(self, reg_id: str) -> int:
        try:
            value = getattr(self._engine.cpu, reg_id)
            if hasattr(value, "as_uint"):
                return value.as_uint()
            return int(value)
        except AttributeError:
            raise ValueError(f"unknown register: {reg_id}")

    def reg_write(self, reg_id: str, value: int) -> None:
        try:
            setattr(self._engine.cpu, reg_id, value)
        except AttributeError:
            raise ValueError(f"unknown register: {reg_id}")

    def _get_pc_reg(self) -> str:
        return get_pc_register(self.exe.arch)

    def _get_sp_reg(self) -> str:
        return get_sp_register(self.exe.arch)

    def emulate(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        count: int = 0,
        timeout: int = 0,
    ) -> None:
        if start is not None:
            self.set_pc(start)

        if count > 0:
            # run specific number of instructions
            self._engine.run(count)
        elif end is not None:
            # run until we reach end address
            current_pc = self.get_pc()
            instructions_run = 0
            max_instructions = 10000  # safety limit

            while current_pc != end and instructions_run < max_instructions:
                self._engine.run(1)
                current_pc = self.get_pc()
                instructions_run += 1

            if instructions_run >= max_instructions:
                raise EmulationError(
                    f"emulation exceeded maximum instructions ({max_instructions})"
                )
        else:
            # run indefinitely (with safety limit)
            self._engine.run(1000)

        self.log.ped(f"emulation completed")

    def halt(self) -> None:
        # maat doesn't have explicit halt, just exit the emulation loop
        pass

    def set_pc(self, value: int) -> None:
        pc_name = self._get_pc_reg()
        self.reg_write(pc_name, value)

    def get_pc(self) -> int:
        pc_name = self._get_pc_reg()
        return self.reg_read(pc_name)

    def get_reg_by_name(self, name: str) -> int:
        return self.reg_read(name)

    def set_reg_by_name(self, name: str, value: int) -> None:
        self.reg_write(name, value)

    # symbolic execution specific methods
    def symbolize_register(self, reg_name: str, comment: str = "") -> Any:
        try:
            var = maat.Var(self.exe.arch.bits, comment or f"sym_{reg_name}")
            setattr(self._engine.cpu, reg_name, var)
            self._symbolic_vars[reg_name] = var
            self.log.dbg(f"symbolized register {reg_name} -> {comment}")
            return var
        except Exception as e:
            raise ValueError(f"failed to symbolize register {reg_name}: {e}")

    def symbolize_memory(
        self, address: int, size: int, word_size: int = 1, comment: str = ""
    ) -> List[Any]:
        if word_size not in [1, 2, 4, 8]:
            raise ValueError(f"invalid word size: {word_size} (must be 1, 2, 4, or 8)")

        if size % word_size != 0:
            raise ValueError(f"size {size} must be divisible by word_size {word_size}")

        count = size // word_size
        sym_vars = []

        for i in range(count):
            addr = address + (i * word_size)
            var_name = f"{comment}_{i}" if comment else f"mem_{addr:x}"
            var = maat.Var(word_size * 8, var_name)

            # write symbolic value to memory
            self._engine.mem.write(addr, var, word_size)
            sym_vars.append(var)

        self.log.dbg(
            f"symbolized memory 0x{address:x}-0x{address+size:x} "
            f"as {count} {word_size}-byte units"
        )
        return sym_vars

    def get_symbolic_expression(self, reg_name: str) -> Any:
        try:
            return getattr(self._engine.cpu, reg_name)
        except AttributeError:
            raise ValueError(f"unknown register: {reg_name}")

    def add_memory_hook(self, hook_func: Callable, event_type: str = "read") -> None:
        if event_type == "read":
            event = maat.EVENT.MEM_R
        elif event_type == "write":
            event = maat.EVENT.MEM_W
        else:
            raise ValueError(f"unsupported event type: {event_type}")

        hook_name = f"hook_{len(self._hook_callbacks)}"
        self._hook_callbacks[hook_name] = hook_func

        # create wrapper that calls the user's hook function
        def hook_wrapper(engine, data):
            hook_func(engine, data)
            return maat.ACTION.CONTINUE

        self._engine.hooks.add(
            event, maat.WHEN.BEFORE, callbacks=[hook_wrapper], data=[], name=hook_name
        )
        self.log.dbg(f"added {event_type} hook: {hook_name}")

    def solve_constraints(self, constraint: Any = None) -> Dict[str, Any]:
        solver = maat.Solver()
        if constraint is not None:
            solver.add(constraint)

        if solver.check():
            model = solver.get_model()
            # update engine variables with model
            self._engine.vars.update_from(model)
            return model
        else:
            return {}

    def get_engine(self) -> Any:
        return self._engine

    def run_single_instruction(self) -> None:
        self._engine.run(1)
