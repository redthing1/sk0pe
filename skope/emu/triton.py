#!/usr/bin/env python3
"""
Triton symbolic execution engine implementation of the emulator interface
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
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

try:
    from triton import (
        TritonContext,
        ARCH,
        MODE,
        MemoryAccess,
        CPUSIZE,
        Instruction,
        EXCEPTION,
        AST_REPRESENTATION,
        CALLBACK,
    )

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    TritonContext = None
    ARCH = None
    MODE = None
    MemoryAccess = None
    CPUSIZE = None
    Instruction = None
    EXCEPTION = None
    CALLBACK = None


class CallbackType(Enum):
    """triton callback types"""

    GET_CONCRETE_MEMORY_VALUE = 0
    GET_CONCRETE_REGISTER_VALUE = 1
    SET_CONCRETE_MEMORY_VALUE = 2
    SET_CONCRETE_REGISTER_VALUE = 3
    SYMBOLIC_SIMPLIFICATION = 4


@dataclass
class RegisterSymbolicInfo:
    """information about a register's symbolic state"""

    name: str
    concrete_value: int
    is_symbolic: bool
    symbolic_id: Optional[int] = None
    ast: Optional[str] = None
    simplified_ast: Optional[str] = None
    comment: Optional[str] = None


class TritonEmulator(BareMetalEmulator):
    """triton symbolic execution engine implementation"""

    def __init__(self, executable: Executable, hooks: Hook = Hook.DEFAULT):
        if not HAS_TRITON:
            raise RuntimeError("triton is not installed")

        self._ctx: Optional[TritonContext] = None
        self._arch_to_triton: Dict[Arch, Any] = {}
        self._symbolic_regs: set = set()
        self._symbolic_mem: set = set()
        self._memory_lazy_load_callback: Optional[Callable] = None

        super().__init__(executable, hooks)

    def _setup(self) -> None:
        """initialize Triton context"""
        # map architecture to Triton constants
        arch_map = {
            Arch.X86: ARCH.X86,
            Arch.X64: ARCH.X86_64,
            Arch.ARM64: ARCH.AARCH64,
        }

        triton_arch = arch_map.get(self.exe.arch)
        if not triton_arch:
            raise ValueError(f"unsupported architecture: {self.exe.arch}")

        # create triton context
        self._ctx = TritonContext(triton_arch)
        self.log.dbg(f"created triton context for {self.exe.arch.name}")

        # configure triton modes for optimal performance
        self._ctx.setMode(MODE.ALIGNED_MEMORY, True)
        self._ctx.setMode(MODE.CONSTANT_FOLDING, True)
        self._ctx.setMode(MODE.AST_OPTIMIZATIONS, True)
        self._ctx.setMode(MODE.ONLY_ON_SYMBOLIZED, True)
        self.log.dbg("configured triton modes for concolic execution")

        # # more readable ast syntax
        # self._ctx.setAstRepresentationMode(AST_REPRESENTATION.PYTHON)

        # load segments
        self._load_segments()

        # set up memory layout
        self._setup_memory_layout()

        # install hooks
        self._install_hooks()

    def _setup_memory_layout(self) -> None:
        """set up heap and stack"""
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

        # initialize sp to top of stack
        self.sp = self.stack_base + self.stack_size
        self.log.dbg(f"stack pointer initialized to 0x{self.sp:x}")

    def _load_segments(self) -> None:
        """load all segments from the executable"""
        segments = self.exe.get_segments()
        self.log.trc(f"loading {len(segments)} segments")

        # load and write segments to memory
        for i, seg in enumerate(segments):
            self.log.dbg(
                f"loading segment {i} @ addr=0x{seg.address:x} size={seg.size} perms={seg.permissions}"
            )
            self.mem_write(seg.address, seg.data)

        self.log.dbg("all segments loaded")

    def _install_hooks(self) -> None:
        """install requested hooks"""
        # triton doesn't have explicit hooks like unicorn
        # hooks are handled through the processing method
        self.log.dbg("hooks will be handled during instruction processing")

        # install callback for lazy loading
        # create a weak reference to avoid reference cycles
        # unfortunately this is required to prevent weird issues with nanobind?
        import weakref

        weak_self = weakref.ref(self)

        def memory_callback(ctx, mem_access):
            self_ref = weak_self()
            if self_ref:
                self_ref._memory_access_callback(ctx, mem_access)

        self.add_callback(CallbackType.GET_CONCRETE_MEMORY_VALUE, memory_callback)

    def _memory_access_callback(self, ctx, mem_access):
        """callback for lazy loading memory"""
        addr = mem_access.getAddress()
        size = mem_access.getSize()
        if not self.is_memory_defined(addr, size):
            self._handle_memory_fault(addr)

    def _handle_memory_fault(self, address: int) -> bool:
        """override to use triton's memory API"""
        if region := self.exe.get_memory_region(address):
            start, data, perms = region
            # use triton's memory API
            self.set_memory_area(start, list(data))

            # call the lazy load callback if set
            if self._memory_lazy_load_callback:
                self._memory_lazy_load_callback(self, start, len(data))

            return True
        return False

    def set_memory_lazy_load_callback(
        self, callback: Callable[[Any, int, int], None]
    ) -> None:
        """set callback for when memory is lazily loaded.
        callback signature: (emulator, address, size) -> None"""
        self._memory_lazy_load_callback = callback

    def _map(self, address: int, size: int, permissions: Permission) -> None:
        """internal method to map memory in the engine"""
        # triton doesn't require explicit memory mapping
        # memory is implicitly available when accessed
        self.log.dbg(f"memory region 0x{address:x}-0x{address+size:x} is available")

    def mem_read(self, address: int, size: int) -> bytes:
        self.log.ped(f"memory read @ 0x{address:x} (size: {size})")
        result = bytearray()
        for i in range(size):
            byte_val = self._ctx.getConcreteMemoryValue(address + i)
            result.append(byte_val)
        return bytes(result)

    def mem_write(self, address: int, data: bytes) -> None:
        self.log.ped(f"memory write @ 0x{address:x} (size: {len(data)})")
        self._ctx.setConcreteMemoryAreaValue(address, list(data))

    def reg_read(self, reg_id: str) -> int:
        """read register by name (triton uses string register names)"""
        reg = getattr(self._ctx.registers, reg_id, None)
        if not reg:
            raise ValueError(f"unknown register: {reg_id}")
        return self._ctx.getConcreteRegisterValue(reg)

    def reg_write(self, reg_id: str, value: int) -> None:
        """write register by name (triton uses string register names)"""
        reg = getattr(self._ctx.registers, reg_id, None)
        if not reg:
            raise ValueError(f"unknown register: {reg_id}")
        self._ctx.setConcreteRegisterValue(reg, value)

    def _get_pc_reg(self) -> str:
        """get the program counter register name for this architecture"""
        return get_pc_register(self.exe.arch)

    def _get_sp_reg(self) -> str:
        """get the stack pointer register name for this architecture"""
        return get_sp_register(self.exe.arch)

    def emulate(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        count: int = 0,
        timeout: int = 0,
    ) -> None:
        """run emulation, raising exceptions on error"""
        if start is None:
            start = self.pc

        pc = start
        executed = 0
        last_exception = EXCEPTION.NO_FAULT

        while pc:
            # check end condition
            if end and pc >= end:
                break
            if count > 0 and executed >= count:
                break

            # fetch instruction
            try:
                # read up to 16 bytes for instruction
                inst_bytes = self.mem_read(pc, 16)
            except Exception as e:
                self.log.err(f"failed to read instruction at 0x{pc:x}")
                raise EmulationError(f"cannot read instruction at 0x{pc:x}: {e}") from e

            # create Triton instruction
            inst = Instruction()
            inst.setOpcode(inst_bytes)
            inst.setAddress(pc)

            # process hooks
            if self.hooks & Hook.CODE_EXECUTE:
                if not self.hook_code_execute(pc, inst.getSize()):
                    break

            # process instruction
            try:
                last_exception = self._ctx.processing(inst)
                if last_exception == EXCEPTION.FAULT_UD:
                    self.log.err(f"undefined instruction at 0x{pc:x}")
                    raise EmulationError(f"undefined instruction at 0x{pc:x}")
                elif last_exception != EXCEPTION.NO_FAULT:
                    self.log.err(f"exception {last_exception} at 0x{pc:x}")
                    raise EmulationError(
                        f"emulation exception {last_exception} at 0x{pc:x}"
                    )
            except EmulationError:
                raise  # re-raise our own exceptions
            except Exception as e:
                self.log.err(f"error processing instruction at 0x{pc:x}: {e}")
                raise EmulationError(
                    f"error processing instruction at 0x{pc:x}: {e}"
                ) from e

            # handle memory access hooks
            for mem_access in inst.getLoadAccess():
                if self.hooks & Hook.MEMORY_READ:
                    addr = mem_access.getAddress()
                    size = mem_access.getSize()
                    value = mem_access.getValue()
                    self.hook_memory_read(addr, size, value)

            for mem_access in inst.getStoreAccess():
                if self.hooks & Hook.MEMORY_WRITE:
                    addr = mem_access.getAddress()
                    size = mem_access.getSize()
                    value = mem_access.getValue()
                    self.hook_memory_write(addr, size, value)

            # update pc
            pc = self.pc
            executed += 1

        self.log.ped(f"executed {executed} instructions")

    def halt(self) -> None:
        """stop emulation"""
        # triton doesn't have explicit halt, just exit the emulation loop
        pass

    # symbolic execution specific methods
    def symbolize_register(self, reg_name: str, comment: str = "") -> Any:
        """make a register symbolic"""
        reg = getattr(self._ctx.registers, reg_name, None)
        if not reg:
            raise ValueError(f"unknown register: {reg_name}")

        sym_var = self._ctx.symbolizeRegister(reg, comment)
        self._symbolic_regs.add(reg_name)
        self.log.dbg(f"symbolized register {reg_name} -> {sym_var}")
        return sym_var

    def symbolize_memory(
        self, address: int, size: int, word_size: int = 1, comment: str = ""
    ) -> List[Any]:
        """
        symbolize a memory region with configurable word size.

        Args:
            address: starting address
            size: total size in bytes to symbolize
            word_size: size of each symbolic unit in bytes (1, 2, 4, or 8)
            comment: optional comment/prefix for naming

        Returns:
            list of symbolic variables created
        """
        if word_size not in [1, 2, 4, 8]:
            raise ValueError(f"invalid word size: {word_size} (must be 1, 2, 4, or 8)")

        if size % word_size != 0:
            raise ValueError(f"size {size} must be divisible by word_size {word_size}")

        # map word sizes to Triton CPU sizes
        size_map = {
            1: CPUSIZE.BYTE,
            2: CPUSIZE.WORD,
            4: CPUSIZE.DWORD,
            8: CPUSIZE.QWORD,
        }

        cpu_size = size_map[word_size]
        count = size // word_size
        sym_vars = []

        for i in range(count):
            addr = address + (i * word_size)
            mem_access = MemoryAccess(addr, cpu_size)
            sym_var = self._ctx.symbolizeMemory(mem_access)

            # use clean naming without archaic terminology
            if comment:
                sym_var.setAlias(f"{comment}_{i}")
            else:
                sym_var.setAlias(f"mem_{addr:x}")

            sym_vars.append(sym_var)

            # track all bytes in this word
            for j in range(word_size):
                self._symbolic_mem.add(addr + j)

        self.log.dbg(
            f"symbolized memory 0x{address:x}-0x{address+size:x} "
            f"as {count} {word_size}-byte units"
        )
        return sym_vars

    def get_symbolic_registers(self) -> Dict[str, Any]:
        """get current symbolic state of registers"""
        result = {}
        for reg_name in self._symbolic_regs:
            reg = getattr(self._ctx.registers, reg_name, None)
            if reg:
                sym_expr = self._ctx.getSymbolicRegister(reg)
                if sym_expr:
                    result[reg_name] = sym_expr
        return result

    def get_path_constraints(self) -> List[Any]:
        """get accumulated path constraints"""
        return self._ctx.getPathConstraints()

    def get_path_predicate(self) -> Any:
        """get the path predicate (conjunction of all constraints)"""
        return self._ctx.getPathPredicate()

    def solve_constraints(self, constraint: Any = None) -> Dict[int, Any]:
        """use SMT solver to find concrete values"""
        if constraint is None:
            constraint = self.get_path_predicate()

        model = self._ctx.getModel(constraint)
        return model

    def get_ast_context(self) -> Any:
        """get the AST context for building custom constraints"""
        return self._ctx.getAstContext()

    def push_path_constraint(self, constraint: Any) -> None:
        """add a custom path constraint"""
        self._ctx.pushPathConstraint(constraint)
        self.log.dbg("pushed custom path constraint")

    def get_register_ast(self, reg_name: str) -> Any:
        """get the symbolic AST for a register"""
        reg = getattr(self._ctx.registers, reg_name, None)
        if not reg:
            raise ValueError(f"unknown register: {reg_name}")
        return self._ctx.getRegisterAst(reg)

    def simplify_ast(self, ast: Any) -> Any:
        """simplify an AST expression"""
        return self._ctx.simplify(ast)

    def get_all_registers(self) -> List[Any]:
        """get all registers for the current architecture"""
        return self._ctx.getAllRegisters()

    def is_memory_defined(self, address: int, size: int) -> bool:
        """check if memory at address is defined with concrete values"""
        return self._ctx.isConcreteMemoryValueDefined(address, size)

    def set_memory_area(self, address: int, data: List[int]) -> None:
        """set concrete memory area value"""
        self._ctx.setConcreteMemoryAreaValue(address, data)

    def get_context(self) -> Any:
        """get the underlying Triton context - use sparingly"""
        return self._ctx

    def set_concrete_register_value(self, reg_name: str, value: int) -> None:
        """set concrete value for a register by name"""
        reg = getattr(self._ctx.registers, reg_name, None)
        if not reg:
            raise ValueError(f"unknown register: {reg_name}")
        self._ctx.setConcreteRegisterValue(reg, value)

    def get_concrete_register_value(self, reg_name: str) -> int:
        """get concrete value for a register by name"""
        reg = getattr(self._ctx.registers, reg_name, None)
        if not reg:
            raise ValueError(f"unknown register: {reg_name}")
        return self._ctx.getConcreteRegisterValue(reg)

    def get_reg_by_name(self, name: str) -> int:
        """read register by name (architecture-independent)"""
        return self.get_concrete_register_value(name)

    def set_reg_by_name(self, name: str, value: int) -> None:
        """write register by name (architecture-independent)"""
        self.set_concrete_register_value(name, value)

    def concretize_all_registers(self) -> None:
        self._ctx.concretizeAllRegister()

    def concretize_all_memory(self) -> None:
        self._ctx.concretizeAllMemory()

    def clear_path_constraints(self) -> None:
        self._ctx.clearPathConstraints()

    def is_register_symbolized(self, reg_name: str) -> bool:
        """check if a register is symbolic"""
        reg = getattr(self._ctx.registers, reg_name, None)
        if not reg:
            raise ValueError(f"unknown register: {reg_name}")
        return self._ctx.isRegisterSymbolized(reg)

    def is_memory_symbolized(self, address: int, size: int, word_size: int = 1) -> bool:
        if word_size not in [1, 2, 4, 8]:
            raise ValueError(f"invalid word size: {word_size} (must be 1, 2, 4, or 8)")
        if size % word_size != 0:
            raise ValueError(f"size {size} must be divisible by word_size {word_size}")
        # map word sizes to Triton CPU sizes
        size_map = {
            1: CPUSIZE.BYTE,
            2: CPUSIZE.WORD,
            4: CPUSIZE.DWORD,
            8: CPUSIZE.QWORD,
        }
        cpu_size = size_map[word_size]
        mem_access = MemoryAccess(address, cpu_size)
        return self._ctx.isMemorySymbolized(mem_access)

    def get_general_purpose_registers(self) -> List[Any]:
        """get general purpose registers for the current architecture"""
        all_regs = self.get_all_registers()

        if self.exe.arch == Arch.ARM64:
            # get x0-x30 for arm64
            gp_regs = []
            for reg in all_regs:
                name = reg.getName()
                if name.startswith("x") and name[1:].isdigit():
                    reg_num = int(name[1:])
                    if 0 <= reg_num <= 30:
                        gp_regs.append(reg)
            return gp_regs

        elif self.exe.arch == Arch.X64:
            # get standard x64 gp registers
            gp_names = {
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
            }
            return [reg for reg in all_regs if reg.getName() in gp_names]

        elif self.exe.arch == Arch.X86:
            # get standard x86 gp registers
            gp_names = {"eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp"}
            return [reg for reg in all_regs if reg.getName() in gp_names]

        else:
            self.log.warn(
                f"gp register enumeration not implemented for {self.exe.arch}"
            )
            return []

    def get_register_symbolic_info(self, reg_name: str) -> RegisterSymbolicInfo:
        """get detailed symbolic information about a register"""
        reg = getattr(self._ctx.registers, reg_name, None)
        if not reg:
            raise ValueError(f"unknown register: {reg_name}")

        info = RegisterSymbolicInfo(
            name=reg_name,
            concrete_value=self._ctx.getConcreteRegisterValue(reg),
            is_symbolic=self._ctx.isRegisterSymbolized(reg),
        )

        sym_expr = self._ctx.getSymbolicRegister(reg)
        if sym_expr:
            info.symbolic_id = sym_expr.getId()
            info.comment = sym_expr.getComment()

            ast = sym_expr.getAst()
            info.ast = str(ast)

            # try to simplify the AST
            simplified = (
                self._ctx.simplify(ast, True) if hasattr(self._ctx, "simplify") else ast
            )
            if str(simplified) != str(ast):
                info.simplified_ast = str(simplified)

        return info

    def symbolize_all_gp_registers(self, prefix: str = "init") -> Dict[str, Any]:
        """symbolize all general purpose registers with optional prefix"""
        gp_regs = self.get_general_purpose_registers()
        symbolized = {}

        for reg in gp_regs:
            reg_name = reg.getName()
            comment = f"{prefix}_{reg_name}"
            sym_var = self._ctx.symbolizeRegister(reg, comment)
            self._symbolic_regs.add(reg_name)
            symbolized[reg_name] = sym_var
            self.log.dbg(f"symbolized {reg_name} -> {comment}")

        return symbolized

    def set_pc(self, value: int) -> None:
        """set program counter value"""
        pc_name = self._get_pc_reg()
        self.set_concrete_register_value(pc_name, value)

    def get_pc(self) -> int:
        """get program counter value"""
        pc_name = self._get_pc_reg()
        return self.get_concrete_register_value(pc_name)

    def add_callback(
        self, callback_type: CallbackType, callback_func: Callable
    ) -> None:
        """
        add a callback for specific Triton events.

        Args:
            callback_type: type of callback (from CallbackType enum)
            callback_func: function to call. Signature depends on callback type:
                - GET_CONCRETE_MEMORY_VALUE: (ctx, memory_access) -> None
                - GET_CONCRETE_REGISTER_VALUE: (ctx, register) -> None
                - SET_CONCRETE_MEMORY_VALUE: (ctx, memory_access, value) -> None
                - SET_CONCRETE_REGISTER_VALUE: (ctx, register, value) -> None
                - SYMBOLIC_SIMPLIFICATION: (ctx, node) -> node
        """
        if not HAS_TRITON or not CALLBACK:
            raise RuntimeError("triton callbacks not available")

        # map our enum to triton's callback constants
        callback_map = {
            CallbackType.GET_CONCRETE_MEMORY_VALUE: CALLBACK.GET_CONCRETE_MEMORY_VALUE,
            CallbackType.GET_CONCRETE_REGISTER_VALUE: CALLBACK.GET_CONCRETE_REGISTER_VALUE,
            CallbackType.SET_CONCRETE_MEMORY_VALUE: CALLBACK.SET_CONCRETE_MEMORY_VALUE,
            CallbackType.SET_CONCRETE_REGISTER_VALUE: CALLBACK.SET_CONCRETE_REGISTER_VALUE,
            CallbackType.SYMBOLIC_SIMPLIFICATION: CALLBACK.SYMBOLIC_SIMPLIFICATION,
        }

        triton_callback = callback_map.get(callback_type)
        if triton_callback is None:
            raise ValueError(f"unknown callback type: {callback_type}")

        self._ctx.addCallback(triton_callback, callback_func)
        self.log.dbg(f"added callback for {callback_type.name}")

    def remove_callback(
        self, callback_type: CallbackType, callback_func: Callable
    ) -> None:
        """remove a previously added callback"""
        if not HAS_TRITON or not CALLBACK:
            raise RuntimeError("triton callbacks not available")

        callback_map = {
            CallbackType.GET_CONCRETE_MEMORY_VALUE: CALLBACK.GET_CONCRETE_MEMORY_VALUE,
            CallbackType.GET_CONCRETE_REGISTER_VALUE: CALLBACK.GET_CONCRETE_REGISTER_VALUE,
            CallbackType.SET_CONCRETE_MEMORY_VALUE: CALLBACK.SET_CONCRETE_MEMORY_VALUE,
            CallbackType.SET_CONCRETE_REGISTER_VALUE: CALLBACK.SET_CONCRETE_REGISTER_VALUE,
            CallbackType.SYMBOLIC_SIMPLIFICATION: CALLBACK.SYMBOLIC_SIMPLIFICATION,
        }

        triton_callback = callback_map.get(callback_type)
        if triton_callback is None:
            raise ValueError(f"unknown callback type: {callback_type}")

        self._ctx.removeCallback(triton_callback, callback_func)
        self.log.dbg(f"removed callback for {callback_type.name}")
