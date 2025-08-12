#!/usr/bin/env python3
"""
lief loader - flexible multi-format binary loader using lief

supports elf, pe, mach-o, and other formats that lief can parse
"""

from typing import List, Optional, Dict, Any, Tuple
import lief
from redlog import get_logger
from ..emu.base import Executable, Arch, Segment, Permission, MemoryError


# architecture mapping from lief to our arch enum
ARCH_MAP = {
    lief.Header.ARCHITECTURES.X86: Arch.X86,
    lief.Header.ARCHITECTURES.X86_64: Arch.X64,
    lief.Header.ARCHITECTURES.ARM64: Arch.ARM64,
    # add more architectures as needed:
    # lief.Header.ARCHITECTURES.ARM: Arch.ARM,
    # lief.Header.ARCHITECTURES.MIPS: Arch.MIPS,
}


class LiefExecutable(Executable):
    """base executable for any lief-supported format"""

    def __init__(self, binary: lief.Binary):
        self.log = get_logger("lief.executable")

        if not binary:
            raise ValueError("binary cannot be none")

        try:
            # extract everything we need immediately
            self._format = binary.format.name
            self._arch = self._extract_arch(binary)
            self._base_address = self._extract_base_address(binary)
            self._entry_point = binary.abstract.header.entrypoint

            # extract segments immediately
            self._segments = self._extract_segments(binary)

            # extract function info
            self._exported_functions = self._extract_exported_functions(binary)
            self._symbols = self._extract_symbols(binary)
        finally:
            # always delete the binary to avoid leaks
            del binary

    @classmethod
    def from_path(cls, binary_path: str) -> "LiefExecutable":
        """load binary from path and return appropriate subclass"""
        binary = lief.parse(binary_path)
        if not binary:
            raise ValueError(f"failed to parse binary: {binary_path}")

        # return format-specific subclass if needed
        if isinstance(binary, lief.ELF.Binary):
            return ElfExecutable(binary)
        elif isinstance(binary, lief.PE.Binary):
            return PeExecutable(binary)
        elif isinstance(binary, lief.MachO.Binary):
            return MachoExecutable(binary)
        else:
            # fallback to generic
            return cls(binary)

    @property
    def arch(self) -> Arch:
        """get architecture"""
        return self._arch

    @property
    def base_address(self) -> int:
        """get base address"""
        return self._base_address

    @property
    def entry_point(self) -> int:
        """get entry point address"""
        return self._entry_point

    def get_segments(self) -> List[Segment]:
        """get loadable segments"""
        return self._segments

    def _extract_arch(self, binary: lief.Binary) -> Arch:
        """extract architecture from binary"""
        lief_arch = binary.abstract.header.architecture
        arch = ARCH_MAP.get(lief_arch)
        if not arch:
            raise ValueError(f"unsupported architecture: {lief_arch.name}")
        return arch

    def _extract_base_address(self, binary: lief.Binary) -> int:
        """extract base address from binary"""
        # try to get minimum section address
        min_addr = None
        for section in binary.abstract.sections:
            if section.size > 0 and section.virtual_address > 0:
                if min_addr is None or section.virtual_address < min_addr:
                    min_addr = section.virtual_address
        return min_addr if min_addr is not None else 0

    def _extract_segments(self, binary: lief.Binary) -> List[Segment]:
        """extract segments from binary"""
        segments = []

        # use abstract sections api when possible
        for section in binary.abstract.sections:
            if section.size > 0:
                # convert section to our segment format
                segments.append(
                    Segment(
                        address=int(section.virtual_address),
                        data=bytes(section.content),  # bytes() makes a copy
                        permissions=self._get_permissions(section),
                    )
                )

        return segments

    def _extract_exported_functions(self, binary: lief.Binary) -> Dict[str, int]:
        """extract exported functions from binary"""
        functions = {}
        for func in binary.abstract.exported_functions:
            # extract primitive values only
            name = str(func.name) if func.name else ""
            addr = int(func.address)
            if name:
                functions[name] = addr
        return functions

    def _extract_symbols(self, binary: lief.Binary) -> List[Dict[str, Any]]:
        """extract symbols from binary"""
        symbols = []
        for sym in binary.abstract.symbols:
            # extract primitive values only
            name = str(sym.name) if sym.name else ""
            value = int(sym.value) if sym.value else 0
            is_func = bool(getattr(sym, "is_function", False))
            if name:  # only keep named symbols
                symbols.append({"name": name, "value": value, "is_function": is_func})
        return symbols

    def _get_permissions(self, section) -> Permission:
        """get permissions for a section"""
        try:
            # try to get from section flags if available
            if hasattr(section, "has_characteristic"):  # pe
                return self._from_rwx_flags(
                    section.has_characteristic(
                        lief.PE.SECTION_CHARACTERISTICS.MEM_READ
                    ),
                    section.has_characteristic(
                        lief.PE.SECTION_CHARACTERISTICS.MEM_WRITE
                    ),
                    section.has_characteristic(
                        lief.PE.SECTION_CHARACTERISTICS.MEM_EXECUTE
                    ),
                )
            elif hasattr(section, "flags"):  # elf/macho
                # check common flags with safe defaults
                r = getattr(section, "is_readable", lambda: True)()
                w = getattr(section, "is_writable", lambda: False)()
                x = getattr(section, "is_executable", lambda: False)()
                return self._from_rwx_flags(r, w, x)
        except Exception as e:
            self.log.err(f"failed to get permissions for section at 0x{address:x}: {e}")
            raise MemoryError(
                f"cannot determine permissions for section at 0x{address:x}: {e}"
            ) from e

    def get_memory_region(
        self, address: int
    ) -> Optional[Tuple[int, bytes, Permission]]:
        """get memory region containing address for lazy loading"""
        for seg in self.get_segments():
            if seg.address <= address < seg.address + len(seg.data):
                return seg.address, seg.data, seg.permissions
        return None

    # utility methods using extracted data
    def find_function(self, name: str) -> Optional[int]:
        """find function address by name"""
        # check exported functions
        for func_name, addr in self._exported_functions.items():
            if name in func_name:
                return addr

        # try symbols
        for sym in self._symbols:
            if name in sym["name"] and sym["is_function"]:
                return sym["value"]
        return None

    def find_functions(self, pattern: str) -> Dict[str, int]:
        """find all functions matching pattern"""
        functions = {}

        # exported functions
        for func_name, addr in self._exported_functions.items():
            if pattern in func_name:
                functions[func_name] = addr

        # symbols
        for sym in self._symbols:
            if pattern in sym["name"] and sym["is_function"]:
                functions[sym["name"]] = sym["value"]

        return functions

    def get_imported_functions(self) -> Dict[str, str]:
        """get imported functions"""
        # base class doesn't extract imports - override in subclasses if needed
        return {}

    def get_exported_functions(self) -> Dict[str, int]:
        """get exported functions"""
        return self._exported_functions

    @staticmethod
    def _from_rwx_flags(read: bool, write: bool, execute: bool) -> Permission:
        """convert rwx boolean flags to Permission enum"""
        return Permission.from_rwx(read, write, execute)


class ElfExecutable(LiefExecutable):
    """elf-specific executable with additional elf features"""

    def __init__(self, binary: lief.ELF.Binary):
        # extract elf-specific data first
        self._dynamic_symbols = self._extract_dynamic_symbols(binary)
        self._relocations = self._extract_relocations(binary)

        # call parent to extract common data and delete binary
        super().__init__(binary)

    def _extract_segments(self, binary: lief.Binary) -> List[Segment]:
        """override to use elf segments instead of sections for loading"""
        segments = []
        for phdr in binary.segments:
            if phdr.type == lief.ELF.SEGMENT_TYPES.LOAD:
                perms = self._from_rwx_flags(
                    phdr.has(lief.ELF.SEGMENT_FLAGS.R),
                    phdr.has(lief.ELF.SEGMENT_FLAGS.W),
                    phdr.has(lief.ELF.SEGMENT_FLAGS.X),
                )
                segments.append(
                    Segment(
                        address=int(phdr.virtual_address),
                        data=bytes(phdr.content),  # bytes() makes a copy
                        permissions=perms,
                    )
                )
        return segments

    def _extract_base_address(self, binary: lief.Binary) -> int:
        """get elf base address from load segments"""
        min_addr = None
        for seg in binary.segments:
            if seg.type == lief.ELF.SEGMENT_TYPES.LOAD and seg.virtual_address > 0:
                if min_addr is None or seg.virtual_address < min_addr:
                    min_addr = seg.virtual_address
        return min_addr if min_addr is not None else 0x400000  # default elf base

    def _extract_dynamic_symbols(self, binary: lief.ELF.Binary) -> Dict[str, int]:
        """extract dynamic symbols"""
        symbols = {}
        for sym in binary.dynamic_symbols:
            if sym.name and sym.value:
                # extract primitive values only
                symbols[str(sym.name)] = int(sym.value)
        return symbols

    def _extract_relocations(self, binary: lief.ELF.Binary) -> List[Dict[str, Any]]:
        """extract relocations"""
        relocs = []
        for reloc in binary.relocations:
            relocs.append(
                {
                    "address": int(reloc.address),
                    "type": str(reloc.type),
                    "symbol": str(reloc.symbol.name) if reloc.has_symbol else None,
                    "addend": int(reloc.addend),
                }
            )
        return relocs

    # elf-specific utilities
    def get_dynamic_symbols(self) -> Dict[str, int]:
        """get dynamic symbol table"""
        return self._dynamic_symbols

    def get_relocations(self) -> List[Dict[str, Any]]:
        """get relocations"""
        return self._relocations


class PeExecutable(LiefExecutable):
    """pe-specific executable with additional pe features"""

    def __init__(self, binary: lief.PE.Binary):
        # extract pe-specific data first
        self._imports = self._extract_imports(binary)
        self._exports = self._extract_exports(binary)
        self._has_resources = binary.has_resources

        # call parent to extract common data and delete binary
        super().__init__(binary)

    def _extract_base_address(self, binary: lief.Binary) -> int:
        """get pe image base"""
        return int(binary.optional_header.imagebase)

    def _extract_imports(self, binary: lief.PE.Binary) -> Dict[str, List[str]]:
        """extract imports organized by dll"""
        imports = {}
        for imp in binary.imports:
            dll_name = str(imp.name)
            imports[dll_name] = [str(entry.name) for entry in imp.entries if entry.name]
        return imports

    def _extract_exports(self, binary: lief.PE.Binary) -> Dict[str, int]:
        """extract export table"""
        exports = {}
        if binary.has_exports:
            base = int(binary.optional_header.imagebase)
            for entry in binary.get_export().entries:
                if entry.name:
                    exports[str(entry.name)] = base + int(entry.address)
        return exports

    # pe-specific utilities
    def get_imports(self) -> Dict[str, List[str]]:
        """get imports organized by dll"""
        return self._imports

    def get_exports(self) -> Dict[str, int]:
        """get export table"""
        return self._exports

    def get_resources(self) -> Optional[Any]:
        """get resource tree if available"""
        # resources are too complex to cache and rarely needed
        return None


class MachoExecutable(LiefExecutable):
    """mach-o specific executable with additional features"""

    def __init__(self, binary: lief.MachO.Binary):
        # extract mach-o specific data first
        self._dylibs = self._extract_dylibs(binary)
        self._rpaths = self._extract_rpaths(binary)
        self._macho_symbols = self._extract_macho_symbols(binary)

        # call parent to extract common data and delete binary
        super().__init__(binary)

    def _extract_segments(self, binary: lief.Binary) -> List[Segment]:
        """extract segments from mach-o binary with proper permissions"""
        segments = []

        # for mach-o, we need to look at segments, not sections
        for seg in binary.segments:
            # get the initial protection (what the segment starts with)
            init_prot = seg.init_protection

            # convert mach-o VM_PROT_* flags to permission enum
            # vm_prot_read = 1, vm_prot_write = 2, vm_prot_execute = 4
            perms = Permission.from_rwx(
                read=(init_prot & 1) != 0,
                write=(init_prot & 2) != 0,
                execute=(init_prot & 4) != 0,
            )

            # get segment content (not seg.data which is the header)
            if seg.file_size > 0:
                # use content, not data!
                content = seg.content
                if content:
                    segments.append(
                        Segment(
                            address=int(seg.virtual_address),
                            data=bytes(content),
                            permissions=perms,
                        )
                    )

        return segments

    def _extract_base_address(self, binary: lief.Binary) -> int:
        """get mach-o base address"""
        min_addr = None
        for seg in binary.segments:
            if seg.virtual_address > 0:
                if min_addr is None or seg.virtual_address < min_addr:
                    min_addr = seg.virtual_address
        return min_addr if min_addr is not None else 0x100000000  # default macos base

    def _extract_dylibs(self, binary: lief.MachO.Binary) -> List[str]:
        """extract linked dynamic libraries"""
        return [str(cmd.name) for cmd in binary.libraries]

    def _extract_rpaths(self, binary: lief.MachO.Binary) -> List[str]:
        """extract runtime search paths"""
        return [str(rpath.path) for rpath in binary.rpaths]

    def _extract_macho_symbols(self, binary: lief.MachO.Binary) -> Dict[str, int]:
        """extract symbol table"""
        symbols = {}
        for sym in binary.symbols:
            if sym.name and sym.value:
                symbols[str(sym.name)] = int(sym.value)
        return symbols

    # mach-o specific utilities
    def get_dylibs(self) -> List[str]:
        """get linked dynamic libraries"""
        return self._dylibs

    def get_rpaths(self) -> List[str]:
        """get runtime search paths"""
        return self._rpaths

    def get_symbols(self) -> Dict[str, int]:
        """get symbol table"""
        return self._macho_symbols


def load_binary(binary_path: str) -> LiefExecutable:
    """convenience function to load any supported binary format"""
    return LiefExecutable.from_path(binary_path)


def create_lief_triton_emulator(binary_path: str, hooks: int = 0) -> "TritonEmulator":
    """create a triton emulator from a lief binary"""
    from ..emu.triton import TritonEmulator

    executable = load_binary(binary_path)
    return TritonEmulator(executable, hooks)


def create_lief_triton_emulator_from_executable(
    executable: LiefExecutable, hooks: int = 0
) -> "TritonEmulator":
    """create a triton emulator from an existing lief executable"""
    from ..emu.triton import TritonEmulator

    return TritonEmulator(executable, hooks)


def create_lief_unicorn_emulator(binary_path: str, hooks: int = 0) -> "UnicornEmulator":
    """create a unicorn emulator from a lief binary"""
    from ..emu.unicorn import UnicornEmulator

    executable = load_binary(binary_path)
    return UnicornEmulator(executable, hooks)


def create_lief_unicorn_emulator_from_executable(
    executable: LiefExecutable, hooks: int = 0
) -> "UnicornEmulator":
    """create a unicorn emulator from an existing lief executable"""
    from ..emu.unicorn import UnicornEmulator

    return UnicornEmulator(executable, hooks)
