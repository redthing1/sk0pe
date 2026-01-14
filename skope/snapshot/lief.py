from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..core.arch import Architecture, parse_arch
from ..core.memory import MemoryMap, MemoryProvider, MemoryRegion
from ..core.permissions import MemoryPermissions
from ..core.state import RegisterFile
from .base import Snapshot

try:
    import lief

    HAS_LIEF = True
except ImportError:  # pragma: no cover
    HAS_LIEF = False
    lief = None


class SnapshotMemoryProvider:
    """Memory provider backed by snapshot regions."""

    def __init__(self, memory_map: MemoryMap):
        self._memory_map = memory_map

    def get_region(self, address: int) -> Optional[MemoryRegion]:
        return self._memory_map.find(address)

    def read(self, address: int, size: int) -> Optional[bytes]:
        region = self.get_region(address)
        if not region or region.data is None:
            return None
        if not region.contains(address, size):
            return None
        offset = address - region.start
        if offset + size > len(region.data):
            return None
        return region.data[offset : offset + size]


class LiefSnapshot:
    """Loader for native binaries using LIEF."""

    @staticmethod
    def load(path: str) -> Snapshot:
        if not HAS_LIEF:
            raise RuntimeError("lief is not installed")

        binary_path = Path(path)
        if not binary_path.exists():
            raise FileNotFoundError(f"binary not found: {binary_path}")

        binary = lief.parse(str(binary_path))
        if not binary:
            raise ValueError(f"failed to parse binary: {binary_path}")

        arch = _arch_from_lief(binary)
        entrypoint = int(getattr(binary, "entrypoint", 0))
        regions = _extract_regions(binary)
        stack_base, stack_size = _default_stack_layout(arch)
        regions.append(
            MemoryRegion(
                start=stack_base,
                end=stack_base + stack_size,
                permissions=MemoryPermissions.RW,
                module_name="[stack]",
                data=None,
            )
        )
        memory_map = MemoryMap(regions)
        memory = SnapshotMemoryProvider(memory_map)

        registers = RegisterFile(
            arch, {arch.pc_name: entrypoint, arch.sp_name: stack_base + stack_size}
        )
        base, size = _compute_module_bounds(regions, entrypoint)

        metadata = {
            "format": (
                getattr(binary, "format", None).name
                if getattr(binary, "format", None)
                else ""
            ),
            "entrypoint": entrypoint,
            "path": str(binary_path),
            "stack_base": stack_base,
            "stack_size": stack_size,
        }
        platform = metadata["format"].lower()

        modules = [
            {
                "name": binary_path.name,
                "base_address": base,
                "size": size,
                "type": "main_executable",
            }
        ]

        thread = {"gpr_values": [], "fpr_values": []}

        return Snapshot(
            arch=arch,
            platform=platform,
            metadata=metadata,
            registers=registers,
            memory_map=memory_map,
            modules=modules,
            thread=thread,
            memory=memory,
        )


def _arch_from_lief(binary) -> Architecture:
    try:
        arch = binary.abstract.header.architecture
    except Exception as exc:
        raise ValueError(f"unable to determine architecture: {exc}")

    arch_map = {
        lief.Header.ARCHITECTURES.X86: Architecture.X86,
        lief.Header.ARCHITECTURES.X86_64: Architecture.X64,
        lief.Header.ARCHITECTURES.ARM64: Architecture.ARM64,
    }
    if arch in arch_map:
        return arch_map[arch]

    return parse_arch(arch.name)


def _extract_regions(binary) -> List[MemoryRegion]:
    if isinstance(binary, lief.ELF.Binary):
        return _extract_elf_regions(binary)
    if isinstance(binary, lief.MachO.Binary):
        return _extract_macho_regions(binary)
    return _extract_section_regions(binary)


def _extract_section_regions(binary) -> List[MemoryRegion]:
    regions: List[MemoryRegion] = []
    for section in binary.abstract.sections:
        if section.size <= 0:
            continue
        perms = _section_permissions(section)
        start = int(section.virtual_address)
        data = bytes(section.content) if section.content else None
        end = start + (len(data) if data else int(section.size))
        regions.append(
            MemoryRegion(
                start=start,
                end=end,
                permissions=perms,
                module_name=binary.name if hasattr(binary, "name") else "",
                data=data,
            )
        )
    return regions


def _extract_elf_regions(binary) -> List[MemoryRegion]:
    regions: List[MemoryRegion] = []
    for seg in binary.segments:
        if seg.type != lief.ELF.SEGMENT_TYPES.LOAD:
            continue
        perms = MemoryPermissions.from_rwx(
            seg.has(lief.ELF.SEGMENT_FLAGS.R),
            seg.has(lief.ELF.SEGMENT_FLAGS.W),
            seg.has(lief.ELF.SEGMENT_FLAGS.X),
        )
        start = int(seg.virtual_address)
        size = int(seg.virtual_size)
        data = bytes(seg.content) if seg.content else None
        end = start + (size if size > 0 else (len(data) if data else 0))
        regions.append(
            MemoryRegion(
                start=start,
                end=end,
                permissions=perms,
                module_name=binary.name if hasattr(binary, "name") else "",
                data=data,
            )
        )
    return regions


def _extract_macho_regions(binary) -> List[MemoryRegion]:
    regions: List[MemoryRegion] = []
    for seg in binary.segments:
        init_prot = seg.init_protection
        perms = MemoryPermissions.from_rwx(
            read=(init_prot & 1) != 0,
            write=(init_prot & 2) != 0,
            execute=(init_prot & 4) != 0,
        )
        start = int(seg.virtual_address)
        size = int(seg.virtual_size)
        content = seg.content
        data = bytes(content) if content else None
        end = start + (size if size > 0 else (len(data) if data else 0))
        regions.append(
            MemoryRegion(
                start=start,
                end=end,
                permissions=perms,
                module_name=binary.name if hasattr(binary, "name") else "",
                data=data,
            )
        )
    return regions


def _section_permissions(section) -> MemoryPermissions:
    read = getattr(section, "is_readable", lambda: True)()
    write = getattr(section, "is_writable", lambda: False)()
    execute = getattr(section, "is_executable", lambda: False)()
    return MemoryPermissions.from_rwx(read, write, execute)


def _compute_module_bounds(
    regions: Iterable[MemoryRegion], entrypoint: int
) -> (int, int):
    starts = [region.start for region in regions]
    ends = [region.end for region in regions]
    if not starts:
        return entrypoint, 0
    base = min(starts)
    size = max(ends) - base
    return base, size


def _default_stack_layout(arch: Architecture) -> (int, int):
    if arch in (Architecture.X64, Architecture.ARM64):
        return 0x7FFF00000000, 0x100000
    return 0xBFFF0000, 0x100000
