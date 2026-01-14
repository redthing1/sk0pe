from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Protocol

from .permissions import MemoryPermissions


@dataclass(frozen=True)
class MemoryRegion:
    start: int
    end: int
    permissions: MemoryPermissions
    module_name: str = ""
    data: Optional[bytes] = None

    @property
    def size(self) -> int:
        return max(0, self.end - self.start)

    def contains(self, address: int, size: int = 1) -> bool:
        if size <= 0:
            return False
        return self.start <= address and (address + size) <= self.end


class MemoryProvider(Protocol):
    """Protocol for resolving memory bytes from a snapshot."""

    def read(self, address: int, size: int) -> Optional[bytes]: ...

    def get_region(self, address: int) -> Optional[MemoryRegion]: ...


class MemoryMap:
    """Sorted container for memory regions."""

    def __init__(self, regions: Iterable[MemoryRegion]):
        self._regions: List[MemoryRegion] = sorted(regions, key=lambda r: r.start)

    def iter_regions(self) -> Iterator[MemoryRegion]:
        return iter(self._regions)

    def __len__(self) -> int:
        return len(self._regions)

    def find(self, address: int) -> Optional[MemoryRegion]:
        for region in self._regions:
            if region.contains(address, 1):
                return region
        return None

    def contains(self, address: int, size: int = 1) -> bool:
        if size <= 0:
            return False
        end = address + size
        current = address
        for region in self._regions:
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
