"""
skope.emu - emulation components
"""

from .base import (
    Arch,
    Hook,
    Permission,
    MemoryRegion,
    Segment,
    Executable,
    RawCodeBlob,
    Emulator,
    BareMetalEmulator,
)
from .unicorn import UnicornEmulator
from .triton import TritonEmulator
from .maat import MaatEmulator

__all__ = [
    # base types
    "Arch",
    "Hook",
    "Permission",
    "MemoryRegion",
    "Segment",
    # abstract interfaces
    "Executable",
    "Emulator",
    "BareMetalEmulator",
    # concrete implementations
    "RawCodeBlob",
    "UnicornEmulator",
    "TritonEmulator",
    "MaatEmulator",
]
