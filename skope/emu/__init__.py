"""
skope.emu - emulation components
"""

from .base import (
    Arch,
    Hook,
    MemoryRegion,
    Segment,
    Executable,
    RawCodeBlob,
    Emulator,
    BareMetalEmulator,
)
from .unicorn import UnicornEmulator
from .miasm import MiasmJitterEmulator

__all__ = [
    # base types
    "Arch",
    "Hook", 
    "MemoryRegion",
    "Segment",
    # abstract interfaces
    "Executable",
    "Emulator",
    "BareMetalEmulator",
    # concrete implementations
    "RawCodeBlob",
    "UnicornEmulator",
    "MiasmJitterEmulator",
]
