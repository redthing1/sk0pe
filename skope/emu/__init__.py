"""skope.emu - emulation components with optional backends."""

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:  # pragma: no cover (typing only)
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
    # concrete implementations (loaded lazily)
    "RawCodeBlob",
    "UnicornEmulator",
    "TritonEmulator",
    "MaatEmulator",
]


def __getattr__(name):
    """Lazily import optional backends when requested."""

    if name == "UnicornEmulator":
        try:
            from .unicorn import UnicornEmulator
        except ModuleNotFoundError as exc:  # pragma: no cover (runtime error path)
            raise ModuleNotFoundError(
                "Unicorn backend requires the 'unicorn' extra: install with skope[unicorn]"
            ) from exc
        return UnicornEmulator

    if name == "TritonEmulator":
        try:
            from .triton import TritonEmulator
        except ModuleNotFoundError as exc:  # pragma: no cover (runtime error path)
            raise ModuleNotFoundError(
                "Triton backend requires the 'triton' extra: install with skope[triton]"
            ) from exc
        return TritonEmulator

    if name == "MaatEmulator":
        try:
            from .maat import MaatEmulator
        except ModuleNotFoundError as exc:  # pragma: no cover (runtime error path)
            raise ModuleNotFoundError(
                "Maat backend requires the 'maat' extra: install with skope[maat]"
            ) from exc
        return MaatEmulator

    raise AttributeError(name)
