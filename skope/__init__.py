from __future__ import annotations

from typing import Optional

from .backends import EmulatorConfig, MaatEmulator, TritonEmulator, UnicornEmulator
from .snapshot import LiefSnapshot, Snapshot, W1DumpSnapshot
from .session import Session


def load_w1dump(path: str) -> Snapshot:
    return W1DumpSnapshot.load(path)


def open_w1dump(
    path: str, *, backend: str = "unicorn", config: Optional[EmulatorConfig] = None
) -> Session:
    snapshot = load_w1dump(path)
    emulator = _create_emulator(snapshot, backend, config)
    return Session(snapshot=snapshot, emulator=emulator)


def load_binary(path: str) -> Snapshot:
    return LiefSnapshot.load(path)


def open_binary(
    path: str, *, backend: str = "unicorn", config: Optional[EmulatorConfig] = None
) -> Session:
    snapshot = load_binary(path)
    emulator = _create_emulator(snapshot, backend, config)
    return Session(snapshot=snapshot, emulator=emulator)


def _create_emulator(
    snapshot: Snapshot, backend: str, config: Optional[EmulatorConfig]
):
    backend_key = backend.lower()
    if config is None:
        config = EmulatorConfig()
    if backend_key == "maat" and config.map_zero_page is None:
        config.map_zero_page = True
    if backend_key == "unicorn":
        return UnicornEmulator(snapshot, config)
    if backend_key == "triton":
        return TritonEmulator(snapshot, config)
    if backend_key == "maat":
        return MaatEmulator(snapshot, config)
    raise ValueError(f"unsupported backend: {backend}")


__all__ = [
    "EmulatorConfig",
    "Session",
    "load_binary",
    "load_w1dump",
    "open_binary",
    "open_w1dump",
]
