#!/usr/bin/env python3
"""Architecture flag metadata and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

from ..base import Arch


FlagDecoder = Callable[[int], Dict[str, bool]]
FlagEncoder = Callable[[Dict[str, bool]], int]
FlagNormalizer = Callable[[int], int]


def _identity(value: int) -> int:
    return value


@dataclass(frozen=True)
class FlagField:
    """Source field within a state object containing flag bits."""

    name: str
    normalize: FlagNormalizer = _identity


@dataclass(frozen=True)
class FlagSpec:
    """Architecture-specific description of processor flags."""

    aggregate: str
    aliases: Tuple[str, ...]
    bit_names: Tuple[str, ...]
    default: int
    state_fields: Tuple[FlagField, ...]
    decode: FlagDecoder
    encode: FlagEncoder


def decode_arm64_cpsr(cpsr: int) -> Dict[str, bool]:
    """Decode ARM64 CPSR/PSTATE value into NZCV flags."""

    return {
        "n": bool(cpsr & (1 << 31)),
        "z": bool(cpsr & (1 << 30)),
        "c": bool(cpsr & (1 << 29)),
        "v": bool(cpsr & (1 << 28)),
    }


def encode_arm64_cpsr(flags: Dict[str, bool]) -> int:
    """Encode NZCV flag dictionary back into CPSR bits."""

    cpsr = 0
    if flags.get("n", False):
        cpsr |= 1 << 31
    if flags.get("z", False):
        cpsr |= 1 << 30
    if flags.get("c", False):
        cpsr |= 1 << 29
    if flags.get("v", False):
        cpsr |= 1 << 28
    return cpsr


def decode_x86_flags(eflags: int) -> Dict[str, bool]:
    """Decode x86/x64 EFLAGS value into individual bits."""

    return {
        "cf": bool(eflags & (1 << 0)),
        "pf": bool(eflags & (1 << 2)),
        "af": bool(eflags & (1 << 4)),
        "zf": bool(eflags & (1 << 6)),
        "sf": bool(eflags & (1 << 7)),
        "df": bool(eflags & (1 << 10)),
        "of": bool(eflags & (1 << 11)),
    }


def encode_x86_flags(flags: Dict[str, bool]) -> int:
    """Encode individual x86/x64 flag bits into an EFLAGS value."""

    eflags = 0
    if flags.get("cf", False):
        eflags |= 1 << 0
    if flags.get("pf", False):
        eflags |= 1 << 2
    if flags.get("af", False):
        eflags |= 1 << 4
    if flags.get("zf", False):
        eflags |= 1 << 6
    if flags.get("sf", False):
        eflags |= 1 << 7
    if flags.get("df", False):
        eflags |= 1 << 10
    if flags.get("of", False):
        eflags |= 1 << 11
    return eflags


def _extract_nzcv(value: int) -> int:
    """Return only NZCV bits from a CPSR/PSTATE value."""

    return value & 0xF0000000


FLAG_SPECS: Dict[Arch, FlagSpec] = {
    Arch.X86: FlagSpec(
        aggregate="eflags",
        aliases=("rflags",),
        bit_names=("cf", "pf", "af", "zf", "sf", "df", "of"),
        default=0x202,
        state_fields=(FlagField("eflags"), FlagField("rflags")),
        decode=decode_x86_flags,
        encode=encode_x86_flags,
    ),
    Arch.X64: FlagSpec(
        aggregate="rflags",
        aliases=("eflags",),
        bit_names=("cf", "pf", "af", "zf", "sf", "df", "of"),
        default=0x202,
        state_fields=(FlagField("rflags"), FlagField("eflags")),
        decode=decode_x86_flags,
        encode=encode_x86_flags,
    ),
    Arch.ARM64: FlagSpec(
        aggregate="nzcv",
        aliases=("cpsr",),
        bit_names=("n", "z", "c", "v"),
        default=0x0,
        state_fields=(FlagField("nzcv"), FlagField("cpsr", _extract_nzcv)),
        decode=lambda value: decode_arm64_cpsr(value),
        encode=lambda flags: encode_arm64_cpsr(flags),
    ),
}


def get_flag_spec(arch: Arch) -> Optional[FlagSpec]:
    """Return the flag specification for an architecture."""

    return FLAG_SPECS.get(arch)


def get_flag_register_names(arch: Arch) -> Tuple[str, ...]:
    """Return canonical aggregate flag register and aliases."""

    spec = get_flag_spec(arch)
    if not spec:
        return tuple()
    return (spec.aggregate,) + spec.aliases


def resolve_flag_register(arch: Arch, name: str) -> Optional[str]:
    """Return canonical aggregate name if *name* refers to flags."""

    spec = get_flag_spec(arch)
    if not spec:
        return None

    normalized = name.lower()
    if normalized == spec.aggregate:
        return spec.aggregate
    for alias in spec.aliases:
        if normalized == alias:
            return spec.aggregate
    return None


def get_default_flag_value(arch: Arch) -> Optional[int]:
    """Return the default architectural flag value, if defined."""

    spec = get_flag_spec(arch)
    return spec.default if spec else None


def find_flag_value_in_state(arch: Arch, state: object) -> Optional[int]:
    """Extract a flag word from a GPR state object if available."""

    spec = get_flag_spec(arch)
    if not spec:
        return None

    for field in spec.state_fields:
        if hasattr(state, field.name):
            raw_value = getattr(state, field.name)
            return field.normalize(raw_value)
    return None


def iter_flag_bits(arch: Arch) -> Iterable[str]:
    """Yield the bit names defined for the architecture."""

    spec = get_flag_spec(arch)
    return spec.bit_names if spec else tuple()


__all__ = [
    "FlagSpec",
    "FlagField",
    "decode_arm64_cpsr",
    "encode_arm64_cpsr",
    "decode_x86_flags",
    "encode_x86_flags",
    "get_flag_spec",
    "get_flag_register_names",
    "resolve_flag_register",
    "get_default_flag_value",
    "find_flag_value_in_state",
    "iter_flag_bits",
]
