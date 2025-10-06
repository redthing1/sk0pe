#!/usr/bin/env python3
"""Architecture-specific utilities exposed to emulator backends."""

from __future__ import annotations

from .registers import (
    REGISTER_NAMES,
    DEFAULT_STACK_BASE,
    get_register_names,
    get_pc_register,
    get_sp_register,
    get_return_register,
    get_word_size,
    get_stack_alignment,
)
from .flags import (
    FlagSpec,
    FlagField,
    decode_arm64_cpsr,
    encode_arm64_cpsr,
    decode_x86_flags,
    encode_x86_flags,
    get_flag_spec,
    get_flag_register_names,
    resolve_flag_register,
    get_default_flag_value,
    find_flag_value_in_state,
    iter_flag_bits,
)

__all__ = [
    # Register metadata
    "REGISTER_NAMES",
    "DEFAULT_STACK_BASE",
    "get_register_names",
    "get_pc_register",
    "get_sp_register",
    "get_return_register",
    "get_word_size",
    "get_stack_alignment",
    # Flag metadata
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
