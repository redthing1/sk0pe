class SkopeError(Exception):
    """Base exception for skope errors."""


class EmulationError(SkopeError):
    """Raised when emulation fails or halts unexpectedly."""


class MemoryError(SkopeError):
    """Raised when memory access or mapping fails."""


class ArchitectureError(SkopeError):
    """Raised for unsupported or mismatched architectures."""
