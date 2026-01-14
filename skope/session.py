from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .backends.base import ExecutionResult
from .snapshot.base import Snapshot
from .backends.base import Emulator


@dataclass
class Session:
    snapshot: Snapshot
    emulator: Emulator

    def run(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        count: int = 0,
        timeout: int = 0,
    ) -> ExecutionResult:
        return self.emulator.run(start=start, end=end, count=count, timeout=timeout)
