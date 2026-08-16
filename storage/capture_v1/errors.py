"""Deterministic Capture validation result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CaptureValidationIssue:
    path: str
    code: str
    message: str

    def format(self) -> str:
        return f"{self.path}: {self.code}: {self.message}"


@dataclass(frozen=True)
class CaptureValidationResult:
    errors: tuple[CaptureValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def format_errors(self) -> str:
        return "\n".join(issue.format() for issue in self.errors)


def merge(*results: CaptureValidationResult | Sequence[CaptureValidationIssue]) -> CaptureValidationResult:
    collected: list[CaptureValidationIssue] = []
    for result in results:
        if isinstance(result, CaptureValidationResult):
            collected.extend(result.errors)
        else:
            collected.extend(result)
    return CaptureValidationResult(tuple(collected))
