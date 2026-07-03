"""Shared PASS/FAIL sanity-check framework, used by both Compressor
Calculations and Turbine Calculations -- a single SanityCheck record type,
bounds-checking helper, and report formatter, so each machine's module only
needs to supply its own default limits and which fields to check.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SanityCheck:
    name: str
    value: float
    low: float | None
    high: float | None
    unit: str

    @property
    def passed(self) -> bool:
        if self.low is not None and self.value < self.low:
            return False
        if self.high is not None and self.value > self.high:
            return False
        return True


def check(name, value, bounds, unit=""):
    low, high = bounds
    return SanityCheck(name=name, value=value, low=low, high=high, unit=unit)


def format_sanity_report(title: str, checks) -> str:
    lines = [f"\n===== {title} ====="]
    all_passed = True
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        all_passed = all_passed and c.passed
        if c.low is not None and c.high is not None:
            limit_str = f"{c.low} - {c.high}"
        elif c.low is not None:
            limit_str = f"> {c.low}"
        elif c.high is not None:
            limit_str = f"< {c.high}"
        else:
            limit_str = "-"
        lines.append(f"{c.name:28s} | limit: {limit_str:15s} | value: {c.value:10.4f} {c.unit:12s} | {status}")
    lines.append("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED -- review design point")
    return "\n".join(lines)
