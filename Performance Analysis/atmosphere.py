"""Standard atmosphere for off-design performance sweeps -- re-exports
Brayton Cycle Analysis's validated isa_atmosphere (troposphere + isothermal
lower stratosphere) rather than duplicating it, and adds the standard hot-
/cold-day offset convention used in performance analysis: a "hot day" or
"cold day" atmosphere is the same ISA pressure profile with a constant
temperature offset (MIL-STD-210 / ISA+dT convention) -- pressure altitude
is unaffected by the temperature shift, only static temperature changes.
"""

import importlib.util
from pathlib import Path

# Loaded by file path, not by module name: this folder has its own
# mission.py (flight envelope grids), so `import mission`/`from mission
# import ...` would collide with -- and, depending on import order,
# silently resolve to the wrong file entirely -- Brayton Cycle Analysis's
# same-named module (confirmed live: sys.path alone doesn't fix this once
# "mission" is cached under either file in sys.modules).
_BRAYTON_MISSION_PATH = Path(__file__).resolve().parent.parent / "Brayton Cycle Analysis" / "mission.py"
_spec = importlib.util.spec_from_file_location("_brayton_mission", _BRAYTON_MISSION_PATH)
_brayton_mission = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_brayton_mission)
isa_atmosphere = _brayton_mission.isa_atmosphere

__all__ = ["isa_atmosphere", "isa_atmosphere_with_offset"]


def isa_atmosphere_with_offset(altitude_m: float, delta_T: float = 0.0):
    """(T [K], P [Pa]) at altitude_m, with a constant temperature offset
    applied (delta_T > 0 for a "hot day", < 0 for a "cold day"). Pressure
    is left at its standard-day value -- ISA+dT convention."""
    T, P = isa_atmosphere(altitude_m)
    return T + delta_T, P
