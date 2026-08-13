#!/usr/bin/env python3
"""Generate deterministic SafeNest mmWave M-B10A setup evidence."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mmwave_m_b10a_selection import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
