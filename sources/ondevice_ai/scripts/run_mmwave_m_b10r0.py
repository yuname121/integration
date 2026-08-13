#!/usr/bin/env python3
"""Command-line entry point for M-B10R0 holdout policy evidence generation."""

from __future__ import annotations

import sys

from mmwave_m_b10r0_holdout_policy import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
