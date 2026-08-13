#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
integrated_node/run_demo.py
SafeNest V4 On-Device AI Interactive Verification Simulator
"""

from __future__ import annotations
import sys
import time
import argparse
from pathlib import Path

from integrated_node.run_node import SafeNestIntegratedNode


def run_demo_simulation(steps: int = 10):
    print("=" * 60)
    print("  SafeNest V4 On-Device AI Demo Simulator")
    print("=" * 60)
    node = SafeNestIntegratedNode(mode="mock")
    node.start()

    scenarios = [
        ("0: NORMAL", {"thermal44": "NORMAL", "mmwave": "NORMAL", "co2": "NORMAL", "pir": "MOTION"}),
        ("1: CO2 ELEVATED", {"thermal44": "NORMAL", "mmwave": "NORMAL", "co2": "ELEVATED", "pir": "MOTION"}),
        ("2: HUMAN FALL", {"thermal44": "FALL", "mmwave": "NORMAL", "co2": "NORMAL", "pir": "MOTION"}),
        ("3: APNEA", {"thermal44": "NORMAL", "mmwave": "APNEA", "co2": "NORMAL", "pir": "NO_MOTION"}),
    ]

    for name, scen in scenarios:
        print(f"\n--- Triggering Scenario: {name} ---")
        node.sensors["thermal44"].set_scenario(scen["thermal44"])
        node.sensors["mmwave"].set_scenario(scen["mmwave"])
        node.sensors["co2"].set_scenario(scen["co2"])
        node.sensors["pir"].set_scenario(scen["pir"])

        output = node.step()
        print(f"Risk Score: {output.risk_score:.1f} | Level: {output.level} | System Status: {output.system_status}")
        print(f"Reasons: {output.reasons}")
        time.sleep(0.2)

    node.shutdown()
    print("\n✅ Simulation demo completed successfully.")


if __name__ == "__main__":
    run_demo_simulation()
