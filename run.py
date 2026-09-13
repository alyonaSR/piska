#!/usr/bin/env python3
"""
Точка входа.

    python run.py demo              все три сценария ТЗ
    python run.py demo --scenario quality_risk
    python run.py demo --seed 42

Воспроизводимость: seed фиксируется, трейсы пишутся в artifacts/traces/.
"""

from __future__ import annotations

import argparse
import random

import numpy as np

from src.data.state_builder import SCENARIOS, build_demo_state
from src.explain import print_operator_report
from src.orchestrator import Orchestrator
from src.tracing import save_trace

TITLES = {
    "normal": "СЦЕНАРИЙ 1. Устойчивый режим (лишних воздействий быть не должно)",
    "quality_risk": "СЦЕНАРИЙ 2. Риск ухудшения качества",
    "degraded_data": "СЦЕНАРИЙ 3. Неполные / устаревшие / аномальные данные",
}


def run_scenario(name: str, orch: Orchestrator) -> None:
    print("\n" + "#" * 78)
    print("#  " + TITLES[name])
    print("#" * 78)

    state = build_demo_state(name)
    trace = orch.run_cycle(state)

    print(f"\n-- обмен между агентами --")
    print(f"   QualityAssess     risk={trace.quality.spec_risk_prob}  "
          f"conf={trace.quality.confidence}  model={trace.quality.model_id}")
    print(f"   ReliabilityAssess severity={trace.reliability.severity_index} "
          f"({trace.reliability.severity_class})")
    print(f"   allowed_ranges    {trace.reliability.allowed_ranges}")
    print(f"   Candidates        сгенерировано {len(trace.candidates)}")
    print(f"   GateVerdicts      прошло {sum(v.passed for v in trace.verdicts)} "
          f"из {len(trace.verdicts)}")

    rejected = [v for v in trace.verdicts if not v.passed]
    if rejected:
        print(f"   пример отбраковки  {rejected[0].candidate_id}: {rejected[0].violated[0]}")

    print()
    print(print_operator_report(trace.recommendation))
    path = save_trace(trace, tag=name)
    print(f"\ntrace -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["demo"])
    ap.add_argument("--scenario", choices=list(SCENARIOS), default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    orch = Orchestrator()
    for name in ([args.scenario] if args.scenario else SCENARIOS):
        run_scenario(name, orch)


if __name__ == "__main__":
    main()
