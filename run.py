#!/usr/bin/env python3
"""
Точка входа.

    python run.py demo              все три сценария ТЗ
    python run.py demo --scenario quality_risk
    python run.py demo --seed 42

    python run.py ask "почему не подняли температуру сильнее?"
    python run.py ask --scenario normal "почему ничего не меняем?"

Воспроизводимость: seed фиксируется, трейсы пишутся в artifacts/traces/.
Команда ask объясняет УЖЕ принятое решение и на него не влияет: без ключа
языковой модели она отвечает шаблоном по тем же числам.
"""

from __future__ import annotations

import argparse
import glob
import os
import random

import numpy as np

from src.data.state_builder import SCENARIOS, build_demo_state
from src.explain import print_operator_report
from src.llm import ask
from src.orchestrator import Orchestrator
from src.tracing import TRACE_DIR, load_trace, recommendation_from_dict, save_trace

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

    # Обмен между агентами печатает сам отчёт (блок 10): он часть ответа
    # системы, а не отладочный вывод демо-скрипта.
    print()
    print(print_operator_report(trace.recommendation, trace))
    path = save_trace(trace, tag=name)
    print(f"\ntrace -> {path}")


def load_demo_telemetry():
    """Обе установки в одном кадре: агент надёжности смотрит и АВТ, и 24-2000."""
    import pandas as pd

    from src.data.loaders import load_telemetry

    print("загрузка телеметрии из data/ ...")
    frame = pd.concat([load_telemetry("AVT"), load_telemetry("242000")], axis=1)
    print(f"  {len(frame)} точек, {frame.index.min():%Y-%m-%d} .. {frame.index.max():%Y-%m-%d}")
    return frame


def run_question(question: str, scenario: str, orch: Orchestrator) -> None:
    trace = orch.run_cycle(build_demo_state(scenario))
    answer = ask(question, trace)

    print("\n" + "=" * 78)
    print(f"ВОПРОС ОПЕРАТОРА ({scenario}): {question}")
    print("=" * 78)
    print(answer.text)
    print(f"\n[источник ответа: {answer.source}]", end="")
    if answer.unverified:
        # Числа, которых нет в решении. Прятать нельзя: оператор должен
        # видеть, какой части текста верить не следует.
        print(f"  [!] не подтверждены данными: {answer.unverified}", end="")
    print()


def replay(path: str = None) -> None:
    """
    Отчёт из сохранённого трейса. Воспроизводимость по ТЗ — это в том
    числе возможность показать решение недельной давности как есть,
    ничего не пересчитывая.
    """
    if path is None:
        traces = sorted(glob.glob(os.path.join(TRACE_DIR, "*.json")))
        if not traces:
            raise SystemExit("трейсов нет: сначала `python run.py demo`")
        path = traces[-1]

    data = load_trace(path)
    meta = data.get("meta") or {}

    print(f"\nтрейс: {path}")
    if meta:
        print(f"записан {meta.get('записан')}, коммит {meta.get('код', {}).get('git_commit')}, "
              f"модели {meta.get('модели')}")
        print(f"конфиги: {meta.get('конфиги')}")
    print(print_operator_report(recommendation_from_dict(data["recommendation"])))


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--scenario", choices=list(SCENARIOS), default=None)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument(
        "--telemetry", action="store_true",
        help="подгрузить исторические ряды из data/: агент надёжности "
             "считает факторы со скользящим окном, без них работают "
             "только два фактора из пяти",
    )

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", parents=[common], help="сценарии раздела 6 ТЗ")
    ask_parser = sub.add_parser(
        "ask", parents=[common], help="вопрос по принятому решению"
    )
    ask_parser.add_argument("question", nargs="+", help="вопрос оператора")

    replay_parser = sub.add_parser("replay", help="отчёт из сохранённого трейса")
    replay_parser.add_argument("path", nargs="?", default=None,
                               help="путь к трейсу; без аргумента — последний")

    args = ap.parse_args()

    if args.command == "replay":
        replay(args.path)
        return

    random.seed(args.seed)
    np.random.seed(args.seed)

    orch = Orchestrator(telemetry=load_demo_telemetry() if args.telemetry else None)

    if args.command == "ask":
        run_question(" ".join(args.question), args.scenario or "quality_risk", orch)
        return

    for name in ([args.scenario] if args.scenario else SCENARIOS):
        run_scenario(name, orch)


if __name__ == "__main__":
    main()
