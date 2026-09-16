"""
Доли компонентов дизельного пула.

Зона ответственности: Person 1 (Lead / Архитектор).

Явных компонентов смешения в выданных данных нет, поэтому принято
модельное допущение: дизельный пул = два потока с АВТ, тяжёлый AVT:F30
(фр. 290-350 C) и лёгкий AVT:F32 (фр. 240-290 C). Доли — их массовая
пропорция. Состав и границы долей лежат в config/constraints.yaml,
раздел blending.

ЗАЧЕМ ЭТО НУЖНО GATE: изменение отборов меняет пропорцию компонентов,
а ТЗ требует жёсткой проверки "доли компонентов составляют 100%".
Сама по себе сумма долей равна единице по построению, поэтому Gate
дополнительно проверяет, что доля тяжёлого компонента остаётся в
исторически наблюдавшемся окне: иначе проверка была бы тавтологией.

ЧТО СОЗНАТЕЛЬНО НЕ ВОШЛО: присадки и вовлечение других фракций.
В выданном пакете нет ни расходов присадок, ни рецептур смешения,
а выдумывать компоненты, которых нет в данных, хуже, чем честно
ограничить пул двумя потоками и пометить это как допущение.
"""

from __future__ import annotations

from typing import Dict, Optional

from .contracts import ProcessState
from .data.tags import load_config


def blend_fractions(
    state: ProcessState, deltas: Optional[Dict[str, float]] = None
) -> Optional[Dict[str, float]]:
    """
    Доли компонентов после применения дельт кандидата.

    None означает "не знаю": нет расхода хотя бы одного компонента или
    суммарный расход ниже min_pool_tph. Gate получает честное незнание
    вместо деления на почти ноль — проверка при этом просто не
    выполняется, а не проходит молча.
    """
    cfg = load_config("constraints")["blending"]
    deltas = deltas or {}

    flows: Dict[str, float] = {}
    for tag in cfg["components"]:
        current = state.tag(tag)
        if current is None:
            return None
        flows[tag] = current + deltas.get(tag, 0.0)

    total = sum(flows.values())
    if total < float(cfg["min_pool_tph"]):
        return None

    return {tag: flow / total for tag, flow in flows.items()}
