"""
Проверка ответа языковой модели.

Зона ответственности: Person 1 (Lead / Архитектор).

ПРИНЦИП: модель формулирует, но не вычисляет. Каждое число в её ответе
обязано встречаться в данных решения. Если появилось число, которого в
решении нет, — это выдумка, и оператору она уходит либо с явной пометкой,
либо не уходит вовсе (config/llm.yaml, ключ strict).

Это тот же принцип, на котором стоит Gate: убедительность текста не даёт
никаких прав. Gate не пропускает опасный вариант из-за красивого
объяснения, а здесь объяснение не может добавить чисел к решению.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

# Числа вида 10, 9.74, -0.138, 0,5 (запятая как разделитель тоже бывает)
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")

# Допуск сравнения: модель округляет 0.262 до 0.26, и это не выдумка.
_TOLERANCE = 0.011


def extract_numbers(text: str) -> List[float]:
    out: List[float] = []
    for match in _NUMBER.finditer(text or ""):
        try:
            out.append(float(match.group().replace(",", ".")))
        except ValueError:
            continue
    return out


def unknown_numbers(text: str, allowed: Sequence[float]) -> List[float]:
    """
    Числа ответа, которых нет в данных решения.

    Округление разрешено: 0.26 при исходных 0.262 — то же число.
    Проценты тоже: 29% в тексте против 0.291 в данных.
    """
    known = list(allowed)
    out: List[float] = []
    for value in extract_numbers(text):
        if _matches(value, known):
            continue
        out.append(value)
    return out


def _matches(value: float, known: Iterable[float]) -> bool:
    """
    Послабления только те, которые не пропускают выдумку.

    Проценты и округление до целого разрешены лишь для ЦЕЛЫХ чисел в
    тексте: иначе правило «0.03 в данных это 3% в тексте» принимало за
    своё любое число около трёх, включая выдуманное 3.14.
    """
    is_whole = abs(value - round(value)) < 1e-9

    for candidate in known:
        if abs(value - candidate) <= _TOLERANCE:
            return True
        if not is_whole:
            continue
        # доля 0.291 в данных -> "29%" в тексте
        if abs(candidate) <= 1.0 and abs(value - candidate * 100.0) <= 0.51:
            return True
        # округление до целого: запас 0.88 -> "около 1"
        if abs(value - round(candidate)) < 1e-9 and abs(value - candidate) < 0.5:
            return True
    return False
