"""
Трейс-логирование цикла.

Зона ответственности: Person 1.

Каждый прогон пишет полный JSON: вход -> оценки агентов -> все кандидаты
-> вердикты Gate -> финал. Закрывает критерии ТЗ 'Воспроизводимость'
и 'Объяснимость': логику любого решения можно проверить постфактум.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from .contracts import DecisionTrace

TRACE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "traces"
)


def save_trace(trace: DecisionTrace, tag: str = "") -> str:
    os.makedirs(TRACE_DIR, exist_ok=True)
    name = f"{trace.ts:%Y%m%dT%H%M}{'_' + tag if tag else ''}.json"
    path = os.path.join(TRACE_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace.to_dict(), f, ensure_ascii=False, indent=2)
    return path


def load_trace(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
