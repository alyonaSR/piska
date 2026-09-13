"""
Резолвер тегов и загрузка конфигов.

Зона ответственности: Person 2 (Data Engineer).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Tuple

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "config")


@lru_cache(maxsize=None)
def load_config(name: str) -> Dict[str, Any]:
    with open(os.path.join(CONFIG_DIR, f"{name}.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_tag(key: str) -> Tuple[str, str]:
    """'242000:T5' -> ('242000', 'T5'). Без установки — ошибка, и это намеренно."""
    if ":" not in key:
        raise KeyError(
            f"Тег '{key}' без установки. Код T6 на АВТ и на 24-2000 — разные величины."
        )
    unit, code = key.split(":", 1)
    return unit, code


def make_tag(unit: str, code: str) -> str:
    return f"{unit}:{code}"


def tag_info(key: str) -> Dict[str, Any]:
    cat = load_config("tags").get("catalog", {})
    return cat.get(key, {"desc": "нет в справочнике", "units": "?", "verified": False})


def is_verified(key: str) -> bool:
    return bool(tag_info(key).get("verified", False))


def manipulated_vars() -> Dict[str, Any]:
    return load_config("constraints")["manipulated_vars"]


def quality_specs() -> Dict[str, Any]:
    return load_config("constraints")["quality_specs"]


def refusal_rules() -> Dict[str, Any]:
    return load_config("constraints")["refusal_rules"]
