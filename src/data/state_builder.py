"""
L0. Сборка ProcessState.

Зона ответственности: Person 2 (Data Engineer).

Пока реальный источник не подключён, build_demo_state() отдаёт
правдоподобные состояния для трёх сценариев из ТЗ. Все остальные
слои работают на них уже сегодня.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from ..contracts import Measurement, ProcessState
from .loaders import detect_stuck
from .tags import load_config

# Три обязательных демо-сценария из раздела 6 ТЗ
SCENARIOS = ("normal", "quality_risk", "degraded_data")


def build_demo_state(scenario: str = "normal", ts: Optional[datetime] = None) -> ProcessState:
    """
    ЗАГЛУШКА. Замена на build_state() не затрагивает другие слои:
    возвращается тот же ProcessState.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Сценарий должен быть из {SCENARIOS}")
    ts = ts or datetime(2026, 3, 14, 8, 20)

    # базовый режим, значения взяты из реальных диапазонов телеметрии
    tags: Dict[str, float] = {
        # медианы по очищенной истории 2023-2026, см. config/constraints.yaml
        "242000:T5": 370.4,     # температура реактора Р-201
        "242000:F26": 258.6,    # расход сырья объёмный
        "242000:T18": 68.7,     # похоже на ВА вспышки
        "AVT:F30": 128.4,       # отбор фр.290-350
        "AVT:F32": 81.7,        # отбор фр.240-290
        "AVT:F28": 275.6,       # пар в К-9
        "AVT:F14": 253.5,       # 1 ЦО
        "AVT:P22": 1.12,        # давление верха К-2
        "AVT:T33": 338.3,       # низ К-2 (признак, не управляемая)
        "AVT:T71": 304.3,       # температура отбора ДТ (признак)
        "AVT:F65": 922.4,       # производительность К-2 (возмущение)
    }
    dq_flags: List[str] = []

    lims = {
        "sulfur_mgkg": Measurement(8.6, ts - timedelta(hours=6), 360.0, "LIMS", "mg/kg"),
        "flash_c": Measurement(68.0, ts - timedelta(hours=6), 360.0, "LIMS", "degC"),
        "cfpp_c": Measurement(-6.0, ts - timedelta(hours=30), 1800.0, "LIMS", "degC"),
        "d15_kgm3": Measurement(836.1, ts - timedelta(hours=6), 360.0, "LIMS", "kg/m3"),
    }
    pak = {
        "sulfur_mgkg": Measurement(8.4, ts, 0.0, "PAK", "mg/kg"),
        "d15_kgm3": Measurement(835.8, ts, 0.0, "PAK", "kg/m3"),
    }

    if scenario == "quality_risk":
        # режим утяжелился: больше отбор ДТ, ниже температура реактора
        tags["AVT:F30"] = 141.0     # отбор поднят, хвост тяжелее
        tags["AVT:F32"] = 89.0
        tags["242000:T5"] = 366.5   # температура реактора ниже обычной
        pak["sulfur_mgkg"] = Measurement(9.6, ts, 0.0, "PAK", "mg/kg")
        lims["sulfur_mgkg"] = Measurement(9.4, ts - timedelta(hours=9), 540.0, "LIMS", "mg/kg")

    if scenario == "degraded_data":
        # ЛИМС протух, ПАК залип
        lims["sulfur_mgkg"] = Measurement(8.2, ts - timedelta(hours=52), 3120.0, "LIMS", "mg/kg")
        pak["sulfur_mgkg"] = Measurement(8.37, ts, 0.0, "PAK", "mg/kg", healthy=False)
        pak["d15_kgm3"] = Measurement(None, None, None, "PAK", "kg/m3", healthy=False)
        dq_flags += [
            "PAK sulfur залип: 214 точек подряд без изменения (35.7 ч)",
            "LIMS sulfur старше 48 ч",
            "PAK D15 отсутствует",
        ]

    return ProcessState(ts=ts, tags=tags, lims=lims, pak=pak, dq_flags=dq_flags)


def build_state(
    ts: datetime,
    telemetry: pd.DataFrame,
    lims_long: pd.DataFrame,
    pak_long: pd.DataFrame,
) -> ProcessState:
    """
    РЕАЛЬНАЯ сборка.

    TODO(Person 2):
      1. срез телеметрии на ts (последняя точка <= ts), NaN -> в dq_flags
      2. для каждого показателя качества взять последнее значение ЛИМС и ПАК
         СТРОГО <= ts, посчитать age_min
      3. ПАК: проставить healthy через detect_stuck()
      4. никаких join по номеру строки
    """
    raise NotImplementedError("Person 2: сборка ProcessState из реальных источников")


def scan_data_quality(df: pd.DataFrame, freq_min: int = 10) -> List[str]:
    """Быстрый отчёт о качестве среза. Результат идёт в dq_flags."""
    flags: List[str] = []
    for col in df.columns:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        nan_share = float(s.isna().mean())
        if nan_share > 0.05:
            flags.append(f"{col}: пропусков {nan_share:.0%}")
        stuck = detect_stuck(s.dropna())
        if stuck.any():
            hours = stuck.sum() * freq_min / 60.0
            flags.append(f"{col}: залипание суммарно {hours:.0f} ч")
    return flags
