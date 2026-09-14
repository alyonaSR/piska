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
from .loaders import FEED_POINT, TARGET_POINT, detect_stuck
from .tags import load_config, refusal_rules

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
        "242000:T5": 365.1,     # температура ГСС на выходе Р-201
        "242000:F26": 256.2,    # расход сырья объёмный
        "242000:P24": 0.59,     # свежий ВСГ
        "242000:T18": 68.7,     # похоже на ВА вспышки
        "AVT:T33": 348.2,       # низ К-2
        "AVT:F30": 61.4,        # отбор фр.290-350
        "AVT:F32": 44.0,        # отбор фр.240-290
        "AVT:F65": 219.3,       # производительность К-2
        "AVT:P67": 1.10,        # давление верха К-2
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
        tags["AVT:F30"] = 66.5
        tags["AVT:T33"] = 355.0
        tags["242000:T5"] = 361.0
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


TAG_STALE_MIN = 30.0     # тег старше получаса -> кладём, но поднимаем флаг


def build_state(
    ts: datetime,
    telemetry: pd.DataFrame,
    lims_long: pd.DataFrame,
    pak_long: pd.DataFrame,
) -> ProcessState:
    """
    РЕАЛЬНАЯ сборка ProcessState на момент ts.

    ЕДИНСТВЕННОЕ ЖЁСТКОЕ ПРАВИЛО: ничего строго позже ts.
    Ни телеметрии, ни ЛИМС, ни ПАК. Никакой интерполяции, никакого bfill.
    Всё, что попадает в состояние, было известно оператору в момент ts.

    telemetry — DatetimeIndex, колонки вида 'AVT:T33'
    lims_long — длинный фрейм из load_lims()
    pak_long  — длинный фрейм из load_pak()
    """
    ts = pd.Timestamp(ts)
    flags: List[str] = []

    tags = _slice_telemetry(ts, telemetry, flags)
    lims = _latest_lims(ts, lims_long, flags)
    pak = _latest_pak(ts, pak_long, flags)

    return ProcessState(ts=ts.to_pydatetime(), tags=tags, lims=lims, pak=pak, dq_flags=flags)


def _slice_telemetry(ts, telemetry: pd.DataFrame, flags: List[str]) -> Dict[str, float]:
    """
    По каждому тегу — последнее непустое значение не позже ts.

    Именно по каждому, а не одна строка целиком: в строке на ts часть
    датчиков может быть в NaN, и брать её как есть значило бы потерять
    половину состояния.
    """
    tags: Dict[str, float] = {}
    if telemetry is None or telemetry.empty:
        flags.append("телеметрия не передана")
        return tags

    past = telemetry.loc[:ts]
    if past.empty:
        flags.append(f"нет телеметрии до {ts}")
        return tags

    stale = []
    for col in past.columns:
        idx = past[col].last_valid_index()
        if idx is None:
            flags.append(f"{col}: нет ни одного значения до {ts}")
            continue
        tags[col] = float(past.at[idx, col])
        age_min = (ts - idx).total_seconds() / 60.0
        if age_min > TAG_STALE_MIN:
            stale.append(f"{col} ({age_min / 60:.1f} ч)")

    if stale:
        head = ", ".join(stale[:5])
        tail = f" и ещё {len(stale) - 5}" if len(stale) > 5 else ""
        flags.append(f"устаревшие теги: {head}{tail}")
    return tags


def _latest_lims(ts, lims_long: pd.DataFrame, flags: List[str]) -> Dict[str, Measurement]:
    """
    Последний лабораторный результат по каждому показателю товарной точки.

    Сера в сырье гидроочистки кладётся под префиксом feed:, иначе она
    перезатёрла бы товарную (9460 мг/кг против 8.6) и всё поехало бы.
    """
    out: Dict[str, Measurement] = {}
    if lims_long is None or lims_long.empty:
        flags.append("ЛИМС не передан")
        return out

    max_age = float(refusal_rules().get("max_lims_age_min", 1440))

    for point, prefix in ((TARGET_POINT, ""), (FEED_POINT, "feed:")):
        sub = lims_long[(lims_long["sample_point"] == point) & (lims_long["ts"] <= ts)]
        if prefix == "feed:":
            sub = sub[sub["param"] == "sulfur_mgkg"]
        for param, grp in sub.groupby("param"):
            row = grp.loc[grp["ts"].idxmax()]
            age = (ts - row["ts"]).total_seconds() / 60.0
            out[f"{prefix}{param}"] = Measurement(
                value=float(row["value"]),
                ts=row["ts"].to_pydatetime(),
                age_min=round(age, 1),
                source="LIMS",
                units=str(row["units"]),
                healthy=not bool(row.get("outlier", False)),
            )

    key = "sulfur_mgkg"
    if key not in out:
        flags.append(f"ЛИМС: нет ни одного анализа серы до {ts}")
    elif out[key].age_min > max_age:
        flags.append(
            f"ЛИМС sulfur старше {max_age / 60:.0f} ч: {out[key].age_min / 60:.1f} ч"
        )
    return out


def _latest_pak(ts, pak_long: pd.DataFrame, flags: List[str]) -> Dict[str, Measurement]:
    """
    Последнее показание каждого поточного анализатора.

    Значение кладётся ВСЕГДА, даже если анализатор залип: контракт
    требует сохранить его с healthy=False, а не выбросить. Решение,
    доверять ли ему, принимает потребитель.
    """
    out: Dict[str, Measurement] = {}
    if pak_long is None or pak_long.empty:
        flags.append("ПАК не передан")
        return out

    for param, grp in pak_long[pak_long["ts"] <= ts].groupby("param"):
        row = grp.loc[grp["ts"].idxmax()]
        healthy = bool(row.get("healthy", True)) and not bool(row.get("outlier", False))
        age = (ts - row["ts"]).total_seconds() / 60.0
        out[param] = Measurement(
            value=float(row["value"]),
            ts=row["ts"].to_pydatetime(),
            age_min=round(age, 1),
            source="PAK",
            units=str(row["units"]),
            healthy=healthy,
        )
        if not healthy:
            flags.append(f"ПАК {param}: анализатор признан неисправным (залипание)")

    for expected in ("sulfur_mgkg", "d15_kgm3"):
        if expected not in out:
            flags.append(f"ПАК {expected} отсутствует на {ts}")
    return out


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