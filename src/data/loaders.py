"""
Загрузка сырых источников.

Зона ответственности: Person 2 (Data Engineer).

СТАТУС: каркас. Функции читают файлы и приводят к единому виду,
но разбор шапки ЛИМС и калибровка ПАК помечены TODO.

ПРАВИЛА (из ТЗ, нарушать нельзя):
  - синхронизация ТОЛЬКО по времени, никогда по номеру строки
  - возраст анализа хранить всегда
  - единицы указывать явно
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .tags import load_config

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data")


def _path(fname: str) -> str:
    return os.path.join(DATA_DIR, fname)


def load_telemetry(unit: str) -> pd.DataFrame:
    """
    unit: 'AVT' | '242000'

    Возвращает DataFrame с DatetimeIndex и колонками вида 'AVT:T33'.
    Служебные колонки Unnamed:* выбрасываются.
    """
    cfg = load_config("tags")
    fname = {"AVT": "avt_tags.csv", "242000": "242000_tags.csv"}[unit]
    df = pd.read_csv(_path(fname))

    df = df.drop(columns=[c for c in cfg["service_columns"] if c in df.columns])
    df[cfg["time_key"]] = pd.to_datetime(df[cfg["time_key"]])
    df = df.set_index(cfg["time_key"]).sort_index()

    df = _clean_sentinels(df, unit, cfg)
    df.columns = [f"{unit}:{c}" for c in df.columns]
    return df


def _clean_sentinels(df: pd.DataFrame, unit: str, cfg: dict) -> pd.DataFrame:
    """
    Значения-маркеры и физически невозможные значения -> NaN.

    Найдено в данных 24-2000: у восьми колонок максимум ровно 307.00,
    у трёх ровно 10.00. Одинаковый максимум у физически разных величин
    это не измерение. Расходы с отрицательным знаком (F26 до -23.75)
    тоже не измерение.
    """
    out = df.copy()
    sentinels = cfg.get("sentinel_values", [])
    neg_prefixes = tuple(cfg.get("reject_negative_prefixes", []))

    for col in out.columns:
        s = out[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        for sv in sentinels:
            # маркер считаем маркером, только если он реально является максимумом
            if np.isclose(s.max(skipna=True), sv, atol=1e-6):
                s = s.mask(np.isclose(s, sv, atol=1e-6))
        if col.startswith(neg_prefixes):
            s = s.mask(s < 0)
        out[col] = s
    return out


def load_lims(path: Optional[str] = None) -> pd.DataFrame:
    """
    Разбор ЛИМС в длинный формат: ts | sample_point | param | value | units

    TODO(Person 2):
      Шапка занимает 3 строки, колонки идут парами (дата, значение),
      блоки точек отбора заданы объединёнными ячейками в строке 0.
      ВНИМАНИЕ: строка единиц НЕ выровнена со строкой названий параметров
      (50%.T подписан как кг/м3). Единицы восстанавливать по смыслу
      параметра, а не по позиции.

      Целевая точка отбора для товарного продукта:
        "Установка 'Гидроочистка'.. Точка отбора '2'. Продукт 'Дизельное топливо'"
        там Mg.Sulfur (1462 значения, мг/кг) — главный жёсткий показатель.
    """
    raise NotImplementedError("Person 2: разбор шапки ЛИМС")


def load_pak(path: Optional[str] = None) -> pd.DataFrame:
    """
    Поточные анализаторы в длинный формат: ts | param | value | units | healthy

    TODO(Person 2):
      Колонки парами (дата, значение). Сера с 2023-01-01, D15 только с 2025-03-05.
      Обязательно проставить healthy через detect_stuck(): в ряду серы найден
      эпизод длиной 6731 точка (1122 часа) с нулевым приращением.
    """
    raise NotImplementedError("Person 2: разбор ПАК")


def detect_stuck(s: pd.Series, min_points: int = 6) -> pd.Series:
    """
    Детектор залипшего анализатора. Возвращает булеву маску 'значение мёртвое'.

    min_points=6 при шаге 10 минут = один час без изменения значения.
    Реализация целиком, без заглушки: это готовый демо-сценарий
    'неполные или аномальные данные' из ТЗ.
    """
    same_as_prev = s.diff().eq(0)
    grp = (~same_as_prev).cumsum()
    run_len = same_as_prev.groupby(grp).transform("sum")
    return same_as_prev & (run_len >= min_points)


def asof_join(
    telemetry: pd.DataFrame,
    lab: pd.DataFrame,
    param: str,
    tolerance_h: float = 72.0,
) -> pd.DataFrame:
    """
    As-of join лабораторных данных к телеметрии. ТОЛЬКО назад по времени.

    Добавляет колонки <param>__value и <param>__age_min.
    Возраст анализа — обязательное поле, не примечание: если ЛИМС
    18-часовой давности, а режим меняли 4 часа назад, этот анализ
    ничего не говорит о текущем продукте.
    """
    left = telemetry.reset_index().rename(columns={telemetry.index.name or "index": "ts"})
    right = (
        lab[lab["param"] == param][["ts", "value"]]
        .dropna()
        .sort_values("ts")
        .rename(columns={"value": f"{param}__value"})
    )
    right[f"{param}__meas_ts"] = right["ts"]

    merged = pd.merge_asof(
        left.sort_values("ts"),
        right,
        on="ts",
        direction="backward",
        tolerance=pd.Timedelta(hours=tolerance_h),
    )
    merged[f"{param}__age_min"] = (
        (merged["ts"] - merged[f"{param}__meas_ts"]).dt.total_seconds() / 60.0
    )
    return merged.set_index("ts")
