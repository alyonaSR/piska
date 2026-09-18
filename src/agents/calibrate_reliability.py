"""
Калибровка агента надёжности по истории.

Зона ответственности: Person 2 (Data Engineer).

ЗАЧЕМ: пороги тяжести режима нельзя брать с потолка. В исходном каркасе
стояло "0 при 345 C, 1 при 380 C", и в результате ОБЫЧНЫЙ режим получал
severity 0.564 и класс elevated, а более рискованный — 0.488 и normal.

РЕШЕНИЕ: severity считается как положение текущего режима в историческом
распределении (перцентильный ранг). Тогда "обычный режим = normal"
выполняется по построению, а не случайно.

Запуск разовый, результат лежит в config/reliability.yaml:
    python -m src.agents.calibrate_reliability
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd
import yaml

from ..data.loaders import load_telemetry
from ..data.tags import CONFIG_DIR

# Сетка перцентилей, по которой храним распределение каждого фактора.
GRID = np.arange(0, 101, 5)

# Окно для метрик нестабильности. Час = 6 отсчётов по 10 минут.
INSTAB_WINDOW = "1h"

# Окно для целевой температуры печи. Уставку DCS мы не видим, поэтому
# берём скользящую медиану за неделю как прокси "к чему стремится режим".
SETPOINT_WINDOW = "7D"


def running_thresholds(avt: pd.DataFrame, ht: pd.DataFrame) -> Dict[str, float]:
    """
    Пороги "установка работает" для обеих установок.

    И АВТ, и гидроочистка стоят около 4% времени, и в эти периоды значения
    падают на порядки: T55 до 2.7 C, 242000:T5 до -1.2 C, F26 до нуля.
    Если не отделить их от рабочего режима, происходит две беды:
      1) квантили считаются по распределению с 4% мусора;
      2) остановленная установка получает severity ~0.02 и класс normal.

    Пороги берём как долю от медианы: заведомо ниже рабочего режима
    и заведомо выше холодного железа.
    """
    return {
        "running_min_t55": round(0.9 * float(avt["AVT:T55"].median()), 1),
        "running_min_t5": round(0.5 * float(ht["242000:T5"].median()), 1),
        "running_min_f9": round(0.2 * float(ht["242000:F9"].median()), 1),
    }


def running_mask(avt: pd.DataFrame, ht: pd.DataFrame, thr: Dict[str, float]) -> pd.Series:
    """Обе установки в работе."""
    return (
        (avt["AVT:T55"] >= thr["running_min_t55"])
        & (ht["242000:T5"] >= thr["running_min_t5"])
        & (ht["242000:F9"] >= thr["running_min_f9"])
    )


def running_min_t55(t55: pd.Series) -> float:
    """
    Порог "печь работает". Рабочий диапазон T55 узкий (p5=375.8, p50=381.5),
    а при останове значения падают до единиц градусов. Берём 90% от медианы:
    заведомо ниже рабочего режима и заведомо выше холодной печи.
    """
    return 0.9 * float(t55.median())


def build_factors(avt: pd.DataFrame, ht: pd.DataFrame, thr: Dict[str, float]) -> pd.DataFrame:
    """
    Пять прокси-факторов тяжести режима. Все считаются только по прошлому:
    rolling по времени не заглядывает вперёд.

    Периоды останова маскируются: распределение должно описывать рабочий
    режим, а не смесь работы и холодного железа.
    """
    f = pd.DataFrame(index=avt.index)
    ok = running_mask(avt, ht, thr)

    # 1. Уровень температуры реактора: выше — жёстче режим, быстрее коксование
    f["reactor_temp"] = ht["242000:T5"].where(ok)

    # 2. Нагрузка: выше расход сырья — выше объёмная скорость, меньше
    #    время пребывания в реакторе.
    #    ВНИМАНИЕ: раньше здесь стоял 242000:F26. По ИСПРАВЛЕННОМУ справочнику
    #    F26 — расход гидроочищенного ДТ в цех №8, то есть ПРОДУКТ, а сырьё
    #    это F9 (массовый). Численно разницы почти нет: F9 и F26 коррелируют
    #    на 1.000 с отношением 0.85 т/м3, но называть вещи надо правильно.
    f["load"] = ht["242000:F9"].where(ok)

    # 3. Тепловое напряжение печи АВТ: превышение над скользящей целевой.
    #    Уставка DCS недоступна, прокси — медиана за 7 суток.
    #
    #    ВАЖНО: остановы маскируются. 4.2% времени T55 ниже 350 C (до 2.7 C),
    #    это холодная печь. Если их не убрать, медиана за неделю проседает,
    #    и после пуска метрика показывает "+142 C к целевой" — бессмыслица.
    t55 = avt["AVT:T55"]
    running = t55.where(ok)
    f["thermal_stress"] = running - running.rolling(SETPOINT_WINDOW, min_periods=6).median()

    # 4-5. Гидродинамическая и тепловая нестабильность: разброс за час.
    #      Высокий разброс = режим ходит, оборудование работает на усталость.
    f["pressure_instability"] = (
        avt["AVT:P67"].rolling(INSTAB_WINDOW, min_periods=3).std().where(ok)
    )
    f["temp_instability"] = (
        ht["242000:T5"].rolling(INSTAB_WINDOW, min_periods=3).std().where(ok)
    )

    return f


def calibrate(save: bool = True) -> Dict:
    avt = load_telemetry("AVT")
    ht = load_telemetry("242000")
    thr = running_thresholds(avt, ht)
    f = build_factors(avt, ht, thr)

    quantiles = {}
    for col in f.columns:
        s = f[col].dropna()
        quantiles[col] = [round(float(v), 6) for v in np.percentile(s, GRID)]

    cfg = {
        "_comment": (
            "Сгенерировано src/agents/calibrate_reliability.py по всей истории "
            "телеметрии. Severity = перцентильный ранг текущего режима, поэтому "
            "медианный режим даёт ~0.5 и класс normal по построению."
        ),
        "grid": [int(g) for g in GRID],
        "instab_window": INSTAB_WINDOW,
        "setpoint_window": SETPOINT_WINDOW,
        **thr,
        "quantiles": quantiles,
        # Веса свёртки. Температура реактора весит больше всех: именно она
        # определяет скорость коксообразования.
        "weights": {
            "reactor_temp": 0.35,
            "load": 0.20,
            "thermal_stress": 0.20,
            "pressure_instability": 0.125,
            "temp_instability": 0.125,
        },
        # Границы классов заполняются ниже по распределению самого severity.
        "class_thresholds": {},
    }

    # Второй проход: severity — это среднее пяти рангов, а среднее тянет
    # к центру. Если брать пороги 0.75/0.90 "на глаз", класс high не
    # сработает никогда. Поэтому калибруем пороги по распределению самого
    # severity: elevated = верхняя четверть истории, high = верхние 5%.
    ranks = pd.DataFrame(index=f.index)
    grid01 = GRID / 100.0
    for col in f.columns:
        ranks[col] = np.interp(f[col], quantiles[col], grid01)

    w = cfg["weights"]
    sev = sum(ranks[c] * w[c] for c in f.columns) / sum(w.values())
    sev = sev.dropna()

    cfg["class_thresholds"] = {
        "elevated": round(float(sev.quantile(0.75)), 3),
        "high": round(float(sev.quantile(0.95)), 3),
    }
    cfg["_severity_distribution"] = {
        "p10": round(float(sev.quantile(0.10)), 3),
        "p50": round(float(sev.quantile(0.50)), 3),
        "p90": round(float(sev.quantile(0.90)), 3),
        "p99": round(float(sev.quantile(0.99)), 3),
    }

    if save:
        path = os.path.join(CONFIG_DIR, "reliability.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
        print(f"записано: {path}")

    return cfg


if __name__ == "__main__":
    cfg = calibrate()
    for k, v in cfg["quantiles"].items():
        print(f"{k:>22}: p5 {v[1]:>9.3f}  p50 {v[10]:>9.3f}  p95 {v[19]:>9.3f}")