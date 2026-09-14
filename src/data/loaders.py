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



# --------------------------------------------------------------------------
# Справочники ЛИМС. Person 2.
# --------------------------------------------------------------------------

# Товарный продукт. Две точки после слова "Гидроочистка" — это не опечатка,
# так в выгрузке. Именно здесь Mg.Sulfur, главный жёсткий показатель.
TARGET_POINT = "Установка 'Гидроочистка'.. Точка отбора '2'. Продукт 'Дизельное топливо'"

# Сырьё гидроочистки. Mass.Sulfur здесь — сера НА ВХОДЕ, сильнейший
# предиктор серы на выходе. Значений мало (132), но признак ценный.
FEED_POINT = "Установка 'Гидроочистка'. Точка отбора '1'. Продукт 'ФРАКЦ_ДИЗ'."

# raw_param -> (канонический ключ, единицы, множитель конверсии)
#
# ЕДИНИЦЫ БЕРЁМ ОТСЮДА, А НЕ ИЗ ФАЙЛА: строка 2 выгрузки съехала.
# Проверено: 50%.T подписан как кг/м3, EBP.T как "% об.", D15 как °С.
# Канонические ключи обязаны совпадать с QualityAssess.predictions.
LIMS_PARAM_MAP = {
    "Mg.Sulfur":            ("sulfur_mgkg", "mg/kg", 1.0),
    "Mass.Sulfur":          ("sulfur_mgkg", "mg/kg", 10_000.0),   # % масс -> мг/кг
    "D15":                  ("d15_kgm3",    "kg/m3", 1.0),
    "FlashPoint":           ("flash_c",     "degC",  1.0),
    "CFPP":                 ("cfpp_c",      "degC",  1.0),
    "FilterabilityLimit.T": ("cfpp_c",      "degC",  1.0),
    "CloudPoint":           ("cloud_c",     "degC",  1.0),
    "CloudPoint_1":         ("cloud_c",     "degC",  1.0),
    "PourPoint":            ("pour_c",      "degC",  1.0),
    "IBP.T":                ("ibp_c",       "degC",  1.0),
    "50%.T":                ("t50_c",       "degC",  1.0),
    "90%.T":                ("t90_c",       "degC",  1.0),
    "95%.T":                ("t95_c",       "degC",  1.0),
    "EBP.T":                ("ebp_c",       "degC",  1.0),
    "I250":                 ("i250_pct",    "%vol",  1.0),
    "I350":                 ("i350_pct",    "%vol",  1.0),
    "CetaneNumber":         ("cetane",      "-",     1.0),
}

# Физически возможные значения. Выход за границы НЕ удаляется,
# а помечается флагом outlier: след сохраняется, фильтрует потребитель.
PLAUSIBLE = {
    "sulfur_mgkg": (0.0, 5000.0),
    "d15_kgm3":    (700.0, 950.0),
    "flash_c":     (20.0, 120.0),
    "cfpp_c":      (-60.0, 20.0),
    "cloud_c":     (-60.0, 20.0),
    "pour_c":      (-70.0, 20.0),
    "ibp_c":       (50.0, 300.0),
    "t50_c":       (150.0, 400.0),
    "t90_c":       (200.0, 450.0),
    "t95_c":       (200.0, 450.0),
    "ebp_c":       (200.0, 500.0),
    "i250_pct":    (0.0, 100.0),
    "i350_pct":    (0.0, 100.0),
    "cetane":      (20.0, 80.0),
}


# raw-имя колонки ПАК -> (канонический ключ, единицы, множитель)
# ppm по массе тождественно равен мг/кг: множитель 1.0, но конверсия
# зафиксирована явно, ТЗ требует воспроизводимости пересчётов.
PAK_PARAM_MAP = {
    "24-2000:Mg.Sulfur": ("sulfur_mgkg", "mg/kg", 1.0),
    "24-2000:D15":       ("d15_kgm3",    "kg/m3", 1.0),
}

PAK_COLUMNS = ["ts", "param", "raw_param", "value", "units", "healthy", "outlier"]

PAK_STUCK_MIN_POINTS = 6      # 6 точек по 10 минут = час без изменения значения

LIMS_COLUMNS = ["ts", "sample_point", "param", "raw_param", "value", "units", "outlier"]


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

    excluded = [c for c in df.columns if f"{unit}:{c}" in (cfg.get("exclude_tags") or {})]
    df = df.drop(columns=excluded)
    df.columns = [f"{unit}:{c}" for c in df.columns]
    return df


def _clean_sentinels(df: pd.DataFrame, unit: str, cfg: dict) -> pd.DataFrame:
    """
    Маркеры шкалы и физически невозможные значения.

    ПРАВИЛО ПО ОТРИЦАТЕЛЬНЫМ (проверено на avt_tags.csv):
      прежнее "всё отрицательное -> NaN" калечило данные. У AVT:P52
      65% значений это -0.00, у AVT:F69 отрицательные не глубже -0.45
      при p99 = 307. Это дрожание нуля, а не аномалия.

      Поэтому масштабный порог: 2% от p99 колонки.
        |x| < порога  -> 0    (прибор дрожит около нуля, расхода нет)
        x < -порога   -> NaN  (только для расходов: F/W/Q; для давлений
                               и перепадов крупный минус может быть физикой)

    Маркер шкалы считаем маркером, только если он реально является
    максимумом колонки.
    """
    out = df.copy()
    sentinels = cfg.get("sentinel_values", [])
    rule = cfg.get("negative_rule", {}) or {}
    share = float(rule.get("noise_share_of_p99", 0.02))
    nan_prefixes = tuple(rule.get("nan_below_threshold_prefixes", ["F", "W", "Q"]))
    zero_prefixes = tuple(rule.get("zero_clip_prefixes", ["F", "W", "Q", "P", "L", "D"]))

    for col in out.columns:
        s = out[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue

        for sv in sentinels:
            if np.isclose(s.max(skipna=True), sv, atol=1e-6):
                s = s.mask(np.isclose(s, sv, atol=1e-6))

        p99 = s.quantile(0.99)
        if pd.notna(p99) and p99 > 0 and col.startswith(zero_prefixes):
            # температуры сюда не попадают: -5 degC это физика, а не дрожание нуля
            thr = share * float(p99)
            s = s.mask(s.between(-thr, 0, inclusive="left"), 0.0)
            if col.startswith(nan_prefixes):
                s = s.mask(s < -thr)

        out[col] = s
    return out


def load_lims(path: Optional[str] = None, return_report: bool = False):
    """
    Разбор ЛИМС в длинный формат:
        ts | sample_point | param | raw_param | value | units | outlier

    Структура файла (проверена на выгрузке 1522 x 108):
      строка 0 — точка отбора, заполнена только в первой колонке блока
      строка 1 — код показателя, только в чётной колонке
      строка 2 — единицы, СЪЕХАВШИЕ, не используются
      строка 3 — в нечётной колонке объявленное количество значений
      строка 4+ — данные, колонки парами (дата, значение)

    return_report=True дополнительно отдаёт сверку declared vs parsed.
    """
    path = path or _lims_path()
    raw = pd.read_excel(path, header=None)

    n_blocks = int(raw.iloc[0].notna().sum())
    if n_blocks != 6:
        raise ValueError(
            f"Ожидалось 6 блоков точек отбора, найдено {n_blocks}. "
            "Формат выгрузки изменился, ffill по строке 0 больше не безопасен."
        )

    points = raw.iloc[0].ffill()
    frames, report = [], []

    for c in range(0, raw.shape[1], 2):
        raw_param = raw.iloc[1, c]
        if pd.isna(raw_param):
            continue
        raw_param = str(raw_param).strip()
        point = str(points[c]).strip()
        declared = pd.to_numeric(raw.iloc[3, c + 1], errors="coerce")

        block = raw.iloc[4:, [c, c + 1]].copy()
        block.columns = ["ts", "value"]
        block["ts"] = pd.to_datetime(block["ts"], errors="coerce")
        # 'Pt Created' и прочий текст в ячейке значения -> NaN
        block["value"] = pd.to_numeric(block["value"], errors="coerce")
        block = block.dropna(subset=["ts", "value"])

        parsed = len(block)

        # дубликаты по метке времени: пересдача отменяет первый результат
        dupes = int(block["ts"].duplicated().sum())
        block = block.drop_duplicates(subset="ts", keep="last")

        mapped = LIMS_PARAM_MAP.get(raw_param)
        if mapped is None:
            raise KeyError(
                f"Показатель '{raw_param}' отсутствует в LIMS_PARAM_MAP. "
                "Добавить явно, молча пропускать нельзя."
            )
        param, units, factor = mapped
        block["value"] = block["value"] * factor

        lo, hi = PLAUSIBLE.get(param, (float("-inf"), float("inf")))
        block["outlier"] = ~block["value"].between(lo, hi)

        block["sample_point"] = point
        block["param"] = param
        block["raw_param"] = raw_param
        block["units"] = units
        frames.append(block[LIMS_COLUMNS])

        report.append({
            "sample_point": point,
            "raw_param": raw_param,
            "param": param,
            "declared": declared,
            "parsed": parsed,
            "dupes": dupes,
            "outliers": int(block["outlier"].sum()),
        })

    out = pd.concat(frames, ignore_index=True)

    # после конкатенации ключ (точка, показатель, время) обязан быть уникальным,
    # иначе merge_asof возьмёт произвольную строку и результат перестанет быть
    # воспроизводимым. CloudPoint и CloudPoint_1 маплю в один cloud_c, поэтому
    # пересечения возможны.
    out = (
        out.sort_values(["sample_point", "param", "ts"])
        .drop_duplicates(subset=["sample_point", "param", "ts"], keep="last")
        .reset_index(drop=True)
    )

    if return_report:
        return out, pd.DataFrame(report)
    return out


def _find_data_file(*patterns: str) -> str:
    """
    Поиск файла в data/ по маске, без привязки к точному имени.

    Организаторы раздали всем одинаковые файлы, но имена у них длинные
    и с пробелами. Переименовывать нельзя: у четверых получится
    четыре разных имени и четыре разных бага. Поэтому ищем по маске.
    """
    import glob

    seen = []
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(DATA_DIR, pat)))
        hits = [h for h in hits if not os.path.basename(h).startswith("~$")]
        seen += hits
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"Не найден файл по маскам {patterns} в {DATA_DIR}. "
        f"Что сейчас лежит в папке: {sorted(os.listdir(DATA_DIR)) if os.path.isdir(DATA_DIR) else 'папки нет'}"
    )


def _lims_path() -> str:
    return _find_data_file("*ЛИМС*.xlsx", "*lims*.xlsx", "*LIMS*.xlsx")


def _pak_path() -> str:
    return _find_data_file("*ПАК*.xlsx", "*pak*.xlsx", "*PAK*.xlsx")


def load_pak(path: Optional[str] = None, return_report: bool = False):
    """
    Поточные анализаторы в длинный формат:
        ts | param | raw_param | value | units | healthy | outlier

    Структура файла:
      строка 0 — имя сигнала, только в колонке с датами
      строка 1 — единицы
      строка 2+ — данные, колонки парами (дата, значение)

    ВАЖНО: между парами есть пустая колонка-разделитель, поэтому пары
    ищутся по непустым заголовкам, а не шагом 2 как в ЛИМС.

    Временная сетка НЕ выравнивается: загрузчик отдаёт факты, решение
    о заполнении пропусков принимает слой признаков.
    """
    path = path or _pak_path()
    raw = pd.read_excel(path, header=None)

    header = raw.iloc[0]
    pair_cols = [c for c in range(raw.shape[1] - 1) if pd.notna(header[c])]
    if not pair_cols:
        raise ValueError(f"В {path} не найдено ни одного заголовка сигнала в строке 0")

    frames, report = [], []

    for c in pair_cols:
        raw_param = str(header[c]).strip()
        mapped = PAK_PARAM_MAP.get(raw_param)
        if mapped is None:
            raise KeyError(
                f"Сигнал ПАК '{raw_param}' отсутствует в PAK_PARAM_MAP. "
                "Добавить явно, молча пропускать нельзя."
            )
        param, units, factor = mapped

        block = raw.iloc[2:, [c, c + 1]].copy()
        block.columns = ["ts", "value"]
        block["ts"] = pd.to_datetime(block["ts"], errors="coerce")
        block["value"] = pd.to_numeric(block["value"], errors="coerce")
        block = block.dropna(subset=["ts", "value"])
        block = block.sort_values("ts").drop_duplicates(subset="ts", keep="last")
        block["value"] = block["value"] * factor

        # залипание считаем по упорядоченному во времени ряду, иначе diff врёт
        stuck = detect_stuck(block["value"], min_points=PAK_STUCK_MIN_POINTS)
        block["healthy"] = ~stuck.values

        lo, hi = PLAUSIBLE.get(param, (float("-inf"), float("inf")))
        block["outlier"] = ~block["value"].between(lo, hi)

        block["param"] = param
        block["raw_param"] = raw_param
        block["units"] = units
        frames.append(block[PAK_COLUMNS])

        report.append({
            "raw_param": raw_param,
            "param": param,
            "parsed": len(block),
            "first_ts": block["ts"].min(),
            "last_ts": block["ts"].max(),
            "stuck_points": int(stuck.sum()),
            "stuck_share": round(float(stuck.mean()), 4) if len(block) else 0.0,
            "outliers": int(block["outlier"].sum()),
        })

    out = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["param", "ts"])
        .reset_index(drop=True)
    )

    if return_report:
        return out, pd.DataFrame(report)
    return out


def stuck_episodes(pak: pd.DataFrame, param: str = "sulfur_mgkg") -> pd.DataFrame:
    """
    Границы эпизодов залипания: начало, конец, длительность.

    Отсюда Person 1 берёт конкретные ts для демо-сценария
    'неполные / аномальные данные'.
    """
    s = pak[pak["param"] == param].sort_values("ts").reset_index(drop=True)
    bad = ~s["healthy"]
    if not bad.any():
        return pd.DataFrame(columns=["start", "end", "points", "hours"])

    grp = (bad != bad.shift()).cumsum()
    rows = []
    for _, g in s[bad].groupby(grp[bad]):
        rows.append({
            "start": g["ts"].iloc[0],
            "end": g["ts"].iloc[-1],
            "points": len(g),
            "hours": round((g["ts"].iloc[-1] - g["ts"].iloc[0]).total_seconds() / 3600, 1),
        })
    return pd.DataFrame(rows).sort_values("points", ascending=False).reset_index(drop=True)


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