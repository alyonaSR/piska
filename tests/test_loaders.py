"""
Тесты слоя данных и агента надёжности.

Зона ответственности: Person 2 (Data Engineer).

ГЛАВНЫЙ ТЕСТ — test_no_future_leak. Утечку будущего нельзя заметить глазами:
она проявляется не ошибкой, а подозрительно хорошей метрикой у модели.
Единственный способ доказать её отсутствие — проверка на множестве дат.

Запуск из корня репозитория:
    python -m pytest tests/test_loaders.py -v

Тесты, которым нужны файлы организаторов в data/, пропускаются, если
файлов нет: у Person 1 и Person 4 их может не быть.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.data.loaders import (
    FEED_POINT,
    LIMS_PARAM_MAP,
    PLAUSIBLE,
    TARGET_POINT,
    detect_stuck,
    load_lims,
    load_pak,
    load_telemetry,
)
from src.data.state_builder import build_state

# Даты, на которых гоняем проверку утечки. Сетка редкая, но покрывает
# все четыре года и обе установки в разных режимах.
LEAK_CHECK_DATES = pd.date_range("2023-03-01", "2026-07-01", freq="37D")


def _have_data() -> bool:
    try:
        load_lims()
        return True
    except (FileNotFoundError, ValueError):
        return False


needs_data = pytest.mark.skipif(
    not _have_data(), reason="нет файлов организаторов в data/"
)


# --------------------------------------------------------------------------
# Загрузчики, которым данные не нужны
# --------------------------------------------------------------------------
def test_detect_stuck_finds_flat_run():
    """
    Помечаются ПОВТОРЫ, а не вся серия: из восьми одинаковых значений
    мёртвыми считаются семь, первое — ещё нормальное измерение.
    """
    s = pd.Series([1.0] * 8 + [2.0, 3.0, 4.0])
    assert detect_stuck(s, min_points=6).sum() == 7


def test_detect_stuck_ignores_short_run():
    s = pd.Series([1.0, 1.0, 1.0, 2.0, 3.0])
    assert not detect_stuck(s, min_points=6).any()


def test_param_map_targets_are_canonical():
    """Ключи карты должны совпадать с тем, что ждёт QualityAssess."""
    required = {"sulfur_mgkg", "d15_kgm3", "flash_c", "cfpp_c", "t95_c"}
    produced = {v[0] for v in LIMS_PARAM_MAP.values()}
    assert required <= produced


def test_mass_sulfur_conversion_factor():
    """% масс -> мг/кг это ровно 10000, а не 100 и не 1e6."""
    assert LIMS_PARAM_MAP["Mass.Sulfur"] == ("sulfur_mgkg", "mg/kg", 10_000.0)
    assert LIMS_PARAM_MAP["Mg.Sulfur"][2] == 1.0


def test_plausible_covers_hard_constrained_params():
    for key in ("sulfur_mgkg", "d15_kgm3", "flash_c"):
        lo, hi = PLAUSIBLE[key]
        assert lo < hi


# --------------------------------------------------------------------------
# ЛИМС
# --------------------------------------------------------------------------
@needs_data
def test_lims_counts_match_declared():
    """
    В строке 4 выгрузки указано ожидаемое число значений по каждой колонке.
    Допуск 1: в восьми колонках вместо числа стоит текст 'Pt Created'.
    """
    _, report = load_lims(return_report=True)
    diff = (report["declared"] - report["parsed"]).abs()
    assert (diff <= 1).all(), report[diff > 1]


@needs_data
def test_lims_units_are_canonical():
    """Один param — одна единица измерения, без вариантов."""
    lims = load_lims()
    for param, grp in lims.groupby("param"):
        assert grp["units"].nunique() == 1, param
    assert set(lims.loc[lims["param"] == "sulfur_mgkg", "units"]) == {"mg/kg"}


@needs_data
def test_feed_sulfur_is_orders_above_product():
    """
    Сера в сырье ~9460 мг/кг, в продукте ~8.6. Если конверсия не сработала,
    они окажутся одного порядка — значит Mass.Sulfur осталась в процентах.
    """
    lims = load_lims()
    feed = lims[(lims.sample_point == FEED_POINT) & (lims.param == "sulfur_mgkg")]
    prod = lims[(lims.sample_point == TARGET_POINT) & (lims.param == "sulfur_mgkg")]
    assert feed["value"].median() > 100 * prod["value"].median()


@needs_data
def test_sample_point_is_part_of_key():
    """t50_c встречается в нескольких точках отбора и путать их нельзя."""
    lims = load_lims()
    assert lims[lims.param == "t50_c"]["sample_point"].nunique() > 1


@needs_data
def test_lims_key_is_unique():
    """
    Дубликаты ключа сделали бы merge_asof невоспроизводимым: он взял бы
    произвольную строку из совпадающих.
    """
    lims = load_lims()
    assert not lims.duplicated(subset=["sample_point", "param", "ts"]).any()


# --------------------------------------------------------------------------
# ПАК
# --------------------------------------------------------------------------
@needs_data
def test_pak_marks_stuck_analyzer():
    """Известный факт: анализатор серы залипает, самый длинный эпизод 6731 точка."""
    pak = load_pak()
    sulfur = pak[pak.param == "sulfur_mgkg"]
    assert (~sulfur["healthy"]).any()


@needs_data
def test_pak_key_is_unique():
    pak = load_pak()
    assert not pak.duplicated(subset=["param", "ts"]).any()


# --------------------------------------------------------------------------
# Телеметрия
# --------------------------------------------------------------------------
@needs_data
def test_telemetry_tags_have_unit_prefix():
    """Код T6 на АВТ и на 24-2000 — разные величины, префикс обязателен."""
    tel = load_telemetry("AVT")
    assert all(c.startswith("AVT:") for c in tel.columns)


@needs_data
def test_dead_tags_excluded():
    """AVT:D10 и AVT:F5 исключены осознанно, причины в config/tags.yaml."""
    tel = load_telemetry("AVT")
    assert "AVT:D10" not in tel.columns
    assert "AVT:F5" not in tel.columns


@needs_data
def test_temperatures_not_zero_clipped():
    """
    Правило обнуления отрицательных не должно трогать температуры:
    -5 degC это физика, а не дрожание нуля.
    """
    tel = load_telemetry("AVT")
    t = tel["AVT:T33"].dropna()
    assert t.between(1.0, 500.0).mean() > 0.9


# --------------------------------------------------------------------------
# ГЛАВНОЕ: утечка будущего
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sources():
    tel = pd.concat([load_telemetry("AVT"), load_telemetry("242000")], axis=1)
    return tel, load_lims(), load_pak()


@needs_data
def test_no_future_leak(sources):
    """
    Ни одно измерение в ProcessState не может быть позже момента сборки.

    Это главный тест проекта: если он падает, любая метрика модели
    завышена, а рекомендации построены на том, чего оператор не знал.
    """
    tel, lims, pak = sources
    for ts in LEAK_CHECK_DATES:
        st = build_state(ts, tel, lims, pak)
        for store_name, store in (("lims", st.lims), ("pak", st.pak)):
            for key, m in store.items():
                assert m.ts is None or m.ts <= st.ts, f"{store_name}:{key} на {ts}"


@needs_data
def test_age_is_never_negative(sources):
    tel, lims, pak = sources
    for ts in LEAK_CHECK_DATES:
        st = build_state(ts, tel, lims, pak)
        for store in (st.lims, st.pak):
            for key, m in store.items():
                assert m.age_min is None or m.age_min >= 0, f"{key} на {ts}"


@needs_data
def test_state_is_deterministic(sources):
    """Одинаковый вход — одинаковый выход. Критерий воспроизводимости."""
    tel, lims, pak = sources
    ts = datetime(2024, 4, 23, 12, 0)
    a = build_state(ts, tel, lims, pak)
    b = build_state(ts, tel, lims, pak)
    assert a.tags == b.tags
    assert a.dq_flags == b.dq_flags
    assert {k: v.value for k, v in a.lims.items()} == {k: v.value for k, v in b.lims.items()}


@needs_data
def test_stuck_pak_yields_to_lims(sources):
    """
    23.04.2024: поточный анализатор залип с 16 марта и показывает ~7.7,
    лаборатория в тот день — 107 мг/кг. best_quality обязан выбрать ЛИМС.

    Это же реальный демо-сценарий 'аномальные данные'.
    """
    tel, lims, pak = sources
    st = build_state(datetime(2024, 4, 23, 12, 0), tel, lims, pak)
    assert st.pak["sulfur_mgkg"].healthy is False
    best = st.best_quality("sulfur_mgkg")
    assert best.source == "LIMS"
    assert best.value > 50
    assert any("залип" in f for f in st.dq_flags)


# --------------------------------------------------------------------------
# Агент надёжности
# --------------------------------------------------------------------------
@needs_data
def test_median_regime_is_normal(sources):
    """
    Проверка калибровки. В исходной версии обычный режим получал класс
    elevated, а более рискованный — normal. Медиана обязана быть normal.
    """
    from src.agents.reliability import ReliabilityAgent

    tel, lims, pak = sources
    ag = ReliabilityAgent(telemetry=tel)
    rs = np.random.RandomState(0)
    pool = pd.date_range("2023-03-01", "2026-07-01", freq="12h")
    sev = [
        ag.assess(build_state(pd.Timestamp(ts), tel, lims, pak)).severity_index
        for ts in rs.choice(pool, 60, replace=False)
    ]
    assert 0.35 < float(np.median(sev)) < 0.65


@needs_data
def test_shutdown_detected(sources):
    """15.04.2024 обе установки стояли: класс обязан быть shutdown."""
    from src.agents.reliability import ReliabilityAgent

    tel, lims, pak = sources
    r = ReliabilityAgent(telemetry=tel).assess(
        build_state(datetime(2024, 4, 15, 12, 0), tel, lims, pak)
    )
    assert r.severity_class == "shutdown"
    assert any("остановлена" in f for f in r.factors)


@needs_data
def test_allowed_ranges_narrow_on_high_severity(sources):
    """
    allowed_ranges — единственное место, где агент реально ограничивает
    оптимизатор. При тяжёлом режиме верх диапазона температур обязан
    отрезаться.
    """
    from src.agents.reliability import ReliabilityAgent
    from src.data.tags import manipulated_vars

    tel, lims, pak = sources
    ag = ReliabilityAgent(telemetry=tel)
    base = {t: s["range"] for t, s in manipulated_vars().items()}

    narrowed = False
    rs = np.random.RandomState(1)
    pool = pd.date_range("2023-03-01", "2026-07-01", freq="6h")
    for ts in rs.choice(pool, 200, replace=False):
        r = ag.assess(build_state(pd.Timestamp(ts), tel, lims, pak))
        if r.severity_class in ("elevated", "high"):
            for tag, rng in r.allowed_ranges.items():
                if rng[1] < base[tag][1]:
                    narrowed = True
    assert narrowed, "ни разу не сработало сужение диапазонов"


def test_reliability_works_without_telemetry():
    """
    Person 1 создаёт агента без телеметрии. Он обязан работать,
    честно сообщая, какие факторы не рассчитаны.
    """
    from src.agents.reliability import ReliabilityAgent
    from src.data.state_builder import build_demo_state

    r = ReliabilityAgent().assess(build_demo_state("normal"))
    assert 0.0 <= r.severity_index <= 1.0
    assert any("телеметрия не передана" in a for a in r.assumptions)
    assert r.allowed_ranges