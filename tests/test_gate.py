"""
Тесты жёсткого фильтра.

Зона ответственности: Person 1.
Gate — единственный модуль, где баг напрямую означает опасную рекомендацию.
Тесты на него пишутся ПЕРВЫМИ.

Запуск: python -m pytest tests -q   (или python tests/test_gate.py без pytest)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

from src.contracts import Candidate, Interval, ProcessState
from src.gate import ConstraintGate


def _state():
    return ProcessState(
        ts=datetime(2026, 3, 14, 8, 20),
        tags={"242000:T5": 365.0, "AVT:F30": 61.0, "AVT:T33": 348.0},
    )


def _cand(sulfur_mean, sulfur_half=0.5, deltas=None):
    return Candidate(
        candidate_id="c_test",
        deltas=deltas if deltas is not None else {"242000:T5": 0.0},
        predicted={
            "sulfur_mgkg": Interval(sulfur_mean, sulfur_mean - sulfur_half, sulfur_mean + sulfur_half),
            "flash_c": Interval(68, 65, 71),
            "cfpp_c": Interval(-6, -8, -4),
            "d15_kgm3": Interval(836, 833, 839),
        },
    )


def test_sulfur_checked_on_upper_bound_not_mean():
    """Ключевой тест: mean=9.6 проходит по среднему, но hi=10.4 обязан упасть."""
    g = ConstraintGate()
    v = g.check(_cand(9.6, 0.8), _state())
    assert not v.passed
    assert any("sulfur" in x for x in v.violated)


def test_sulfur_with_margin_passes():
    g = ConstraintGate()
    v = g.check(_cand(8.0, 0.5), _state())
    assert v.passed
    assert v.margins["sulfur_mgkg"] > 0


def test_step_limit_enforced():
    g = ConstraintGate()
    v = g.check(_cand(8.0, 0.5, deltas={"242000:T5": 5.0}), _state())
    assert not v.passed
    assert any("шаг" in x for x in v.violated)


def test_allowed_ranges_from_reliability_agent_narrow_the_gate():
    """Агент надёжности реально ограничивает оптимизатор."""
    g = ConstraintGate()
    c = _cand(8.0, 0.5, deltas={"242000:T5": 1.5})
    assert g.check(c, _state()).passed
    v = g.check(c, _state(), allowed_ranges={"242000:T5": [340.0, 366.0]})
    assert not v.passed


def test_blending_must_sum_to_one():
    g = ConstraintGate()
    ok = g.check(_cand(8.0), _state(), blend_fractions=[0.6, 0.4])
    bad = g.check(_cand(8.0), _state(), blend_fractions=[0.6, 0.5])
    assert ok.passed and not bad.passed


def test_missing_prediction_is_a_violation():
    g = ConstraintGate()
    c = _cand(8.0)
    del c.predicted["sulfur_mgkg"]
    assert not g.check(c, _state()).passed


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"  OK  {name}")
    print("\nвсе тесты прошли")
