"""
Тесты split conformal + Adaptive Conformal Inference.

Зона ответственности: Person 3.
Проверяют математические свойства калибратора (покрытие, поправка на
конечную выборку, направление адаптации), не точность конкретной модели.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.models.conformal import ConformalResidualBounds, finite_sample_quantile


def test_finite_sample_quantile_uses_ceil_correction():
    """
    n=9, alpha=0.1 -> level = ceil(10*0.9)/9 = 9/9 = 1.0 -> максимум выборки.
    Без поправки (n+1) наивный уровень 0.9 дал бы другую (меньшую) точку.
    """
    scores = np.arange(1, 10, dtype=float)  # 1..9
    q = finite_sample_quantile(scores, alpha=0.1)
    assert q == 9.0


def test_finite_sample_quantile_rejects_empty():
    try:
        finite_sample_quantile(np.array([]), alpha=0.1)
        assert False, "должно упасть на пустой выборке"
    except ValueError:
        pass


def test_conformal_bounds_achieve_target_one_sided_coverage():
    """
    Каждая сторона калибруется НЕЗАВИСИМО как односторонняя (1-alpha)
    граница -- именно так их использует Gate (сера смотрит только hi,
    вспышка только lo), поэтому и проверяем одностороннее покрытие
    каждой границы по отдельности, а не совместное двустороннее (оно
    у независимых сторон при alpha=0.1 закономерно ~1-2*alpha=0.8,
    не 0.9 -- это не брак калибровки, а следствие независимой
    двусторонней комбинации двух односторонних 90%-границ).
    """
    rng = np.random.default_rng(42)
    calib = rng.normal(0, 1.0, size=2000)
    cb = ConformalResidualBounds(alpha=0.1).fit(calib)
    offset_lo, offset_hi = cb.bounds()

    test_sample = rng.normal(0, 1.0, size=5000)
    covered_hi = np.mean(test_sample <= offset_hi)
    covered_lo = np.mean(test_sample >= offset_lo)
    assert 0.85 <= covered_hi <= 0.95
    assert 0.85 <= covered_lo <= 0.95


def test_conformal_bounds_are_one_sided_independent():
    """
    Систематически смещённый остаток (модель занижает) должен дать
    ШИРОКУЮ верхнюю границу и УЗКУЮ нижнюю -- стороны считаются раздельно,
    не через симметричный |residual|.
    """
    rng = np.random.default_rng(0)
    calib = rng.normal(3.0, 1.0, size=1000)  # почти всегда положительный остаток
    cb = ConformalResidualBounds(alpha=0.1).fit(calib)
    offset_lo, offset_hi = cb.bounds()
    assert offset_hi > 3.0
    assert offset_lo > -1.0  # нижняя граница не должна быть неоправданно широкой


def test_aci_widens_after_repeated_misses():
    """Систематические пробои сверху должны расширять верхнюю границу."""
    rng = np.random.default_rng(1)
    calib = rng.normal(0, 1.0, size=500)
    cb = ConformalResidualBounds(alpha=0.1, gamma=0.05).fit(calib)
    _, hi_before = cb.bounds()

    for _ in range(30):
        cb.update(residual=10.0)  # намеренно всегда пробивает верхнюю границу

    _, hi_after = cb.bounds()
    assert hi_after > hi_before


def test_aci_narrows_after_stable_period():
    """Стабильный период без пробоев должен постепенно сужать границу."""
    rng = np.random.default_rng(2)
    calib = rng.normal(0, 1.0, size=500)
    cb = ConformalResidualBounds(alpha=0.1, gamma=0.05).fit(calib)
    _, hi_before = cb.bounds()

    for _ in range(30):
        cb.update(residual=0.0)  # всегда безопасно внутри границы

    _, hi_after = cb.bounds()
    assert hi_after < hi_before


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn(); print(f"  OK  {name}")
    print("\nвсе тесты conformal прошли")
