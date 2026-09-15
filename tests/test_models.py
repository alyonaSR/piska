"""
Тесты предсказателей качества.

Зона ответственности: Person 3.
Эти тесты проверяют не точность, а ЗНАКИ и интерфейс. Они должны
продолжать проходить после замены заглушек на обученные модели —
если перестали, модель выучила контур регулирования вместо физики.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.contracts import Interval
from src.models import AVTModel, GOModel
from src.models.base import monotone_vector


def test_avt_interface():
    m = AVTModel()
    out = m.predict({"AVT:F30": 128.4})
    assert set(out) == set(m.outputs)
    assert all(isinstance(v, Interval) for v in out.values())
    assert all(v.lo <= v.mean <= v.hi for v in out.values())


def test_avt_more_diesel_draw_means_heavier_tail():
    """Больше отбор дизельной фракции -> EBP растёт."""
    m = AVTModel()
    low = m.predict({"AVT:F30": 110.0})["feed_ebp_c"].mean
    high = m.predict({"AVT:F30": 150.0})["feed_ebp_c"].mean
    assert high > low


def test_go_hotter_reactor_means_less_sulfur():
    """Аррениус: горячее -> глубже обессеривание. Знак критичен."""
    m = GOModel()
    base = m.predict({"sulfur_anchor": 9.0, "d_go_temp_c": 0.0, "d_feed_tail_c": 0.0})
    hot = m.predict({"sulfur_anchor": 9.0, "d_go_temp_c": 2.0, "d_feed_tail_c": 0.0})
    assert hot["sulfur_mgkg"].mean < base["sulfur_mgkg"].mean


def test_go_heavier_feed_means_more_sulfur():
    """Тяжелее хвост -> труднее удаляемая сера."""
    m = GOModel()
    base = m.predict({"sulfur_anchor": 9.0, "d_go_temp_c": 0.0, "d_feed_tail_c": 0.0})
    heavy = m.predict({"sulfur_anchor": 9.0, "d_go_temp_c": 0.0, "d_feed_tail_c": 5.0})
    assert heavy["sulfur_mgkg"].mean > base["sulfur_mgkg"].mean


def test_bigger_step_widens_interval():
    """Модель уверена только рядом с режимами, которые видела."""
    m = GOModel()
    small = m.predict({"sulfur_anchor": 9.0, "d_go_temp_c": 0.5, "d_feed_tail_c": 0.0})
    big = m.predict({"sulfur_anchor": 9.0, "d_go_temp_c": 5.0, "d_feed_tail_c": 0.0})
    assert big["sulfur_mgkg"].width > small["sulfur_mgkg"].width


def test_missing_features_are_reported_not_raised():
    m = GOModel()
    assert "sulfur_anchor" in m.check_features({"d_go_temp_c": 1.0})


def test_monotone_vector_for_lightgbm():
    cols = ["go_reactor_temp_c", "feed_t95_c", "some_unknown_feature"]
    assert monotone_vector(cols, "sulfur_mgkg") == [-1, 1, 0]


def test_chain_avt_output_feeds_go_input():
    """Цепочка реальна: изменение режима АВТ меняет серу на выходе ГО."""
    avt, go = AVTModel(), GOModel()
    a = avt.predict({"AVT:F30": 128.4})
    b = avt.predict({"AVT:F30": 150.0})
    d_t95 = b["feed_ebp_c"].mean - a["feed_ebp_c"].mean
    assert d_t95 > 0
    s = go.predict({"sulfur_anchor": 8.5, "d_go_temp_c": 0.0, "d_feed_tail_c": d_t95})
    assert s["sulfur_mgkg"].mean > 8.5


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn(); print(f"  OK  {name}")
    print("\nвсе тесты моделей прошли")
