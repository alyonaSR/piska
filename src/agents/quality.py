"""
L1a. Агент качества.

Зона ответственности: Person 1 (Lead / Архитектор).

ЭТО ОБОЛОЧКА, А НЕ МОДЕЛЬ. Сами предсказатели живут в src/models/
и их пишет Person 3. Граница такая:

    src/models/   признаки на входе -> Interval на выходе.
                  Ничего не знает про ProcessState и спецификации.

    QualityAgent  вызывает обе модели, сцепляет их в цепочку АВТ -> ГО,
                  считает confidence по возрасту данных, считает
                  spec_risk_prob по конфигу, собирает QualityAssess.

Такое разделение нужно, чтобы Person 1 и Person 3 не правили один файл.

ЦЕПОЧКА: AVTModel предсказывает качество дизельной фракции, уходящей
в гидроочистку. Её конец кипения (EBP) становится ВХОДОМ GOModel. Чем тяжелее хвост,
тем труднее удаляемая сера и тем более жёсткий режим нужен на ГО.
Это и есть связанность цепочки из ТЗ, выраженная в коде.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..contracts import Interval, ProcessState, QualityAssess
from ..data.tags import quality_specs, refusal_rules
from ..models import AVTModel, GOModel


class QualityAgent:
    """Заглушка с физически осмысленным поведением, чтобы цикл работал уже сегодня."""

    def __init__(self, avt_model=None, go_model=None):
        # Person 3 подменяет эти два объекта обученными.
        # Сигнатура predict(features) -> dict[str, Interval] не меняется,
        # поэтому подмена не затрагивает ни один другой слой.
        self.avt = avt_model or AVTModel()
        self.go = go_model or GOModel()
        self.model_id = f"{self.avt.model_id}+{self.go.model_id}"

    # ------------------------------------------------------------------
    def assess(
        self,
        state: ProcessState,
        deltas: Optional[Dict[str, float]] = None,
    ) -> QualityAssess:
        """
        deltas=None  -> оценка текущего состояния
        deltas={...} -> прогноз для гипотетического режима (вызывает оптимизатор)
        """
        deltas = deltas or {}
        preds = self._predict(state, deltas)
        conf = self._confidence(state)
        risk = self._spec_risk(preds)
        return QualityAssess(
            predictions=preds,
            spec_risk_prob=risk,
            confidence=conf,
            drivers=self._drivers(state, deltas),
            model_id=self.model_id,
        )

    # ------------------------------------------------------------------
    def _predict(self, state: ProcessState, deltas: Dict[str, float]) -> Dict[str, Interval]:
        """
        Цепочка АВТ -> ГО. Две модели, два вызова.

        Логика приращений: измеренная сера уже содержит в себе весь
        текущий режим. Поэтому AVTModel вызывается ДВАЖДЫ — на текущем
        режиме и на предлагаемом — и в GOModel уходит разница по T95.
        """
        # 1. текущий режим АВТ
        f_now = {t: state.tag(t) for t in self.avt.required_features}
        avt_now = self.avt.predict({k: v for k, v in f_now.items() if v is not None})

        # 2. режим АВТ после предлагаемого изменения
        f_new = {t: (v + deltas.get(t, 0.0)) for t, v in f_now.items() if v is not None}
        avt_new = self.avt.predict(f_new)

        # 3. якорь: фактически измеренная сера, приоритет ЛИМС -> ПАК
        base = state.best_quality("sulfur_mgkg")
        anchor = base.value if base and base.value is not None else 8.5

        # 4. гидроочистка получает выход АВТ как вход
        go_features = {
            "sulfur_anchor": anchor,
            "d_go_temp_c": deltas.get("242000:T5", 0.0),
            "d_feed_tail_c": avt_new["feed_ebp_c"].mean - avt_now["feed_ebp_c"].mean,
            "feed_flash_c": avt_new["feed_flash_c"].mean,
            "feed_cfpp_c": avt_new["feed_cfpp_c"].mean,
            "feed_d15_kgm3": avt_new["feed_d15_kgm3"].mean,
        }
        return self.go.predict(go_features)

    # ------------------------------------------------------------------
    def _confidence(self, state: ProcessState) -> float:
        """
        Доверие к прогнозу падает от старых и мёртвых данных.
        Это отдельная величина от spec_risk_prob.
        """
        rules = refusal_rules()
        conf = 0.9

        lims = state.lims.get("sulfur_mgkg")
        if lims and lims.age_min is not None:
            over = lims.age_min / rules["max_lims_age_min"]
            if over > 1.0:
                conf -= min(0.45, 0.25 * over)

        pak = state.pak.get("sulfur_mgkg")
        if pak is not None and not pak.healthy:
            conf -= 0.30

        conf -= 0.05 * len(state.dq_flags)
        return max(0.0, round(conf, 3))

    # ------------------------------------------------------------------
    def _spec_risk(self, preds: Dict[str, Interval]) -> float:
        """
        Грубая вероятность нарушить хотя бы одно требование.
        TODO(Person 3): заменить на P(y > limit) из квантильной модели.
        """
        worst = 0.0
        for param, spec in quality_specs().items():
            iv = preds.get(param)
            if iv is None or iv.width <= 0:
                continue
            if spec["direction"] == "max":
                p = (iv.hi - spec["limit"]) / iv.width
            else:
                p = (spec["limit"] - iv.lo) / iv.width
            worst = max(worst, min(1.0, max(0.0, p)))
        return round(worst, 3)

    # ------------------------------------------------------------------
    def _drivers(self, state: ProcessState, deltas: Dict[str, float]) -> list:
        """TODO(Person 3): заменить на SHAP по обученной модели."""
        out = []
        # пороги = p75 и p25 по очищенной истории, см. config/constraints.yaml
        f30 = state.tag("AVT:F30")
        if f30 and f30 > 140.0:
            out.append(f"высокий отбор дизельной фракции AVT:F30={f30:.1f} т/ч, хвост тяжелее")
        f32 = state.tag("AVT:F32")
        if f32 and f32 > 90.0:
            out.append(f"высокий отбор AVT:F32={f32:.1f} т/ч")
        t5 = state.tag("242000:T5")
        if t5 and t5 < 367.0:
            out.append(f"температура реактора 242000:T5={t5:.1f} C ниже обычной, severity недостаточна")
        for k, v in deltas.items():
            out.append(f"проверяется изменение {k} на {v:+.1f}")
        return out
