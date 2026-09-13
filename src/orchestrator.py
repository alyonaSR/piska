"""
L4. Оркестратор.

Зона ответственности: Person 1 (Lead / Архитектор).

Порядок одного цикла принятия решения ровно по разделу 1 ТЗ:
  1. получить состояние
  2. проверить полноту и актуальность данных
  3. оценить качество и риск выхода за спецификацию
  4. оценить тяжесть режима
  5. сформировать варианты
  6. отбросить недопустимые
  7. сравнить оставшиеся
  8. выбрать и объяснить, либо отказаться

ПРАВИЛО РАЗРЕШЕНИЯ КОНФЛИКТА ЦЕЛЕЙ — лексикографическое, а не взвешенное:
  1) жёсткие ограничения (бинарно, решает Gate)
  2) запас по качеству (больше = лучше)
  3) тяжесть режима (меньше = лучше)
  4) экономика (меньше cost_proxy = лучше)

Экономика стоит последней сознательно. Ключевой принцип ТЗ:
недопустимый режим нельзя компенсировать более высокой
производительностью или меньшими затратами.
"""

from __future__ import annotations

from typing import List, Optional

from .agents.optimizer import OptimizerAgent
from .agents.quality import QualityAgent
from .agents.reliability import ReliabilityAgent
from .contracts import (
    REFUSE,
    Candidate,
    DecisionTrace,
    GateVerdict,
    ProcessState,
    Recommendation,
)
from .data.tags import load_config, manipulated_vars, refusal_rules
from .explain import render_explanation, summarize_freshness
from .gate import ConstraintGate, filter_passed


class Orchestrator:
    def __init__(
        self,
        quality: Optional[QualityAgent] = None,
        reliability: Optional[ReliabilityAgent] = None,
        optimizer: Optional[OptimizerAgent] = None,
        gate: Optional[ConstraintGate] = None,
    ):
        self.quality = quality or QualityAgent()
        self.reliability = reliability or ReliabilityAgent()
        self.optimizer = optimizer or OptimizerAgent(self.quality)
        self.gate = gate or ConstraintGate()
        self.rules = refusal_rules()

    # ------------------------------------------------------------------
    def run_cycle(self, state: ProcessState) -> DecisionTrace:
        q = self.quality.assess(state)
        r = self.reliability.assess(state)

        # шаг 2: если данных не хватает — отказываемся ДО оптимизации
        refusal = self._data_refusal(state, q)
        if refusal is not None:
            return DecisionTrace(state.ts, state, q, r, [], [], refusal)

        candidates = self.optimizer.propose(state, q, r)
        verdicts = [self.gate.check(c, state, r.allowed_ranges) for c in candidates]
        survivors = filter_passed(candidates, verdicts)

        if not survivors:
            rec = self._refuse(
                state, q,
                "ни один из рассмотренных вариантов не проходит жёсткие ограничения",
                verdicts,
            )
            return DecisionTrace(state.ts, state, q, r, candidates, verdicts, rec)

        best = self._rank(survivors, verdicts)[0]
        rec = self._build_recommendation(state, q, r, best, survivors, verdicts)
        return DecisionTrace(state.ts, state, q, r, candidates, verdicts, rec)

    # ------------------------------------------------------------------
    def _data_refusal(self, state, q) -> Optional[Recommendation]:
        """Умение корректно отказаться — часть качественного решения (ТЗ)."""
        reasons = []

        lims = state.lims.get("sulfur_mgkg")
        if lims is None or lims.age_min is None:
            reasons.append("нет лабораторного значения серы")
        elif lims.age_min > self.rules["max_lims_age_min"]:
            reasons.append(
                f"последний лабораторный анализ серы устарел: "
                f"{lims.age_min / 60:.0f} ч при пороге "
                f"{self.rules['max_lims_age_min'] / 60:.0f} ч"
            )

        pak = state.pak.get("sulfur_mgkg")
        if pak is not None and not pak.healthy:
            reasons.append("поточный анализатор серы признан неисправным (залипание)")

        if q.confidence < self.rules["min_confidence"]:
            reasons.append(
                f"доверие к прогнозу {q.confidence:.2f} ниже порога "
                f"{self.rules['min_confidence']}"
            )

        if not reasons:
            return None

        return self._refuse(state, q, "; ".join(reasons), [])

    # ------------------------------------------------------------------
    def _refuse(self, state, q, reason, verdicts) -> Recommendation:
        top_violations = []
        for v in verdicts[:5]:
            top_violations.extend(v.violated)
        return Recommendation(
            ts=state.ts,
            action=REFUSE,
            reason=reason,
            expected_effect={},
            checks_passed=[],
            confidence=q.confidence,
            explanation=(
                "Надёжной рекомендации нет. " + reason + "."
                + (" Примеры отбраковки: " + "; ".join(top_violations[:3])
                   if top_violations else "")
                + " Рискованное управляющее воздействие в этих условиях не выдаётся."
            ),
            alternatives=[],
            data_freshness=summarize_freshness(state),
        )

    # ------------------------------------------------------------------
    def _rank(self, survivors: List[Candidate], verdicts: List[GateVerdict]) -> List[Candidate]:
        """
        Лексикографический выбор. Порядок критериев — это и есть политика системы.

        1) достаточен ли запас по жёсткому ограничению (бинарно)
        2) среди достаточных: МЕНЬШЕ ВОЗДЕЙСТВИЕ.
           Если режим уже безопасен, трогать установку не надо.
           Без этого правила система всегда уходит в максимальную severity
           ради лишнего запаса и нарушает требование ТЗ о том,
           что в устойчивом режиме лишних действий быть не должно.
        3) мягче режим (меньше severity_delta)
        4) дешевле (cost_proxy)

        Экономика последняя сознательно: недопустимый или более жёсткий режим
        нельзя оправдать выгодой.
        """
        vmap = {v.candidate_id: v for v in verdicts}
        target = load_config("constraints")["decision"]["target_margin"]["sulfur_mgkg"]
        mvars = manipulated_vars()

        def effort(c: Candidate) -> float:
            """Нормированная величина воздействия, 0 = ничего не трогаем."""
            return round(sum(
                abs(d) / float(mvars[t]["max_step"]) for t, d in c.deltas.items()
            ), 4)

        def key(c: Candidate):
            margin = vmap[c.candidate_id].margins.get("sulfur_mgkg", 0.0)
            sufficient = margin >= target
            if sufficient:
                return (0, effort(c), c.severity_delta, c.cost_proxy)
            # запаса не хватает — берём вариант с максимальным запасом
            return (1, -round(margin, 3), c.severity_delta, c.cost_proxy)

        return sorted(survivors, key=key)

    # ------------------------------------------------------------------
    def _build_recommendation(self, state, q, r, best, survivors, verdicts) -> Recommendation:
        vmap = {v.candidate_id: v for v in verdicts}
        v = vmap[best.candidate_id]
        s = best.predicted["sulfur_mgkg"]

        if best.is_no_action:
            reason = "режим устойчив, запас по всем жёстким ограничениям сохраняется"
        else:
            reason = (
                f"текущий риск нарушения спецификации {q.spec_risk_prob:.0%}, "
                f"тяжесть режима {r.severity_class}"
            )

        alts = []
        for c in self._rank(survivors, verdicts)[1:4]:
            alts.append({
                "candidate_id": c.candidate_id,
                "deltas": c.deltas,
                "sulfur_hi": round(c.predicted["sulfur_mgkg"].hi, 2),
                "cost_proxy": round(c.cost_proxy, 2),
                "severity_delta": c.severity_delta,
            })

        rec = Recommendation(
            ts=state.ts,
            action=best.deltas,
            reason=reason,
            expected_effect={
                "sulfur_mgkg": {"mean": round(s.mean, 2), "hi": round(s.hi, 2)},
                "margin_to_spec": v.margins.get("sulfur_mgkg"),
                "yield_delta_tph": round(best.yield_delta, 2),
                "cost_proxy": round(best.cost_proxy, 2),
                "severity_delta": best.severity_delta,
            },
            checks_passed=v.checked,
            confidence=q.confidence,
            alternatives=alts,
            data_freshness=summarize_freshness(state),
        )
        rec.explanation = render_explanation(rec, q, r, best, v, len(survivors))
        return rec
