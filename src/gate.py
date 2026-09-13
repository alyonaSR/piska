"""
L3. Жёсткий фильтр.

Зона ответственности: Person 1 (Lead / Архитектор).

ЭТОТ МОДУЛЬ НЕ ИМЕЕТ ПРАВА:
  - импортировать что-либо из agents/
  - вызывать LLM
  - принимать текстовые обоснования

Он только сравнивает числа с config/constraints.yaml.
Это архитектурное свойство: безопасность системы не зависит от того,
что именно сгенерировала языковая модель. Ни один вариант не может
пройти сюда за счёт убедительного объяснения.

Соответствует принципу из литературы по безопасным операторским
агентам: конфликт ограничений разрешается детерминированно,
независимо от вывода LLM.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .contracts import Candidate, GateVerdict, Interval, ProcessState
from .data.tags import manipulated_vars, quality_specs, load_config


class ConstraintGate:
    def __init__(self):
        self.specs = quality_specs()
        self.mvars = manipulated_vars()
        self.blending = load_config("constraints")["blending"]

    # ------------------------------------------------------------------
    def check(
        self,
        candidate: Candidate,
        state: ProcessState,
        allowed_ranges: Optional[Dict[str, List[float]]] = None,
        blend_fractions: Optional[Sequence[float]] = None,
    ) -> GateVerdict:
        violated: List[str] = []
        margins: Dict[str, float] = {}
        checked: List[str] = []

        self._check_quality(candidate, violated, margins, checked)
        self._check_ranges(candidate, state, allowed_ranges, violated, margins, checked)
        self._check_steps(candidate, violated, margins, checked)
        if blend_fractions is not None:
            self._check_blending(blend_fractions, violated, margins, checked)

        return GateVerdict(
            candidate_id=candidate.candidate_id,
            passed=len(violated) == 0,
            violated=violated,
            margins=margins,
            checked=checked,
        )

    # ------------------------------------------------------------------
    def _check_quality(self, cand, violated, margins, checked):
        """
        Спецификация проверяется по КОНСЕРВАТИВНОЙ границе интервала.

        Для серы: hi, а не mean. Прогноз mean=9.1 при лимите 10 выглядит
        безопасно, но hi=10.4 означает реальный риск нарушения.
        Проверка по mean — это ошибка, которая стоит всей задачи.
        """
        for param, spec in self.specs.items():
            iv: Interval = cand.predicted.get(param)
            if iv is None:
                violated.append(f"{param}: прогноз отсутствует, проверка невозможна")
                continue

            limit = float(spec["limit"])
            value = iv.hi if spec["check_on"] == "hi" else iv.lo
            checked.append(
                f"{param} {spec['direction']} {limit} по {spec['check_on']} [{spec['source']}]"
            )

            if spec["direction"] == "max":
                margin = limit - value
            else:
                margin = value - limit

            margins[param] = round(margin, 3)
            if margin < 0:
                violated.append(
                    f"{param}: {value:.2f} против лимита {limit:.2f} "
                    f"(проверка по {spec['check_on']}), нарушение {abs(margin):.2f}"
                )

    # ------------------------------------------------------------------
    def _check_ranges(self, cand, state, allowed_ranges, violated, margins, checked):
        """Абсолютное значение после изменения должно остаться в диапазоне."""
        allowed_ranges = allowed_ranges or {}
        for tag, delta in cand.deltas.items():
            cur = state.tag(tag)
            if cur is None:
                violated.append(f"{tag}: текущее значение неизвестно")
                continue
            new = cur + delta

            lo, hi = self.mvars[tag]["range"]
            src = self.mvars[tag]["source"]
            if tag in allowed_ranges:
                alo, ahi = allowed_ranges[tag]
                lo, hi = max(lo, alo), min(hi, ahi)

            checked.append(f"{tag} в [{lo}, {hi}] [{src}]")
            margin = min(new - lo, hi - new)
            margins[f"{tag}__range"] = round(margin, 3)
            if margin < 0:
                violated.append(
                    f"{tag}: {new:.2f} вне допустимого диапазона [{lo}, {hi}]"
                )

    # ------------------------------------------------------------------
    def _check_steps(self, cand, violated, margins, checked):
        """Размер шага = trust region. Защита от экстраполяции модели."""
        for tag, delta in cand.deltas.items():
            max_step = float(self.mvars[tag]["max_step"])
            checked.append(f"|delta {tag}| <= {max_step}")
            margin = max_step - abs(delta)
            margins[f"{tag}__step"] = round(margin, 3)
            if margin < 0:
                violated.append(
                    f"{tag}: шаг {delta:+.2f} больше допустимого {max_step}"
                )

    # ------------------------------------------------------------------
    def _check_blending(self, fractions, violated, margins, checked):
        """Доли компонентов блендинга обязаны давать 100 процентов."""
        total = float(sum(fractions))
        target = float(self.blending["components_sum"])
        tol = float(self.blending["tolerance"])
        checked.append(f"сумма долей блендинга = {target}")
        margins["blend_sum"] = round(target - total, 9)
        if abs(total - target) > tol:
            violated.append(f"сумма долей блендинга {total:.6f} вместо {target}")
        if any(f < 0 for f in fractions):
            violated.append("отрицательная доля компонента блендинга")


def filter_passed(candidates, verdicts) -> List[Candidate]:
    ok = {v.candidate_id for v in verdicts if v.passed}
    return [c for c in candidates if c.candidate_id in ok]
