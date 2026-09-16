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

import math
from typing import Dict, List, Mapping, Optional, Sequence, Union

from .contracts import Candidate, GateVerdict, Interval, ProcessState
from .data.tags import manipulated_vars, quality_specs, load_config


def _is_number(x) -> bool:
    """
    NaN и бесконечность — не числа, а следствие сбоя выше по цепочке.

    Критично именно здесь: любое сравнение с NaN даёт False, поэтому
    проверка "margin < 0" молча ПРОПУСКАЕТ кандидата с испорченным
    прогнозом. Жёсткий фильтр, который пропускает NaN, не фильтр.
    """
    return isinstance(x, (int, float)) and math.isfinite(x)


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
        blend_fractions: Optional[Union[Mapping[str, float], Sequence[float]]] = None,
    ) -> GateVerdict:
        violated: List[str] = []
        warnings: List[str] = []
        margins: Dict[str, float] = {}
        checked: List[str] = []

        self._check_quality(candidate, violated, warnings, margins, checked)
        self._check_ranges(candidate, state, allowed_ranges, violated, margins, checked)
        self._check_steps(candidate, violated, margins, checked)
        if blend_fractions is not None:
            self._check_blending(blend_fractions, violated, margins, checked)

        return GateVerdict(
            candidate_id=candidate.candidate_id,
            passed=len(violated) == 0,
            violated=violated,
            warnings=warnings,
            margins=margins,
            checked=checked,
        )

    # ------------------------------------------------------------------
    def _check_quality(self, cand, violated, warnings, margins, checked):
        """
        Спецификация проверяется по КОНСЕРВАТИВНОЙ границе интервала.

        Для серы: hi, а не mean. Прогноз mean=9.1 при лимите 10 выглядит
        безопасно, но hi=10.4 означает реальный риск нарушения.
        Проверка по mean — это ошибка, которая стоит всей задачи.

        ЖЁСТКО vs МЯГКО. Отбраковывает только source: spec — требование
        спецификации. Показатели с source: assumption (вспышка, ПТФ,
        плотность) — НАШИ модельные допущения, а не промышленные пределы,
        и ТЗ прямо запрещает выдавать одно за другое. Отбрасывать вариант
        по придуманной нами границе значит молча сузить рабочую область
        установки: лимит ПТФ 0 C для летнего сорта зарежет любой зимний
        режим. Поэтому такие отклонения идут в warnings, доходят до
        оператора в отчёте, но допустимости кандидата не отменяют.

        Диапазоны управляемых переменных и размер шага остаются ЖЁСТКИМИ,
        хотя тоже помечены assumption: ТЗ отдельным пунктом требует, чтобы
        рекомендация не выходила за заданные технологические и модельные
        ограничения, а шаг — это trust region модели.
        """
        for param, spec in self.specs.items():
            hard = spec.get("source") == "spec"
            bucket = violated if hard else warnings

            iv: Interval = cand.predicted.get(param)
            if iv is None:
                bucket.append(f"{param}: прогноз отсутствует, проверка невозможна")
                continue

            limit = float(spec["limit"])
            value = iv.hi if spec["check_on"] == "hi" else iv.lo
            if not _is_number(value):
                bucket.append(
                    f"{param}: прогноз не число ({value}), проверка невозможна"
                )
                continue
            checked.append(
                f"{param} {spec['direction']} {limit} по {spec['check_on']} [{spec['source']}]"
            )

            if spec["direction"] == "max":
                margin = limit - value
            else:
                margin = value - limit

            margins[param] = round(margin, 3)
            if margin < 0:
                bucket.append(
                    f"{param}: {value:.2f} против лимита {limit:.2f} "
                    f"(проверка по {spec['check_on']}), "
                    + ("нарушение " if hard else "отклонение от допущения ")
                    + f"{abs(margin):.2f}"
                )

    # ------------------------------------------------------------------
    def _check_ranges(self, cand, state, allowed_ranges, violated, margins, checked):
        """Абсолютное значение после изменения должно остаться в диапазоне."""
        allowed_ranges = allowed_ranges or {}
        for tag, delta in cand.deltas.items():
            spec = self.mvars.get(tag)
            if spec is None:
                # Правило границ ТЗ: без подтверждённого диапазона параметр
                # трогать нельзя. Раньше здесь падал KeyError — жёсткий
                # фильтр обязан отбраковывать, а не ронять весь цикл.
                violated.append(
                    f"{tag}: не входит в список управляемых переменных, "
                    f"допустимый диапазон не задан"
                )
                continue

            cur = state.tag(tag)
            if cur is None:
                violated.append(f"{tag}: текущее значение неизвестно")
                continue
            if not _is_number(delta):
                violated.append(f"{tag}: изменение не число ({delta})")
                continue
            new = cur + delta

            lo, hi = spec["range"]
            src = spec["source"]
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
            spec = self.mvars.get(tag)
            if spec is None or not _is_number(delta):
                continue            # уже отбраковано в _check_ranges
            max_step = float(spec["max_step"])
            checked.append(f"|delta {tag}| <= {max_step}")
            margin = max_step - abs(delta)
            margins[f"{tag}__step"] = round(margin, 3)
            if margin < 0:
                violated.append(
                    f"{tag}: шаг {delta:+.2f} больше допустимого {max_step}"
                )

    # ------------------------------------------------------------------
    def _check_blending(self, fractions, violated, margins, checked):
        """
        Доли компонентов блендинга обязаны давать 100 процентов (ТЗ, раздел 4).

        Сумма долей равна единице по построению, поэтому одной этой проверки
        мало: она тавтологична. Вторая проверка содержательная — доля каждого
        компонента должна остаться в исторически наблюдавшемся окне
        share_range, то есть система не предлагает пропорцию пула, которой
        установка никогда не видела.

        fractions — отображение 'тег компонента -> доля' либо просто
        последовательность долей: тогда проверяется только сумма.
        """
        values = list(fractions.values()) if isinstance(fractions, Mapping) else list(fractions)
        if not all(_is_number(f) for f in values):
            violated.append("доли компонентов блендинга не числа")
            return

        total = float(sum(values))
        target = float(self.blending["components_sum"])
        tol = float(self.blending["tolerance"])
        checked.append(f"сумма долей блендинга = {target}")
        # ЗАПАС, а не разность: разность target - total отрицательна при
        # любом превышении, включая укладывающееся в допуск, и оркестратор
        # читал такого кандидата как нарушителя.
        margins["blend_sum"] = round(tol - abs(total - target), 9)
        if abs(total - target) > tol:
            violated.append(f"сумма долей блендинга {total:.6f} вместо {target}")
        if any(f < 0 for f in values):
            violated.append("отрицательная доля компонента блендинга")

        if not isinstance(fractions, Mapping):
            return

        src = self.blending.get("share_source", "assumption")
        for tag, share in fractions.items():
            window = self.blending.get("share_range", {}).get(tag)
            if window is None:
                continue
            lo, hi = float(window[0]), float(window[1])
            checked.append(f"доля {tag} в [{lo}, {hi}] [{src}]")
            margin = min(share - lo, hi - share)
            margins[f"{tag}__share"] = round(margin, 4)
            if margin < 0:
                violated.append(
                    f"{tag}: доля в пуле {share:.3f} вне окна [{lo}, {hi}]"
                )


def filter_passed(candidates, verdicts) -> List[Candidate]:
    ok = {v.candidate_id for v in verdicts if v.passed}
    return [c for c in candidates if c.candidate_id in ok]
