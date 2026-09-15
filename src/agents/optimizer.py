"""
L2. Агент оптимизации.

Зона ответственности: Person 4.

ЗАДАЧА: сгенерировать допустимых кандидатов внутри trust region
и оценить каждого. Отбраковку делает НЕ он, а Gate.

TRUST REGION: кандидаты строятся как дельты не больше max_step
из config/constraints.yaml и не выходят за allowed_ranges агента
надёжности. Это защита от экстраполяции модели в режимы,
которых в истории не было.

ПУТЬ РАЗВИТИЯ:
  шаг 1 (сегодня): сетка по 2-3 управляемым переменным. Быстро, объяснимо,
                   работает на защите.
  шаг 2: scipy.optimize.minimize с penalty или optuna
  шаг 3: Парето-фронт по (качество, экономика, severity) через
         фильтрацию недоминируемых точек
"""

from __future__ import annotations

import itertools
from typing import Dict, List

from ..contracts import Candidate, ProcessState, QualityAssess, ReliabilityAssess
from ..data.tags import manipulated_vars


class OptimizerAgent:
    def __init__(self, quality_agent, n_steps: int = 3):
        self.quality = quality_agent
        self.n_steps = n_steps      # число шагов в каждую сторону по каждой переменной

    # ------------------------------------------------------------------
    def propose(
        self,
        state: ProcessState,
        quality: QualityAssess,
        reliability: ReliabilityAssess,
        active_vars: List[str] = None,
    ) -> List[Candidate]:
        active_vars = active_vars or ["242000:T5", "AVT:F30"]  # см. constraints.yaml
        specs = manipulated_vars()

        grids: Dict[str, List[float]] = {}
        for tag in active_vars:
            step = specs[tag]["max_step"] / self.n_steps
            grids[tag] = [round(step * k, 3) for k in range(-self.n_steps, self.n_steps + 1)]

        candidates: List[Candidate] = []
        for i, combo in enumerate(itertools.product(*[grids[t] for t in active_vars])):
            deltas = {t: d for t, d in zip(active_vars, combo)}
            if not self._within_allowed(state, deltas, reliability):
                continue
            pred = self.quality.assess(state, deltas=deltas).predictions
            candidates.append(
                Candidate(
                    candidate_id=f"c_{i:03d}",
                    deltas=deltas,
                    predicted=pred,
                    cost_proxy=self._cost_proxy(deltas),
                    severity_delta=self._severity_delta(deltas),
                    yield_delta=deltas.get("AVT:F30", 0.0),
                )
            )
        return candidates

    # ------------------------------------------------------------------
    @staticmethod
    def _within_allowed(state, deltas, reliability) -> bool:
        """Кандидат не должен выводить параметр за allowed_ranges."""
        for tag, d in deltas.items():
            cur = state.tag(tag)
            if cur is None:
                return False
            lo, hi = reliability.allowed_ranges.get(tag, [-1e9, 1e9])
            if not (lo <= cur + d <= hi):
                return False
        return True

    # ------------------------------------------------------------------
    @staticmethod
    def _cost_proxy(deltas: Dict[str, float]) -> float:
        """
        Прозрачный стоимостной прокси. Фактических экономических данных
        в пакете нет, ТЗ такое разрешает при явном описании.

        Условные единицы за цикл:
          +1.0 за каждый градус температуры реактора (топливо + водород)
          -0.6 за каждую т/ч отбора дизельной фракции (выручка)

        TODO(Person 4): откалибровать веса или заменить на энергозатраты.
        """
        return (1.0 * deltas.get("242000:T5", 0.0)
                - 0.25 * deltas.get("AVT:F30", 0.0)
                - 0.25 * deltas.get("AVT:F32", 0.0))

    @staticmethod
    def _severity_delta(deltas: Dict[str, float]) -> float:
        """Рост температуры реактора = более жёсткий режим = быстрее деактивация."""
        return round(0.04 * deltas.get("242000:T5", 0.0), 4)

    # ------------------------------------------------------------------
    @staticmethod
    def pareto_front(candidates: List[Candidate]) -> List[Candidate]:
        """
        Недоминируемые точки по (cost_proxy, severity_delta, sulfur hi).
        Все три минимизируются. Необязательный пункт ТЗ, но дешёвый.
        """
        def key(c: Candidate):
            s = c.predicted.get("sulfur_mgkg")
            return (c.cost_proxy, c.severity_delta, s.hi if s else 0.0)

        front: List[Candidate] = []
        for c in candidates:
            kc = key(c)
            if not any(
                all(ko <= kk for ko, kk in zip(key(o), kc)) and key(o) != kc
                for o in candidates
            ):
                front.append(c)
        return front
