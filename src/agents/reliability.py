"""
L1b. Агент надёжности.

Зона ответственности: Person 2 или Person 3.

ЗАДАЧА: оценить тяжесть режима и вернуть allowed_ranges — границы,
за которые оптимизатор выходить не имеет права. Это единственное место
в системе, где один агент реально ограничивает другой.

ПРОКСИ-МЕТРИКИ (прямой разметки деактивации в пакете нет, ТЗ такое разрешает
при явном описании допущений):

  1. WABT = T_вход + 2/3 * (T_выход - T_вход)
     Стандартная отраслевая формула средневзвешенной температуры слоя
     для адиабатического реактора гидроочистки.

  2. Нормализованная температура: сколько градусов нужно, чтобы получить
     целевую серу при фиксированном расходе и качестве сырья. Её рост
     во времени = потеря активности катализатора. Считается обратной
     задачей к модели агента качества.

  3. Скорость деактивации (TIR): наклон нормализованной температуры,
     градусов в месяц. Растёт при более жёстком режиме.

ГЛАВНОЕ ДОПУЩЕНИЕ: дата начала цикла катализатора неизвестна.
Принимаем 2023-01-01 как условное начало. Помечено в assumptions.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..contracts import ProcessState, ReliabilityAssess
from ..data.tags import manipulated_vars


class ReliabilityAgent:
    def __init__(self, cycle_start_iso: str = "2023-01-01"):
        self.cycle_start_iso = cycle_start_iso

    def assess(self, state: ProcessState) -> ReliabilityAssess:
        severity, factors = self._severity(state)
        return ReliabilityAssess(
            severity_index=severity,
            severity_class=self._classify(severity),
            factors=factors,
            allowed_ranges=self._allowed_ranges(state, severity),
            assumptions=[
                "WABT считается как Tвх + 2/3*(Tвых - Tвх), стандартная формула",
                f"начало цикла катализатора принято {self.cycle_start_iso}, ДОПУЩЕНИЕ",
                "прямой разметки деактивации в пакете нет, используется прокси",
                "модельные диапазоны из config/constraints.yaml помечены source: assumption",
            ],
        )

    # ------------------------------------------------------------------
    def _severity(self, state: ProcessState):
        """
        TODO(Person 2/3): заменить на нормализованную температуру из
        обратной задачи к модели серы.

        Сейчас — прозрачная линейная свёртка двух факторов:
        температуры реактора и удельной нагрузки.
        """
        factors: List[str] = []
        t5 = state.tag("242000:T5") or 365.0
        feed = state.tag("242000:F26") or 256.0

        # 0 при 345 C, 1 при 380 C
        t_term = (t5 - 345.0) / 35.0
        if t5 > 370.0:
            factors.append(f"температура реактора {t5:.1f} C, близко к верхней границе")

        # 0 при 180, 1 при 320 м3/ч
        load_term = (feed - 180.0) / 140.0
        if feed > 290.0:
            factors.append(f"высокая нагрузка по сырью {feed:.0f} м3/ч, объёмная скорость растёт")

        sev = max(0.0, min(1.0, 0.65 * t_term + 0.35 * load_term))
        if not factors:
            factors.append("режим в пределах обычного диапазона")
        return round(sev, 3), factors

    @staticmethod
    def _classify(sev: float) -> str:
        if sev < 0.5:
            return "normal"
        if sev < 0.75:
            return "elevated"
        return "high"

    # ------------------------------------------------------------------
    def _allowed_ranges(self, state: ProcessState, severity: float) -> Dict[str, List[float]]:
        """
        Сужает модельные диапазоны из конфига по мере роста тяжести режима.
        Именно этот словарь получает оптимизатор и не может его нарушить.
        """
        out: Dict[str, List[float]] = {}
        for tag, spec in manipulated_vars().items():
            lo, hi = spec["range"]
            if severity > 0.75:
                # при высокой тяжести режима отрезаем верхнюю четверть по температуре
                if spec["units"] == "degC":
                    hi = lo + 0.75 * (hi - lo)
            out[tag] = [round(lo, 2), round(hi, 2)]
        return out
