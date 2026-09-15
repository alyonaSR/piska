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
        # ИСПРАВЛЕНО (Person 4): AVT:T33 убрана. По данным Person 1 (см.
        # excluded_candidates в constraints.yaml) corr(T33, EBP)=-0.010 —
        # это не рычаг, влияния на результат нет, только признак модели.
        # Вместо неё — AVT:F32, второй по силе AVT-рычаг после T5 (corr
        # с EBP +0.254). Полный набор из 6 обоснованных переменных
        # (F30/F32/F28/F14/P22/T5) уже лежит в constraints.yaml с
        # grid_points, но полным перебором по всем 6 сразу через grid
        # search работать не может: 11*11*9*11*13*7 ≈ 1.3 млн комбинаций.
        # Роадмап самого файла говорит "шаг 1: сетка по 2-3 переменным" —
        # держим 3 по умолчанию, остальные три подключаются явно через
        # active_vars=[...] и ждут шага 2 (scipy/optuna), где полный
        # перебор не нужен.
        active_vars = active_vars or ["242000:T5", "AVT:F30", "AVT:F32"]
        specs = manipulated_vars()
 
        grids: Dict[str, List[float]] = {}
        for tag in active_vars:
            spec = specs[tag]
            # grid_points задаётся в constraints.yaml персонально по переменной,
            # чтобы шаг получался круглым числом (0.5 degC, 1.0 t/h), а не 1.333.
            # Раньше единый self.n_steps делил max_step без оглядки на переменную.
            n = int(spec.get("grid_points", self.n_steps))
            step = spec["max_step"] / n
            grids[tag] = [round(step * k, 3) for k in range(-n, n + 1)]
 
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
                    # ИСПРАВЛЕНО (Person 4): AVT:F32 тоже часть дизельного
                    # пула (Person 1: "F30 и F32 вместе задают объём и
                    # состав пула"), раньше в yield_delta не учитывался.
                    yield_delta=deltas.get("AVT:F30", 0.0) + deltas.get("AVT:F32", 0.0),
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
        Прозрачный стоимостной прокси ЧИСТЫХ ЗАТРАТ. Фактических
        экономических данных в пакете нет, ТЗ такое разрешает при явном
        описании допущений.
 
        ИСПРАВЛЕНО (Person 4): раньше сюда подмешивался -0.6*ΔF30 — рост
        отбора дизельной фракции одновременно снижал cost_proxy И
        увеличивал yield_delta. Это двойной учёт одной и той же выгоды:
        один раз как "рост выпуска", второй раз как "мнимая экономия".
        Orchestrator сравнивает их как разные критерии лексикографической
        цепочки (сначала запас, потом воздействие, потом severity, и
        только в конце cost_proxy) — cost_proxy обязан быть чистыми
        затратами, без части выгоды внутри.
 
        Условные единицы за цикл:
          +1.0 за каждый градус температуры реактора 242000:T5 (топливо + водород)
        Рост отбора (AVT:F30, AVT:F32) в cost_proxy не входит — он есть
        в yield_delta. AVT:T33 убрана вместе с исключением из active_vars
        (см. propose()) — она не рычаг, а только признак модели.
 
        TODO(Person 4): заменить вес на реальные энергозатраты, когда
        появится хоть один экономический источник данных. Также нет пока
        слагаемого для AVT:F28 (пар в стриппинг) — расход пара стоит
        денег, но это вне активных по умолчанию переменных.
        """
        return 1.0 * deltas.get("242000:T5", 0.0)
 
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
