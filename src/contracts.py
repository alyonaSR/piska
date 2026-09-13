"""
Контракты обмена между агентами.

ЕДИНСТВЕННЫЙ файл, который нельзя менять в одиночку.
Любое изменение здесь -> поднять SCHEMA_VERSION и сказать команде.

Намеренно используются только dataclasses из стандартной библиотеки:
проект должен запускаться где угодно без установки зависимостей.
Если захотите валидацию типов на рантайме — замена на pydantic
делается механически, поля не меняются.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

SCHEMA_VERSION = "1.0.0"


# --------------------------------------------------------------------------
# Базовые примитивы
# --------------------------------------------------------------------------

@dataclass
class Measurement:
    """
    Одно измерение из ЛИМС / ПАК / ВАК.

    value     — само значение в единицах из units
    ts        — когда измерено (НЕ когда прочитано)
    age_min   — возраст в минутах относительно ProcessState.ts
    source    — 'LIMS' | 'PAK' | 'VAK'. Приоритет достоверности именно такой.
    healthy   — False если анализатор залип / вне диапазона / помечен как сбойный.
                Значение при этом остаётся, но верить ему нельзя.
    units     — обязательно явно, ТЗ требует воспроизводимых конвертаций
    """
    value: Optional[float]
    ts: Optional[datetime]
    age_min: Optional[float]
    source: str
    units: str
    healthy: bool = True

    def is_usable(self, max_age_min: float) -> bool:
        """Годится ли как факт о текущем состоянии."""
        return (
            self.value is not None
            and self.healthy
            and self.age_min is not None
            and self.age_min <= max_age_min
        )


@dataclass
class Interval:
    """
    Прогноз с интервалом.

    ВАЖНО: жёсткие ограничения проверяются по консервативной границе,
    а не по mean. Для серы (чем меньше тем лучше) это hi.
    Если тут будет одно число — вся логика безопасности развалится.
    """
    mean: float
    lo: float
    hi: float

    def conservative(self, direction: str) -> float:
        """direction='upper' -> hi (для серы), 'lower' -> lo (для вспышки)."""
        return self.hi if direction == "upper" else self.lo

    @property
    def width(self) -> float:
        return self.hi - self.lo


# --------------------------------------------------------------------------
# L0 -> вход системы
# --------------------------------------------------------------------------

@dataclass
class ProcessState:
    """
    Снимок установки на момент ts. Единственный вход всего цикла.

    tags   — ключ ВСЕГДА вида 'установка:код', например 'AVT:T33', '242000:T5'.
             Короткое имя без установки запрещено: код T6 на АВТ и на 24-2000
             означает разные физические величины.
    lims   — последние лабораторные значения, ключ = показатель
    pak    — последние показания поточных анализаторов
    dq_flags — что не так с данными на этот момент
    """
    ts: datetime
    tags: Dict[str, float] = field(default_factory=dict)
    lims: Dict[str, Measurement] = field(default_factory=dict)
    pak: Dict[str, Measurement] = field(default_factory=dict)
    dq_flags: List[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def tag(self, key: str) -> Optional[float]:
        if ":" not in key:
            raise KeyError(
                f"Тег '{key}' без установки. Нужно 'AVT:{key}' или '242000:{key}'."
            )
        return self.tags.get(key)

    def best_quality(self, param: str, max_age_min: float = 24 * 60) -> Optional[Measurement]:
        """
        Приоритет источников из ТЗ: ЛИМС -> ПАК -> ВАК.
        Лабораторный результат считается контрольным фактом.
        """
        for store in (self.lims, self.pak):
            m = store.get(param)
            if m is not None and m.is_usable(max_age_min):
                return m
        return None


# --------------------------------------------------------------------------
# L1 -> оценки агентов
# --------------------------------------------------------------------------

@dataclass
class QualityAssess:
    """
    Ответ агента качества.

    predictions     — показатель -> Interval. Ключи: 'sulfur_mgkg', 'cfpp_c',
                      'flash_c', 'd15_kgm3', 't95_c'
    spec_risk_prob  — 0..1, вероятность нарушить хотя бы одно требование спеки
    confidence      — 0..1, доверие к самому прогнозу.
                      Падает при старом ЛИМС, залипшем ПАК, режиме вне обучающей области.
    drivers         — человекочитаемые причины, идут в объяснение оператору
    model_id        — что за модель отработала, для воспроизводимости
    """
    predictions: Dict[str, Interval] = field(default_factory=dict)
    spec_risk_prob: float = 0.0
    confidence: float = 0.0
    drivers: List[str] = field(default_factory=list)
    model_id: str = "stub"
    schema_version: str = SCHEMA_VERSION


@dataclass
class ReliabilityAssess:
    """
    Ответ агента надёжности.

    severity_index  — 0..1, обобщённая тяжесть режима
    severity_class  — 'normal' | 'elevated' | 'high'
    factors         — что именно даёт тяжесть
    allowed_ranges  — тег -> (min, max). КЛЮЧЕВОЕ ПОЛЕ.
                      Именно так один агент реально ограничивает другой:
                      оптимизатор не может выйти за эти границы.
    assumptions     — все допущения явно, ТЗ требует
    """
    severity_index: float = 0.0
    severity_class: str = "normal"
    factors: List[str] = field(default_factory=list)
    allowed_ranges: Dict[str, List[float]] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION


# --------------------------------------------------------------------------
# L2 -> кандидаты
# --------------------------------------------------------------------------

@dataclass
class Candidate:
    """
    Один вариант изменения режима.

    deltas — ИЗМЕНЕНИЕ, не абсолютное значение: {'242000:T5': -2.0}.
             Так честнее (модель не уверена в точном значении) и так
             естественно получается trust region.
    predicted      — что получится, от агента качества
    cost_proxy     — стоимостной прокси, меньше = лучше
    severity_delta — изменение тяжести режима, отрицательное = мягче
    yield_delta    — изменение выпуска, т/ч
    """
    candidate_id: str
    deltas: Dict[str, float] = field(default_factory=dict)
    predicted: Dict[str, Interval] = field(default_factory=dict)
    cost_proxy: float = 0.0
    severity_delta: float = 0.0
    yield_delta: float = 0.0
    schema_version: str = SCHEMA_VERSION

    @property
    def is_no_action(self) -> bool:
        return all(abs(v) < 1e-9 for v in self.deltas.values())


# --------------------------------------------------------------------------
# L3 -> вердикт жёсткого фильтра
# --------------------------------------------------------------------------

@dataclass
class GateVerdict:
    """
    Результат проверки одного кандидата. Только числа, никаких рассуждений.

    violated — список нарушений человекочитаемо
    margins  — ограничение -> запас. Положительный = есть запас.
               Нужен и для ранжирования, и для фразы в отчёте.
    checked  — какие проверки вообще прогнали (идёт в 'Проверка ограничений')
    """
    candidate_id: str
    passed: bool
    violated: List[str] = field(default_factory=list)
    margins: Dict[str, float] = field(default_factory=dict)
    checked: List[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION


# --------------------------------------------------------------------------
# L4 -> финал
# --------------------------------------------------------------------------

REFUSE = "REFUSE"


@dataclass
class Recommendation:
    """
    Итог оператору. Поля 1:1 с таблицей раздела 5 ТЗ,
    чтобы отчёт печатался механически, а не сочинялся руками.

    action — словарь дельт ЛИБО строка 'REFUSE'
    """
    ts: datetime
    action: Union[Dict[str, float], str]
    reason: str
    expected_effect: Dict[str, Any] = field(default_factory=dict)
    checks_passed: List[str] = field(default_factory=list)
    confidence: float = 0.0
    explanation: str = ""
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    data_freshness: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @property
    def is_refusal(self) -> bool:
        return self.action == REFUSE


@dataclass
class DecisionTrace:
    """
    Полный след одного цикла. Пишется в artifacts/traces/*.json.
    Закрывает критерии 'Воспроизводимость' и 'Объяснимость'.
    """
    ts: datetime
    state: ProcessState
    quality: QualityAssess
    reliability: ReliabilityAssess
    candidates: List[Candidate]
    verdicts: List[GateVerdict]
    recommendation: Recommendation
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj
