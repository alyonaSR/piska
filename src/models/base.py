"""
Общий интерфейс предсказателей качества.

Зона ответственности: Person 3 (ML Engineer).

ГРАНИЦА ОТВЕТСТВЕННОСТИ. Модель в этом пакете:
  - принимает ПЛОСКИЙ словарь признаков {имя: число}
  - возвращает словарь {показатель: Interval}
  - НЕ знает про ProcessState, спецификации, возраст анализов и confidence

Всё перечисленное — работа QualityAgent, который эти модели оборачивает.
Такое разделение нужно по двум причинам. Person 1 и Person 3 не правят
один файл. И модель можно тестировать отдельно, подав ей словарь чисел,
без сборки всего состояния установки.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..contracts import Interval


class BaseQualityModel(ABC):
    """
    Контракт предсказателя.

    Любая реализация — регрессия, XGBoost, LightGBM, формула с листа ВАК,
    ансамбль — должна выглядеть снаружи одинаково.
    """

    #: какие признаки модель ждёт на входе
    required_features: List[str] = []
    #: какие показатели она возвращает
    outputs: List[str] = []
    #: для трассировки и воспроизводимости
    model_id: str = "base"

    # ------------------------------------------------------------------
    @abstractmethod
    def predict(self, features: Dict[str, float]) -> Dict[str, Interval]:
        """Признаки -> показатели с интервалом. Единственный обязательный метод."""

    # ------------------------------------------------------------------
    def check_features(self, features: Dict[str, float]) -> List[str]:
        """Каких признаков не хватает. Возвращается, а не кидается."""
        return [f for f in self.required_features
                if f not in features or features[f] is None]

    def fit(self, X, y) -> "BaseQualityModel":
        """
        TODO(Person 3).

        ОБЯЗАТЕЛЬНО: сплит строго по времени.
        train до 2025-12-31, test 2026 год. Случайное перемешивание строк
        временного ряда запрещено ТЗ — оно даёт утечку из будущего.
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "BaseQualityModel":
        raise NotImplementedError


# ----------------------------------------------------------------------
# Ожидаемые знаки коэффициентов.
#
# ГЛАВНЫЙ РИСК ВСЕЙ ЗАДАЧИ: модель, обученная на наблюдательных данных,
# выучивает КОНТУР РЕГУЛИРОВАНИЯ, а не физику. В истории температуру
# реактора поднимали тогда, когда сырьё было плохим. Наивная модель
# увидит "температура вверх -> сера вверх" и порекомендует СНИЖАТЬ
# температуру при риске превышения серы. Это ровно наоборот.
#
# Проверка: обучить модель, посмотреть знаки SHAP. Если не совпали
# с таблицей ниже — модель в текущем виде опасна.
#
# Митигация: monotone_constraints в LightGBM/XGBoost по этим знакам.
# ----------------------------------------------------------------------

EXPECTED_SIGNS: Dict[str, Dict[str, int]] = {
    "sulfur_mgkg": {
        "go_reactor_temp_c": -1,     # Аррениус: горячее -> глубже обессеривание
        "go_feed_rate": +1,          # выше объёмная скорость -> меньше времени контакта
        "feed_t95_c": +1,            # тяжелее хвост -> труднее удаляемая сера
        "catalyst_age_days": +1,     # катализатор садится
        "h2_rate": -1,               # больше водорода -> глубже реакция
    },
    "d15_kgm3": {
        "feed_t95_c": +1,
    },
    "cfpp_c": {
        "feed_t95_c": +1,
    },
}


def monotone_vector(feature_names: List[str], target: str) -> List[int]:
    """
    Готовый вектор для monotone_constraints в LightGBM / XGBoost.

        model = lgb.LGBMRegressor(
            monotone_constraints=monotone_vector(cols, "sulfur_mgkg")
        )

    Признаки, для которых знак неизвестен, получают 0 (ограничения нет).
    """
    signs = EXPECTED_SIGNS.get(target, {})
    return [signs.get(name, 0) for name in feature_names]
