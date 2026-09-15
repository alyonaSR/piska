"""
Обёртка "формула ВАК как baseline + LightGBM на остатках".

Зона ответственности: Person 3 (ML Engineer).

Общий паттерн soft-сенсора для всех показателей AVTModel и не-серных
показателей GOModel: если для показателя есть формула ВАК, из неё
берётся физически осмысленная опорная линия, и модель учится только на
том, что формула не объясняет. Если формулы нет (или она сломана --
см. vak_formulas.avt_240_350_cfpp) -- formula_fn=None, тогда baseline=0
и это вырождается в обычный ML на признаках, без специального кода.

Почему так, а не отдельный predict() на каждый показатель:
  - формула уже несёт знак и физику (Аррениус, материальный баланс),
    LightGBM учит только то, чего формула не знает -- меньше данных
    нужно для той же точности, чем на голом ML
  - один класс тестируется один раз, а не N раз на N показателей

Интервал -- split conformal prediction с онлайн-адаптацией (Adaptive
Conformal Inference), см. conformal.py. Stage 3 из research.pdf,
заменяет прежнюю эвристику "эмпирические 10/90 перцентили остатка без
поправки на конечную выборку и без доказанной гарантии покрытия".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from ..contracts import Interval
from .conformal import ConformalResidualBounds

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


class NotFittedError(RuntimeError):
    pass


def _safe_name(col: str) -> str:
    """LightGBM запрещает спецсимволы в именах колонок, у нас 'AVT:F30'."""
    return col.replace(":", "__")


@dataclass
class FormulaPlusResidual:
    name: str
    formula_fn: Optional[Callable[[Mapping], float]]
    formula_tags: List[str]
    feature_cols: List[str]
    monotone: Optional[List[int]] = None
    fallback_mean: float = 0.0  # если formula_fn=None и модель не обучена

    def __post_init__(self):
        self._model = None
        self._conformal: Optional[ConformalResidualBounds] = None
        self._resid_bias = 0.0  # медиана остатка на калибровке, см. fit()
        self._n_train = 0
        self._n_calib = 0

    # ------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        return self._model is not None or self.formula_fn is not None

    def baseline(self, row: Mapping) -> float:
        if self.formula_fn is None:
            return 0.0
        try:
            return float(self.formula_fn(row))
        except (KeyError, TypeError, ZeroDivisionError):
            # входа формулы не хватает (деградированное состояние) --
            # вырождается в чистый residual, не падает
            return 0.0

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series, train_frac: float = 0.8,
            **lgbm_kwargs) -> "FormulaPlusResidual":
        """
        X: строки времени (сортировка по ts гарантируется здесь), колонки
           -- объединение formula_tags и feature_cols.
        y: целевая величина, тот же индекс.

        Сплит хронологический (train_frac по времени, не случайный),
        ТЗ запрещает shuffle для временных рядов.
        """
        if lgb is None:
            raise ImportError("pip install lightgbm")

        order = X.index.sort_values()
        X = X.loc[order]
        y = y.loc[order]

        n = len(X)
        cut = int(n * train_frac)
        X_train, X_calib = X.iloc[:cut], X.iloc[cut:]
        y_train, y_calib = y.iloc[:cut], y.iloc[cut:]

        base_train = X_train.apply(lambda r: self.baseline(r), axis=1)
        resid_train = y_train - base_train

        params = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                       num_leaves=15, min_child_samples=max(5, n // 50),
                       random_state=42, verbose=-1)
        params.update(lgbm_kwargs)
        if self.monotone is not None:
            params["monotone_constraints"] = self.monotone

        model = lgb.LGBMRegressor(**params)
        # LightGBM не принимает ':' в именах колонок (наш формат тега
        # 'AVT:F30') -- санитизируем только на границе с LightGBM,
        # feature_cols и вход predict_one остаются в исходном формате
        safe_X = X_train[self.feature_cols].rename(columns=_safe_name)
        model.fit(safe_X, resid_train)
        self._model = model
        self._n_train = len(X_train)

        if len(X_calib) >= 5:
            base_calib = X_calib.apply(lambda r: self.baseline(r), axis=1)
            safe_calib = X_calib[self.feature_cols].rename(columns=_safe_name)
            pred_calib = base_calib + model.predict(safe_calib)
            err = (y_calib - pred_calib)

            # НАЙДЕНО (Stage 2 bias-фикс): калибровочный остаток не
            # центрирован в нуле -- модель обучена на первых train_frac
            # исторических точках, а calib -- уже дальше по времени
            # (chronological split, не shuffle). На реальном демо-сценарии
            # (дата ближе к концу истории, чем к train-части) это давало
            # систематическое ЗАВЫШЕНИЕ серы на ~2-2.5 мг/кг относительно
            # ЛИМС -- не шум, устойчивый сдвиг на всех трёх demo-сценариях.
            # Медиана остатка -- честная оценка сдвига (устойчивее к
            # выбросам, чем среднее), добавляется к точечному прогнозу.
            self._resid_bias = float(np.median(err))
            err_debiased = err - self._resid_bias

            # Stage 3: split conformal + ACI вместо наивных np.quantile.
            # Калибруется на ДЕ-СМЕЩЁННОМ остатке, чтобы интервал был
            # честной оценкой оставшейся неопределённости вокруг уже
            # скорректированного центра, а не заодно тащил на себе
            # исправление смещения.
            self._conformal = ConformalResidualBounds().fit(err_debiased)
            self._n_calib = len(X_calib)
        else:
            # мало данных на калибровку -- эвристический запас,
            # честно шире, чем typical residual std
            spread = float(resid_train.std()) if len(resid_train) > 1 else 1.0
            self._resid_bias = 0.0
            self._conformal = ConformalResidualBounds().fit(
                np.array([-1.5 * spread, 1.5 * spread])
            )

        return self

    # ------------------------------------------------------------------
    def predict_one(self, features: Dict[str, float]) -> Interval:
        base = self.baseline(features)
        if self.formula_fn is None and base == 0.0:
            # нет ни формулы, ни (пока) обученной модели -- 0.0 физически
            # бессмысленно для большинства показателей, используем
            # опорную константу вместо неё, пока не обучено
            base = self.fallback_mean

        if self._model is None:
            # формула без обученного остатка -- честно широкий интервал,
            # чтобы Gate не поверил точечному прогнозу больше, чем он стоит
            half = 8.0
            return Interval(base, base - half, base + half)

        row = pd.DataFrame([{c: features.get(c) for c in self.feature_cols}])
        safe_row = row[self.feature_cols].rename(columns=_safe_name)
        resid = float(self._model.predict(safe_row)[0])
        # + resid_bias: коррекция систематического сдвига калибровки, см. fit()
        mean = base + resid + (self._resid_bias or 0.0)
        if self._conformal is not None:
            offset_lo, offset_hi = self._conformal.bounds()
        else:
            offset_lo, offset_hi = -1.0, 1.0
        lo, hi = mean + offset_lo, mean + offset_hi
        if lo > hi:
            lo, hi = hi, lo
        return Interval(mean, lo, hi)

    # ------------------------------------------------------------------
    def observe(self, features: Dict[str, float], y_true: float) -> None:
        """
        Adaptive Conformal Inference: онлайн-шаг, когда пришёл РЕАЛЬНЫЙ
        факт (новый анализ ЛИМС) для ранее сделанного прогноза.

        Не вызывается автоматически ни из какого продакшен-цикла --
        для этого нужен живой поток решений с обратной связью, это
        зона Orchestrator (Person 1). Метод готов быть подключённым,
        покрыт tests/test_conformal.py.
        """
        if self._model is None or self._conformal is None:
            return
        row = pd.DataFrame([{c: features.get(c) for c in self.feature_cols}])
        safe_row = row[self.feature_cols].rename(columns=_safe_name)
        resid = float(self._model.predict(safe_row)[0])
        pred = self.baseline(features) + resid + self._resid_bias
        self._conformal.update(y_true - pred)

    # ------------------------------------------------------------------
    def to_state(self) -> dict:
        return {
            "name": self.name, "model": self._model,
            "conformal": self._conformal.to_state() if self._conformal else None,
            "resid_bias": self._resid_bias,
            "n_train": self._n_train, "n_calib": self._n_calib,
            "fallback_mean": self.fallback_mean,
        }

    @classmethod
    def from_state(cls, state: dict, formula_fn, formula_tags, feature_cols,
                    monotone=None, fallback_mean: float = 0.0):
        """
        БАГ, НАЙДЕН И ИСПРАВЛЕН (до Stage 3): cls(...) без fallback_mean
        тихо обнулял его (дефолт дата-класса 0.0), хотя вызывающая
        сторона (AVTModel.load / GOModel.load) прекрасно знает правильное
        значение из _SPECS. Для показателя без формулы и без обученного
        остатка (feed_flash_c до появления LIMS-точки с flash_c на АВТ)
        predict_one() тогда возвращал Interval(0.0, -8.0, 8.0) вместо
        Interval(68.0, 60.0, 76.0) -- физически бессмысленный ноль вместо
        честного "формула отсутствует, используем опорную константу".
        Это и роняло spec_risk_prob до 1.0 на demo-сценарии normal.

        Артефакты, сохранённые ДО Stage 3, хранят resid_lo/resid_hi
        (плоские числа) вместо conformal (state калибратора) -- строим
        ConformalResidualBounds из них как разовый откат, дальше
        используется честная калибровка при следующем переобучении.
        """
        obj = cls(state["name"], formula_fn, formula_tags, feature_cols, monotone)
        obj._model = state["model"]
        if state.get("conformal") is not None:
            obj._conformal = ConformalResidualBounds.from_state(state["conformal"])
        elif state.get("resid_lo") is not None and state.get("resid_hi") is not None:
            legacy = ConformalResidualBounds()
            legacy._hi_scores = [state["resid_hi"]]
            legacy._lo_scores = [-state["resid_lo"]]
            obj._conformal = legacy
        obj._resid_bias = state.get("resid_bias", 0.0)
        obj._n_train = state.get("n_train", 0)
        obj._n_calib = state.get("n_calib", 0)
        obj.fallback_mean = state.get("fallback_mean", fallback_mean)
        return obj
