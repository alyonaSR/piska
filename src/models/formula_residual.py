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

Интервал сейчас -- эмпирические квантили остатка на калибровочном
хвосте (chronological, без shuffle). TODO(Stage 3): заменить на split
conformal / adaptive conformal inference из research.pdf.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from ..contracts import Interval

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
        self._resid_lo = None   # квантиль 0.1 остатка на калибровке
        self._resid_hi = None   # квантиль 0.9 остатка на калибровке
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
            self._resid_lo = float(np.quantile(err, 0.1))
            self._resid_hi = float(np.quantile(err, 0.9))
            self._n_calib = len(X_calib)
        else:
            # мало данных на калибровку -- эвристический запас,
            # честно шире, чем typical residual std
            spread = float(resid_train.std()) if len(resid_train) > 1 else 1.0
            self._resid_lo, self._resid_hi = -1.5 * spread, 1.5 * spread

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
        mean = base + resid
        lo = mean + (self._resid_lo or -1.0)
        hi = mean + (self._resid_hi or 1.0)
        if lo > hi:
            lo, hi = hi, lo
        return Interval(mean, lo, hi)

    # ------------------------------------------------------------------
    def to_state(self) -> dict:
        return {
            "name": self.name, "model": self._model,
            "resid_lo": self._resid_lo, "resid_hi": self._resid_hi,
            "n_train": self._n_train, "n_calib": self._n_calib,
        }

    @classmethod
    def from_state(cls, state: dict, formula_fn, formula_tags, feature_cols, monotone=None):
        obj = cls(state["name"], formula_fn, formula_tags, feature_cols, monotone)
        obj._model = state["model"]
        obj._resid_lo = state["resid_lo"]
        obj._resid_hi = state["resid_hi"]
        obj._n_train = state.get("n_train", 0)
        obj._n_calib = state.get("n_calib", 0)
        return obj
