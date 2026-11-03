"""Pipeline de regresión de la Fase 2, consistente con la Fase 1."""
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder

from src.features import (
    HIGH_CARD_FEATURES,
    LOW_CARD_FEATURES,
    NUMERIC_FEATURES,
    clean_raw,
)

# Parámetros seleccionados en la Fase 1 (model_v1_metadata.json).
BEST_PARAMS = {
    "min_samples_leaf": 10,
    "max_leaf_nodes": 63,
    "max_iter": 1200,
    "max_features": 0.5,
    "learning_rate": 0.05,
    "l2_regularization": 1.0,
}


def build_preprocessor(seed: int = 42) -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    low_card = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Desconocido")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=20,
            sparse_output=False,
        )),
    ])
    high_card = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Desconocido")),
        ("target_enc", TargetEncoder(
            target_type="continuous",
            cv=KFold(n_splits=5, shuffle=True, random_state=seed),
        )),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("low_card", low_card, LOW_CARD_FEATURES),
        ("high_card", high_card, HIGH_CARD_FEATURES),
    ])


def make_model(seed: int = 42) -> TransformedTargetRegressor:
    """Incluye limpieza, imputación, codificación y estimador en un solo artefacto."""
    regressor = Pipeline([
        ("clean", FunctionTransformer(clean_raw, validate=False)),
        ("preprocess", build_preprocessor(seed)),
        ("model", HistGradientBoostingRegressor(**BEST_PARAMS, random_state=seed)),
    ])
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )
