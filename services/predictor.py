# services/predictor.py

# services/predictor.py

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

FEATURE_COLUMNS = joblib.load(
    MODEL_DIR / "feature_columns.pkl"
)

TICKER_ENCODER = joblib.load(
    MODEL_DIR / "ticker_encoder.pkl"
)

MODELS = {

    "Random Forest":
    joblib.load(
        MODEL_DIR / "random_forest.pkl"
    ),

    "Gradient Boosting":
    joblib.load(
        MODEL_DIR / "gradient_boosting.pkl"
    ),

    "MLP":
    joblib.load(
        MODEL_DIR / "mlp.pkl"
    )
}

SUPPORTED_MODEL_NAMES = frozenset(
    (*MODELS, "ARIMA")
)


def get_model(model_name):

    if model_name == "ARIMA":
        raise ValueError(
            "ARIMA is fitted from the requested stock history and has no "
            "saved estimator."
        )
    return MODELS[model_name]


def get_feature_columns():

    return FEATURE_COLUMNS


def _scorecard_rows(symbol=None):
    scorecard_path = MODEL_DIR / "scorecard.csv"
    if not scorecard_path.exists():
        return pd.DataFrame()

    scorecard = pd.read_csv(scorecard_path)
    if symbol is None:
        if "Symbol" in scorecard.columns:
            return scorecard.loc[
                scorecard["Symbol"].astype(str).str.upper() == "GLOBAL"
            ]
        return scorecard

    if "Symbol" not in scorecard.columns:
        return scorecard.iloc[0:0]
    return scorecard.loc[
        scorecard["Symbol"].astype(str).str.upper()
        == symbol.upper().strip()
    ]


def get_model_metrics(model_name, symbol=None):

    metrics = {
        "RMSE": None,
        "MAE": None,
        "MAPE": None
    }
    scorecard = _scorecard_rows(symbol)
    if scorecard.empty:
        return metrics
    model_scores = scorecard.loc[
        scorecard["Model"] == model_name
    ]

    if model_scores.empty:
        return metrics

    for metric_name in metrics:
        if metric_name in model_scores.columns:
            value = pd.to_numeric(
                model_scores[metric_name].iloc[0],
                errors="coerce"
            )
            if pd.notna(value) and np.isfinite(value):
                metrics[metric_name] = float(value)

    return metrics


def get_model_rankings(symbol=None):
    scorecard = _scorecard_rows(symbol)
    if scorecard.empty:
        return []
    if "Model" not in scorecard.columns or "RMSE" not in scorecard.columns:
        raise ValueError(
            "The model scorecard must contain Model and RMSE columns."
        )

    scorecard["RMSE"] = pd.to_numeric(
        scorecard["RMSE"],
        errors="coerce"
    )
    scorecard = scorecard.loc[
        np.isfinite(scorecard["RMSE"])
    ].sort_values(
        "RMSE",
        kind="stable"
    )

    rankings = []
    for rank, (_, row) in enumerate(scorecard.iterrows()):
        ranking = {
            "rank": rank + 1,
            "Model": row["Model"]
        }
        for metric_name in ("RMSE", "MAE", "MAPE"):
            value = pd.to_numeric(
                row.get(metric_name, np.nan),
                errors="coerce"
            )
            ranking[metric_name] = (
                float(value)
                if pd.notna(value) and np.isfinite(value)
                else None
            )
        rankings.append(ranking)

    return rankings


def encode_symbol(symbol):

    symbol = symbol.upper().strip()

    try:
        return int(
            TICKER_ENCODER.transform([symbol])[0]
        )
    except ValueError as exc:
        supported_symbols = ", ".join(
            TICKER_ENCODER.classes_
        )
        raise ValueError(
            f"Symbol {symbol!r} was not included in model training. "
            f"Supported symbols: {supported_symbols}"
        ) from exc


def prepare_features(df):

    return df[FEATURE_COLUMNS]