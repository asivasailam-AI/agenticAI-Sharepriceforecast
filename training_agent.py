import os
import sys
import json
import warnings
import joblib

import numpy as np
import pandas as pd

from nselib import capital_market as cm

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.neural_network import (
    MLPRegressor
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

from statsmodels.tsa.arima.model import ARIMA
from agents.model_evaluation import (
    forecast_arima_test,
    split_by_symbol
)

warnings.filterwarnings("ignore")

# =====================================================
# CONFIG
# =====================================================

SYMBOLS = [

    "RELIANCE",
    "TCS",
    "INFY",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "AXISBANK",
    "KOTAKBANK",
    "LT",
    "ITC",
    "IRCTC",
    "IOB",
    "SUNTV",
    "INDIANB",
    "PAGEIND",
    "WIPRO",
    "ITCHOTELS",
    "APOLLOHOSP",
    "FORTIS",
    "WAAREEENER",
    "NDTV",
    "TECHM",
    "TATASTEEL",
    "BHARTIARTL"
]

FROM_DATE = "01-01-2021"
TO_DATE = "03-11-2026"

N_LAGS = 10

TEST_SIZE = 0.20

MODEL_DIR = "models"
RANDOM_FOREST_ONLY = "--random-forest-only" in sys.argv

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

# =====================================================
# DATA AGENT
# =====================================================

def load_data(symbol):

    raw = cm.price_volume_data(
        symbol=symbol,
        from_date=FROM_DATE,
        to_date=TO_DATE
    )

    if raw is None or raw.empty:
        return None

    raw.columns = [

        str(col)
        .lower()
        .strip()

        for col in raw.columns

    ]

    raw["date"] = pd.to_datetime(
        raw["date"]
    )

    close_prices = (

        raw["closeprice"]

        .astype("string")

        .str.replace(
            ",",
            "",
            regex=False
        )

    )

    raw["close"] = pd.to_numeric(
        close_prices,
        errors="coerce"
    )

    raw = (

        raw

        .sort_values("date")

        .drop_duplicates("date")

        .set_index("date")

    )

    return raw


# =====================================================
# FEATURE ENGINEERING
# =====================================================

def make_features(close_series):

    df = pd.DataFrame()

    df["close"] = close_series

    for lag in range(
        1,
        N_LAGS + 1
    ):

        df[f"lag_{lag}"] = (
            close_series.shift(lag)
        )

    df["return_1"] = (
        close_series.pct_change()
    )

    df["return_5"] = (
        close_series.pct_change(5)
    )

    df["sma_5"] = (
        close_series
        .rolling(5)
        .mean()
    )

    df["sma_10"] = (
        close_series
        .rolling(10)
        .mean()
    )

    df["sma_20"] = (
        close_series
        .rolling(20)
        .mean()
    )

    df["ema_10"] = (
        close_series
        .ewm(span=10)
        .mean()
    )

    df["ema_20"] = (
        close_series
        .ewm(span=20)
        .mean()
    )

    df["volatility_10"] = (

        close_series

        .pct_change()

        .rolling(10)

        .std()

    )

    delta = close_series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(
        14
    ).mean()

    avg_loss = loss.rolling(
        14
    ).mean()

    rs = avg_gain / avg_loss

    df["rsi_14"] = (

        100

        -

        (
            100
            /
            (1 + rs)
        )

    )

    return df.dropna()


# =====================================================
# METRICS
# =====================================================

def evaluate(actual, predicted):

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    mape = np.mean(

        np.abs(

            (
                actual -
                predicted
            )

            / actual

        )

    ) * 100

    return rmse, mae, mape


def update_random_forest_scorecard(test_df, y_test, predictions):

    test_symbols = test_df["symbol"].to_numpy()
    rows = []

    rmse, mae, mape = evaluate(y_test, predictions)
    rows.append({
        "Symbol": "GLOBAL",
        "Model": "Random Forest",
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape
    })

    for symbol in pd.unique(test_symbols):
        symbol_mask = test_symbols == symbol
        rmse, mae, mape = evaluate(
            y_test.iloc[np.flatnonzero(symbol_mask)],
            predictions[symbol_mask]
        )
        rows.append({
            "Symbol": symbol,
            "Model": "Random Forest",
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape
        })

    scorecard_path = f"{MODEL_DIR}/scorecard.csv"
    if os.path.exists(scorecard_path):
        scorecard = pd.read_csv(scorecard_path)
        if "Model" not in scorecard.columns:
            raise ValueError(
                "The existing scorecard has no Model column."
            )
        scorecard = scorecard.loc[
            scorecard["Model"] != "Random Forest"
        ]
    else:
        scorecard = pd.DataFrame()

    scorecard = pd.concat(
        [scorecard, pd.DataFrame(rows)],
        ignore_index=True
    )
    scorecard.to_csv(scorecard_path, index=False)

    if "Symbol" in scorecard.columns:
        leaderboard = {}
        for symbol, symbol_scores in scorecard.groupby("Symbol"):
            valid_scores = symbol_scores.copy()
            valid_scores["RMSE"] = pd.to_numeric(
                valid_scores["RMSE"],
                errors="coerce"
            )
            valid_scores = valid_scores.loc[
                np.isfinite(valid_scores["RMSE"])
            ].sort_values("RMSE", kind="stable")
            if valid_scores.empty:
                continue
            winner = valid_scores.iloc[0]["Model"]
            if symbol == "GLOBAL":
                leaderboard["GLOBAL_MODEL"] = winner
            else:
                leaderboard[symbol] = winner

        with open(f"{MODEL_DIR}/leaderboard.json", "w") as file:
            json.dump(leaderboard, file, indent=4)

    return rows[0]


# =====================================================
# COLLECT DATA
# =====================================================

all_frames = []

print("\nFetching NSE data...\n")

for symbol in SYMBOLS:

    try:

        print(
            f"Loading {symbol}"
        )

        df = load_data(
            symbol
        )

        if df is None:
            continue

        features = make_features(
            df["close"]
        )

        features["symbol"] = symbol

        all_frames.append(
            features
        )

    except Exception as e:

        print(
            f"{symbol} failed: {e}"
        )

feature_df = pd.concat(
    all_frames
)

print(
    f"\nTotal rows: "
    f"{len(feature_df):,}"
)

# =====================================================
# TICKER ENCODER
# =====================================================

encoder = LabelEncoder()

feature_df["ticker_encoded"] = (

    encoder.fit_transform(
        feature_df["symbol"]
    )

)

joblib.dump(

    encoder,

    f"{MODEL_DIR}/ticker_encoder.pkl"

)

# =====================================================
# FORECAST TARGET
# =====================================================

feature_df["target"] = (

    feature_df
    .groupby("symbol")
    ["close"]
    .shift(-1)

)

feature_df = (
    feature_df.dropna()
)

# =====================================================
# FEATURES
# =====================================================

feature_columns = [

    col

    for col in feature_df.columns

    if col not in [

        "close",
        "target",
        "symbol"

    ]

]

joblib.dump(

    feature_columns,

    f"{MODEL_DIR}/feature_columns.pkl"

)

X = feature_df[
    feature_columns
]

y = feature_df[
    "target"
]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

train_df, test_df = split_by_symbol(
    feature_df,
    TEST_SIZE
)

X_train = train_df[feature_columns]
X_test = test_df[feature_columns]
y_train = train_df["target"]
y_test = test_df["target"]

arima_actual, arima_pred = forecast_arima_test(
    train_df,
    test_df
)
arima_rmse, arima_mae, arima_mape = evaluate(
    arima_actual,
    arima_pred
)

# =====================================================
# RANDOM FOREST
# =====================================================

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train,
    y_train
)

rf_pred = rf.predict(X_test)
rf_rmse, rf_mae, rf_mape = evaluate(y_test, rf_pred)

joblib.dump(
    rf,
    f"{MODEL_DIR}/random_forest.pkl.tmp"
)
os.replace(
    f"{MODEL_DIR}/random_forest.pkl.tmp",
    f"{MODEL_DIR}/random_forest.pkl"
)

if RANDOM_FOREST_ONLY:
    rf_score = update_random_forest_scorecard(
        test_df,
        y_test,
        np.asarray(rf_pred)
    )
    print("\nRandom Forest retraining completed")
    print(f"Validation RMSE: {rf_score['RMSE']:.4f}")
    print(f"Validation MAE: {rf_score['MAE']:.4f}")
    print(f"Validation MAPE: {rf_score['MAPE']:.4f}%")
    print(
        "Saved compact model to "
        f"{MODEL_DIR}/random_forest.pkl"
    )
    raise SystemExit(0)

# =====================================================
# GRADIENT BOOSTING
# =====================================================

gb = GradientBoostingRegressor(
    n_estimators=250,
    learning_rate=0.05,
    random_state=42
)

gb.fit(
    X_train,
    y_train
)

gb_pred = gb.predict(X_test)
gb_rmse, gb_mae, gb_mape = evaluate(y_test, gb_pred)

joblib.dump(
    gb,
    f"{MODEL_DIR}/gradient_boosting.pkl"
)

# =====================================================
# MLP
# =====================================================

mlp = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "mlp",
        MLPRegressor(
            hidden_layer_sizes=(64, 32),
            random_state=42,
            max_iter=700
        )
    )

])

mlp.fit(
    X_train,
    y_train
)

mlp_pred = mlp.predict(
    X_test
)
mlp_rmse, mlp_mae, mlp_mape = evaluate(y_test, mlp_pred)

joblib.dump(
    mlp,
    f"{MODEL_DIR}/mlp.pkl"
)

# =====================================================
# SCORECARD
# =====================================================

model_predictions = {
    "Random Forest": np.asarray(rf_pred),
    "Gradient Boosting": np.asarray(gb_pred),
    "MLP": np.asarray(mlp_pred),
    "ARIMA": np.asarray(arima_pred)
}
test_symbols = test_df["symbol"].to_numpy()
scores_rows = []

for model_name, predictions in model_predictions.items():
    if len(predictions) != len(test_df):
        raise ValueError(
            f"{model_name} produced {len(predictions)} predictions for "
            f"{len(test_df)} validation rows."
        )

    rmse, mae, mape = evaluate(y_test, predictions)
    scores_rows.append({
        "Symbol": "GLOBAL",
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape
    })

    for symbol in pd.unique(test_symbols):
        symbol_mask = test_symbols == symbol
        rmse, mae, mape = evaluate(
            y_test.iloc[np.flatnonzero(symbol_mask)],
            predictions[symbol_mask]
        )
        scores_rows.append({
            "Symbol": symbol,
            "Model": model_name,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape
        })

scores = pd.DataFrame(scores_rows).sort_values(
    ["Symbol", "RMSE"],
    kind="stable"
)

scores.to_csv(
    f"{MODEL_DIR}/scorecard.csv",
    index=False
)

global_scores = scores.loc[scores["Symbol"] == "GLOBAL"]
best_model = global_scores.iloc[0]["Model"]
leaderboard = {"GLOBAL_MODEL": best_model}

for symbol in pd.unique(test_symbols):
    symbol_scores = scores.loc[scores["Symbol"] == symbol]
    leaderboard[symbol] = symbol_scores.iloc[0]["Model"]

with open(
    f"{MODEL_DIR}/leaderboard.json",
    "w"
) as f:

    json.dump(

        leaderboard,

        f,

        indent=4

    )

print("\nModel Ranking")

print(scores)

print(
    f"\nBest Model: "
    f"{best_model}"
)

print(
    "\nArtifacts Created:"
)

print(
    "ticker_encoder.pkl"
)

print(
    "feature_columns.pkl"
)

print(
    "random_forest.pkl"
)

print(
    "gradient_boosting.pkl"
)

print(
    "mlp.pkl"
)

print(
    "leaderboard.json"
)