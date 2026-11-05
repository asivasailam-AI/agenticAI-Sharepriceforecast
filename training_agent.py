import os
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
    n_estimators=500,
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
    f"{MODEL_DIR}/random_forest.pkl"
)

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

scores = pd.DataFrame([

    {
        "Model":
        "Random Forest",

        "RMSE": rf_rmse,
        "MAE": rf_mae,
        "MAPE": rf_mape
    },

    {
        "Model":
        "Gradient Boosting",

        "RMSE": gb_rmse,
        "MAE": gb_mae,
        "MAPE": gb_mape
    },

    {
        "Model":
        "MLP",

        "RMSE": mlp_rmse,
        "MAE": mlp_mae,
        "MAPE": mlp_mape
    },

    {
        "Model":
        "ARIMA",

        "RMSE": arima_rmse,
        "MAE": arima_mae,
        "MAPE": arima_mape
    }

])

scores = scores.sort_values(
    "RMSE"
)

scores.to_csv(
    f"{MODEL_DIR}/scorecard.csv",
    index=False
)

deployable_scores = scores.loc[
    scores["Model"] != "ARIMA"
]
best_model = deployable_scores.iloc[0]["Model"]

with open(
    f"{MODEL_DIR}/leaderboard.json",
    "w"
) as f:

    json.dump(

        {
            "GLOBAL_MODEL":
            best_model
        },

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