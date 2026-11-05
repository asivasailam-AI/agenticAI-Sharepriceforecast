import os
import json
import joblib
import warnings

import numpy as np
import pandas as pd

from nselib import capital_market as cm

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.neural_network import (
    MLPRegressor
)

from sklearn.preprocessing import (
    StandardScaler
)

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

from statsmodels.tsa.arima.model import (
    ARIMA
)

warnings.filterwarnings("ignore")

# =====================================================
# CONFIG
# =====================================================

SYMBOL = "ITCHOTELS"

FROM_DATE = "01-01-2021"

TO_DATE = "15-10-2026"

N_LAGS = 10

TEST_SIZE = 0.20

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# DATA AGENT
# =====================================================

def load_data():

    raw = cm.price_volume_data(
        symbol=SYMBOL,
        from_date=FROM_DATE,
        to_date=TO_DATE
    )

    raw.columns = [
        str(col).lower().strip()
        for col in raw.columns
    ]

    raw["date"] = pd.to_datetime(
        raw["date"]
    )

    close_prices = (
        raw["closeprice"]
        .astype("string")
        .str.replace(",", "", regex=False)
    )
    raw["close"] = pd.to_numeric(
        close_prices
    )

    raw = (
        raw
        .sort_values("date")
        .drop_duplicates("date")
        .set_index("date")
    )

    return raw


# =====================================================
# FEATURE AGENT
# =====================================================

def make_features(close):

    df = pd.DataFrame()

    df["close"] = close

    for lag in range(
        1,
        N_LAGS + 1
    ):
        df[f"lag_{lag}"] = (
            close.shift(lag)
        )

    df["return_1"] = (
        close.pct_change()
    )

    df["return_5"] = (
        close.pct_change(5)
    )

    df["sma_5"] = (
        close.rolling(5).mean()
    )

    df["sma_10"] = (
        close.rolling(10).mean()
    )

    df["sma_20"] = (
        close.rolling(20).mean()
    )

    df["ema_10"] = (
        close
        .ewm(span=10)
        .mean()
    )

    df["ema_20"] = (
        close
        .ewm(span=20)
        .mean()
    )

    df["volatility_10"] = (
        close
        .pct_change()
        .rolling(10)
        .std()
    )

    delta = close.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["rsi_14"] = (
        100 - (100 / (1 + rs))
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
            (actual - predicted)
            / actual
        )
    ) * 100

    return rmse, mae, mape


# =====================================================
# TRAINING
# =====================================================

print("\nLoading NSE data...")

df = load_data()

feature_df = make_features(
    df["close"]
)

feature_columns = [
    c for c in feature_df.columns
    if c != "close"
]

X = feature_df[feature_columns]

y = feature_df["close"]

split_idx = int(
    len(feature_df)
    * (1 - TEST_SIZE)
)

X_train = X.iloc[:split_idx]

X_test = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]

y_test = y.iloc[split_idx:]

# =====================================================
# RANDOM FOREST
# =====================================================

print("Training Random Forest")

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train,
    y_train
)

rf_pred = rf.predict(X_test)

rf_rmse, rf_mae, rf_mape = evaluate(
    y_test,
    rf_pred
)

joblib.dump(
    rf,
    f"{MODEL_DIR}/random_forest.pkl"
)

# =====================================================
# GRADIENT BOOSTING
# =====================================================

print("Training Gradient Boosting")

gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb.fit(
    X_train,
    y_train
)

gb_pred = gb.predict(X_test)

gb_rmse, gb_mae, gb_mape = evaluate(
    y_test,
    gb_pred
)

joblib.dump(
    gb,
    f"{MODEL_DIR}/gradient_boosting.pkl"
)

# =====================================================
# MLP
# =====================================================

print("Training MLP")

mlp = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "mlp",
        MLPRegressor(
            hidden_layer_sizes=(64,32),
            random_state=42,
            max_iter=500
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

mlp_rmse, mlp_mae, mlp_mape = evaluate(
    y_test,
    mlp_pred
)

joblib.dump(
    mlp,
    f"{MODEL_DIR}/mlp.pkl"
)

# =====================================================
# ARIMA
# =====================================================

print("Training ARIMA")

arima_model = ARIMA(
    y_train.to_numpy(),
    order=(5,1,0)
)

arima_fit = (
    arima_model.fit()
)

arima_pred = arima_fit.forecast(
    steps=len(y_test)
)

arima_rmse, arima_mae, arima_mape = evaluate(
    y_test,
    arima_pred
)

joblib.dump(
    arima_fit,
    f"{MODEL_DIR}/arima.pkl"
)

# =====================================================
# SCORECARD
# =====================================================

scores = pd.DataFrame([

    {
        "Model":"ARIMA",
        "RMSE":arima_rmse,
        "MAE":arima_mae,
        "MAPE":arima_mape
    },

    {
        "Model":"Random Forest",
        "RMSE":rf_rmse,
        "MAE":rf_mae,
        "MAPE":rf_mape
    },

    {
        "Model":"Gradient Boosting",
        "RMSE":gb_rmse,
        "MAE":gb_mae,
        "MAPE":gb_mape
    },

    {
        "Model":"MLP",
        "RMSE":mlp_rmse,
        "MAE":mlp_mae,
        "MAPE":mlp_mape
    }

])

scores = scores.sort_values(
    "RMSE"
)

scores.to_csv(
    f"{MODEL_DIR}/scorecard.csv",
    index=False
)

print(scores)

# =====================================================
# BEST MODEL
# =====================================================

best_model = (
    scores.iloc[0]["Model"]
)

leaderboard = {

    SYMBOL: best_model

}

with open(
    f"{MODEL_DIR}/leaderboard.json",
    "w"
) as f:

    json.dump(
        leaderboard,
        f,
        indent=4
    )

# =====================================================
# FEATURE METADATA
# =====================================================

joblib.dump(
    feature_columns,
    f"{MODEL_DIR}/feature_columns.pkl"
)

print("\nTraining completed")

print(
    f"\nBest Model = {best_model}"
)

print("\nSaved:")

print("random_forest.pkl")
print("gradient_boosting.pkl")
print("mlp.pkl")
print("arima.pkl")
print("leaderboard.json")
print("feature_columns.pkl")
print("scorecard.csv")