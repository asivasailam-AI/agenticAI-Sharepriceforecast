#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =========================================================
# 1. IMPORTS
# =========================================================

from nselib import capital_market as cm

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from rich.console import Console
console = Console()
import warnings
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from nselib import capital_market as cm

from rich.console import Console
from rich.table import Table

warnings.filterwarnings("ignore")
logging.getLogger("statsmodels").setLevel(logging.ERROR)
console = Console()

RANDOM_STATE = 42


# In[2]:


# =========================================================
# 3. CONFIGURATION
# =========================================================

# NSE symbol, e.g. RELIANCE, TCS, INFY, SBIN, HDFCBANK
SYMBOL = "ITCHOTELS"

# nselib expects dd-mm-YYYY for custom ranges.
FROM_DATE = "01-01-2021"
TO_DATE   = "15-10-2026"

# Number of future market sessions to forecast.
FORECAST_DAYS = 5

# Last percentage of observations reserved for chronological testing.
TEST_SIZE = 0.20

# Number of historical lags used by ML/NN agents.
N_LAGS = 10

# Save files
RAW_DATA_FILE = f"{SYMBOL}_NSE_raw.csv"
CLEAN_DATA_FILE = f"{SYMBOL}_NSE_clean.csv"
FORECAST_FILE = f"{SYMBOL}_multi_agent_forecast.csv"
MODEL_SCORES_FILE = f"{SYMBOL}_model_scores.csv"

print(f"Symbol: {SYMBOL}")
print(f"Date range: {FROM_DATE} → {TO_DATE}")


# In[3]:


# =========================================================
# NSE DATA AGENT
# =========================================================
import pandas as pd
def nse_data_agent(symbol: str, from_date: str, to_date: str) -> pd.DataFrame:

    console.print(
        f"\n[bold cyan]NSE Data Agent[/bold cyan] "
        f"→ Fetching {symbol}"
    )

    # Fetch data from NSE through nselib
    raw = cm.price_volume_data(
        symbol=symbol,
        from_date=from_date,
        to_date=to_date
    )

    if raw is None or raw.empty:
        raise ValueError(
            f"No NSE data returned for {symbol} "
            f"between {from_date} and {to_date}"
        )

    # -----------------------------------------------------
    # Standardize NSE column names
    # -----------------------------------------------------

    column_mapping = {
        "symbol": "symbol",
        "series": "series",
        "date": "date",
        "prevclose": "prev_close",
        "openprice": "open",
        "highprice": "high",
        "lowprice": "low",
        "lastprice": "last",
        "closeprice": "close",
        "averageprice": "average_price",
        "totaltradedquantity": "volume",
        "turnover₹": "turnover",
        "no.oftrades": "trades"
    }

    # Convert column names to lowercase for safer matching
    raw.columns = [
        str(col).strip().lower()
        for col in raw.columns
    ]

    # Check required columns
    required_nse_columns = [
        "date",
        "openprice",
        "highprice",
        "lowprice",
        "closeprice"
    ]

    missing = [
        col for col in required_nse_columns
        if col not in raw.columns
    ]

    if missing:
        raise ValueError(
            f"NSE response is missing required columns: {missing}. "
            f"Available columns: {list(raw.columns)}"
        )

    # Rename columns
    df = raw.rename(columns=column_mapping)

    # -----------------------------------------------------
    # Data type conversion
    # -----------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "last",
        "average_price",
        "volume",
        "turnover",
        "trades"
    ]

    for col in numeric_columns:
        if col in df.columns:
            values = (
                df[col]
                .astype("string")
                .str.replace(",", "", regex=False)
            )
            df[col] = pd.to_numeric(
                values,
                errors="coerce"
            )

    # -----------------------------------------------------
    # Clean data
    # -----------------------------------------------------

    df = df.dropna(
        subset=[
            "date",
            "open",
            "high",
            "low",
            "close"
        ]
    )

    df = df.sort_values("date")

    df = df.drop_duplicates(
        subset=["date"],
        keep="last"
    )

    df = df.set_index("date")

    console.print(
        f"[green]✓ NSE data loaded: "
        f"{len(df):,} trading sessions[/green]"
    )

    console.print(
        f"Date range: {df.index.min().date()} "
        f"→ {df.index.max().date()}"
    )

    return df

# Agent 1 output
nse_data = nse_data_agent(
    SYMBOL,
    FROM_DATE,
    TO_DATE
)

print("NSE data shape:", nse_data.shape)
display(nse_data.tail())


# In[4]:


# ============================================================
# AGENT 2 — FEATURE ENGINEERING AGENT
# ============================================================

def make_features(close_series):

    x = pd.DataFrame(index=close_series.index)

    # Target
    x["close"] = close_series

    # Lag features
    for lag in range(1, N_LAGS + 1):
        x[f"lag_{lag}"] = close_series.shift(lag)

    # Returns
    x["return_1d"] = close_series.pct_change()

    # Moving averages
    x["sma_5"] = close_series.rolling(5).mean()
    x["sma_10"] = close_series.rolling(10).mean()
    x["sma_20"] = close_series.rolling(20).mean()

    # Exponential moving averages
    x["ema_5"] = close_series.ewm(span=5, adjust=False).mean()
    x["ema_10"] = close_series.ewm(span=10, adjust=False).mean()
    x["ema_20"] = close_series.ewm(span=20, adjust=False).mean()

    # Volatility
    x["volatility_10"] = (
        close_series.pct_change().rolling(10).std()
    )

    # RSI
    delta = close_series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    x["rsi_14"] = 100 - (100 / (1 + rs))

    return x.dropna()


# Use Agent 1 output
feature_agent_output = make_features(
    nse_data["close"]
)

console.print("[bold cyan]Feature Engineering Agent[/bold cyan]")
print("Feature matrix:", feature_agent_output.shape)

display(feature_agent_output.tail())


# In[5]:


# ============================================================
# AGENT 3 — TRAIN / TEST SPLIT
# ============================================================

# Use the output produced by Agent 2
data = feature_agent_output.copy()

# Chronological split — DO NOT shuffle time-series data
split_index = int(len(data) * (1 - TEST_SIZE))

train_df = data.iloc[:split_index].copy()
test_df = data.iloc[split_index:].copy()

console.print("[bold cyan]Train/Test Split Agent[/bold cyan]")

print(f"Total observations : {len(data)}")
print(f"Training observations : {len(train_df)}")
print(f"Testing observations  : {len(test_df)}")

print(f"\nTraining period: {train_df.index.min()} → {train_df.index.max()}")
print(f"Testing period : {test_df.index.min()} → {test_df.index.max()}")

display(train_df.tail())
display(test_df.head())


# In[6]:


# ============================================================
# 7. METRICS
# ============================================================

def evaluate(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))

    # Avoid division by zero
    mask = actual != 0
    mape = (
        np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask]))
        * 100
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape
    }


# In[7]:


# =========================================================
# 9. AGENT 4 — MACHINE LEARNING FORECASTING AGENT
# =========================================================

ML_FEATURES = [
    *[f"lag_{i}" for i in range(1, N_LAGS + 1)],
    "return_1",
    "return_5",
    "sma_5",
    "sma_10",
    "sma_20",
    "ema_10",
    "ema_20",
    "volatility_10",
    "rsi_14",
]


def latest_feature_row(history: pd.Series) -> pd.DataFrame:
    """Generate one prediction row from the latest known/forecast history."""
    f = make_features(history, N_LAGS)
    return f[ML_FEATURES].tail(1)


def ml_recursive_forecast(
    history: pd.Series,
    horizon: int,
    model
) -> np.ndarray:
    train_features = make_features(history, N_LAGS)
    X_train = train_features[ML_FEATURES]
    y_train = train_features["target"]

    model.fit(X_train, y_train)

    extended = history.copy()
    predictions = []

    for _ in range(horizon):
        x_next = latest_feature_row(extended)[ML_FEATURES]
        pred = float(model.predict(x_next)[0])
        predictions.append(pred)

        next_index = extended.index[-1] + pd.offsets.BDay(1)
        extended.loc[next_index] = pred

    return np.asarray(predictions)


def machine_learning_agent(train: pd.Series, test: pd.Series):
    console.print(
        "[bold yellow]Machine Learning Agent → "
        "Random Forest + Gradient Boosting[/bold yellow]"
    )

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            loss="huber",
            random_state=RANDOM_STATE,
        ),
    }

    outputs = {}
    for name, model in models.items():
        pred = ml_recursive_forecast(train, len(test), model)
        outputs[name] = {
            "name": name,
            "pred": pred,
            "metrics": evaluate(test.values, pred),
            "model": model,
        }

    return outputs


# In[8]:


# =========================================================
# 10. AGENT 5 — NEURAL NETWORK FORECASTING AGENT
# =========================================================

def neural_network_agent(train: pd.Series, test: pd.Series):
    console.print("[bold yellow]Neural Network Agent → MLPRegressor[/bold yellow]")

    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0005,
            learning_rate_init=0.001,
            max_iter=800,
            early_stopping=True,
            random_state=RANDOM_STATE,
        )
    )

    pred = ml_recursive_forecast(train, len(test), model)

    return {
        "name": "MLP",
        "pred": pred,
        "metrics": evaluate(test.values, pred),
        "model": model,
    }


# In[9]:


def fit_full_and_forecast(model_name, history, horizon):

    # Make sure history is numeric
    history = pd.Series(
        history.values,
        dtype="float64"
    )

    # ========================================
    # ARIMA
    # ========================================

    if model_name == "ARIMA":

        model = ARIMA(
            history.values,
            order=(5, 1, 0)
        )

        fitted_model = model.fit()

        forecast = fitted_model.forecast(
            steps=horizon
        )

        return np.asarray(forecast, dtype=float)


    # ========================================
    # Machine Learning Models
    # ========================================

    if model_name in [
        "Random Forest",
        "Gradient Boosting",
        "MLP"
    ]:

        values = history.values

        X = []
        y = []

        for i in range(N_LAGS, len(values)):

            X.append(
                values[i-N_LAGS:i][::-1]
            )

            y.append(values[i])

        X = np.asarray(X)
        y = np.asarray(y)


        # ------------------------------------
        # Random Forest
        # ------------------------------------

        if model_name == "Random Forest":

            model = RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            )


        # ------------------------------------
        # Gradient Boosting
        # ------------------------------------

        elif model_name == "Gradient Boosting":

            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )


        # ------------------------------------
        # MLP
        # ------------------------------------

        else:

            model = make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    max_iter=500,
                    random_state=42
                )
            )


        # Train model
        model.fit(X, y)

        # Recursive forecast
        return ml_recursive_forecast(
            history,
            horizon,
            model
        )


    raise ValueError(
        f"Unknown model: {model_name}"
    )


# In[10]:


def ml_recursive_forecast(history, horizon, model):

    # Convert history to numeric values only
    values = np.asarray(history, dtype=float).copy()

    predictions = []

    for _ in range(horizon):

        # Use the same feature structure used during model training
        X_next = values[-N_LAGS:][::-1].reshape(1, -1)

        # Predict next closing price
        prediction = model.predict(X_next)[0]

        predictions.append(float(prediction))

        # Add prediction to history
        values = np.append(values, prediction)

    return np.asarray(predictions, dtype=float)


# In[11]:


model_predictions = {}

forecast_history = train_df["close"]
forecast_horizon = len(test_df)

print("Generating model predictions...")

model_predictions["ARIMA"] = fit_full_and_forecast(
    "ARIMA",
    forecast_history,
    forecast_horizon
)

model_predictions["Random Forest"] = fit_full_and_forecast(
    "Random Forest",
    forecast_history,
    forecast_horizon
)

model_predictions["Gradient Boosting"] = fit_full_and_forecast(
    "Gradient Boosting",
    forecast_history,
    forecast_horizon
)

model_predictions["MLP"] = fit_full_and_forecast(
    "MLP",
    forecast_history,
    forecast_horizon
)

print("\nModel Predictions Generated:")

for model_name, predictions in model_predictions.items():
    print(
        f"{model_name:20s} → "

        f"{len(predictions)} predictions"
    )


# In[12]:


model_predictions = {}

forecast_history = train_df["close"]
forecast_horizon = len(test_df)

print("Generating model predictions...")

model_predictions["ARIMA"] = fit_full_and_forecast(
    "ARIMA",
    forecast_history,
    forecast_horizon
)

model_predictions["Random Forest"] = fit_full_and_forecast(
    "Random Forest",
    forecast_history,
    forecast_horizon
)

model_predictions["Gradient Boosting"] = fit_full_and_forecast(
    "Gradient Boosting",
    forecast_history,
    forecast_horizon
)

model_predictions["MLP"] = fit_full_and_forecast(
    "MLP",
    forecast_history,
    forecast_horizon
)

print("\nModel Predictions Generated:")

for model_name, predictions in model_predictions.items():
    print(
        f"{model_name:20s} → "
        f"{len(predictions)} predictions"
    )


# In[13]:


# ============================================================
# VALIDATION AGENT
# ============================================================

def validation_agent(train_df, test_df, model_predictions):
    """
    Validate forecasting models using the NSE test dataset.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset.
    test_df : pd.DataFrame
        Holdout/test dataset containing actual close prices.
    model_predictions : dict
        Dictionary containing predictions from each forecasting agent.

    Returns
    -------
    validation_results : dict
        RMSE, MAE and MAPE for each model.
    """

    console.print("\n[bold cyan]Validation Agent[/bold cyan]")

    # Actual NSE closing prices
    actual_close = test_df["close"].values

    validation_results = {}

    for model_name, predicted in model_predictions.items():

        # Convert prediction to numpy array
        predicted = np.asarray(predicted, dtype=float)

        # Make sure actual and predicted lengths match
        n = min(len(actual_close), len(predicted))

        actual = actual_close[:n]
        pred = predicted[:n]

        # Calculate metrics
        metrics = evaluate(actual, pred)

        validation_results[model_name] = metrics

        print(
            f"{model_name:20s} | "
            f"RMSE: {metrics['RMSE']:.4f} | "
            f"MAE: {metrics['MAE']:.4f} | "
            f"MAPE: {metrics['MAPE']:.2f}%"
        )

    return validation_results

validation_results = validation_agent(
    train_df,
    test_df,
    model_predictions
)


# In[14]:


print("model_predictions exists:",
      "model_predictions" in globals())

print("validation_results exists:",
      "validation_results" in globals())


# In[15]:


validation_results = validation_agent(
    train_df,
    test_df,
    model_predictions
)


# In[16]:


import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))

plt.plot(
    test_df.index,
    test_df["close"].values,
    label="Actual",
    linewidth=2
)

for name, predictions in model_predictions.items():

    predictions = np.asarray(predictions, dtype=float)

    n = min(len(test_df), len(predictions))

    plt.plot(
        test_df.index[:n],
        predictions[:n],
        label=name,
        alpha=0.8
    )

plt.title(
    f"{SYMBOL} — NSE Holdout Forecast Comparison"
)

plt.xlabel("Date")
plt.ylabel("Closing Price")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[17]:


print("nse_data:", "nse_data" in globals())
print("feature_agent_output:", "feature_agent_output" in globals())
print("train_df:", "train_df" in globals())
print("test_df:", "test_df" in globals())


# In[18]:


# ============================================================
# FORECASTING AGENTS
# ============================================================

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

console.print("\n[bold cyan]Forecasting Agents[/bold cyan]")

model_predictions = {}

# ------------------------------------------------------------
# AGENT 3A — ARIMA
# ------------------------------------------------------------

try:
    arima_model = ARIMA(
        train_df["close"].to_numpy(),
        order=(5, 1, 0)
    )

    arima_fit = arima_model.fit()

    arima_pred = arima_fit.forecast(
        steps=len(test_df)
    )

    model_predictions["ARIMA"] = np.asarray(arima_pred)

    print("✓ ARIMA predictions generated")

except Exception as e:
    print(f"✗ ARIMA failed: {e}")


# ------------------------------------------------------------
# Prepare ML features
# ------------------------------------------------------------

feature_columns = [
    col for col in train_df.columns
    if col != "close"
]

X_train = train_df[feature_columns]
y_train = train_df["close"]

X_test = test_df[feature_columns]


# ------------------------------------------------------------
# AGENT 3B — Random Forest
# ------------------------------------------------------------

try:

    rf_model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)

    model_predictions["Random Forest"] = rf_pred

    print("✓ Random Forest predictions generated")

except Exception as e:
    print(f"✗ Random Forest failed: {e}")


# ------------------------------------------------------------
# AGENT 3C — Gradient Boosting
# ------------------------------------------------------------

try:

    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    gb_model.fit(X_train, y_train)

    gb_pred = gb_model.predict(X_test)

    model_predictions["Gradient Boosting"] = gb_pred

    print("✓ Gradient Boosting predictions generated")

except Exception as e:
    print(f"✗ Gradient Boosting failed: {e}")


# ------------------------------------------------------------
# AGENT 3D — Neural Network / MLP
# ------------------------------------------------------------

try:

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp_model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=500,
        random_state=42
    )

    mlp_model.fit(
        X_train_scaled,
        y_train
    )

    mlp_pred = mlp_model.predict(X_test_scaled)

    model_predictions["MLP"] = mlp_pred

    print("✓ MLP predictions generated")

except Exception as e:
    print(f"✗ MLP failed: {e}")




# ------------------------------------------------------------
# Check results
# ------------------------------------------------------------

print("\nModels successfully generated:")

for name, pred in model_predictions.items():
    print(
        f"{name:20s} → "
        f"{len(pred)} predictions"
    )


# In[19]:


print("model_predictions exists:",
      "model_predictions" in globals())

print("\nAvailable models:")

for name, pred in model_predictions.items():
    print(name, len(pred))


# In[20]:


# ============================================================
# VALIDATION AGENT
# ============================================================

def validation_agent(train_df, test_df, model_predictions):
    """
    Validate forecasting models using the NSE test dataset.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset.
    test_df : pd.DataFrame
        Holdout/test dataset containing actual close prices.
    model_predictions : dict
        Dictionary containing predictions from each forecasting agent.

    Returns
    -------
    validation_results : dict
        RMSE, MAE and MAPE for each model.
    """

    console.print("\n[bold cyan]Validation Agent[/bold cyan]")

    # Actual NSE closing prices
    actual_close = test_df["close"].values

    validation_results = {}

    for model_name, predicted in model_predictions.items():

        # Convert prediction to numpy array
        predicted = np.asarray(predicted, dtype=float)

        # Make sure actual and predicted lengths match
        n = min(len(actual_close), len(predicted))

        actual = actual_close[:n]
        pred = predicted[:n]

        # Calculate metrics
        metrics = evaluate(actual, pred)

        validation_results[model_name] = metrics

        print(
            f"{model_name:20s} | "
            f"RMSE: {metrics['RMSE']:.4f} | "
            f"MAE: {metrics['MAE']:.4f} | "
            f"MAPE: {metrics['MAPE']:.2f}%"
        )

    return validation_results


# In[21]:


# ============================================================
# CREATE MODEL SCORECARD
# ============================================================

scores = pd.DataFrame([
    {
        "Model": model,
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "MAPE": metrics["MAPE"]
    }
    for model, metrics in validation_results.items()
])

# Sort by RMSE
scores = scores.sort_values(
    by=["RMSE", "MAE"],
    ascending=[True, True]
).reset_index(drop=True)

console.print("\n[bold cyan]Model Scorecard[/bold cyan]")

display(scores)


# In[22]:


validation_results = validation_agent(
    train_df,
    test_df,
    model_predictions
)


# In[23]:


print(validation_results)


# In[24]:


import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))

plt.plot(
    test_df.index,
    test_df["close"].values,
    label="Actual",
    linewidth=2
)

for name, predictions in model_predictions.items():

    predictions = np.asarray(predictions, dtype=float)

    n = min(len(test_df), len(predictions))

    plt.plot(
        test_df.index[:n],
        predictions[:n],
        label=name,
        alpha=0.8
    )

plt.title(
    f"{SYMBOL} — NSE Holdout Forecast Comparison"
)

plt.xlabel("Date")
plt.ylabel("Closing Price")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# In[25]:


# ============================================================
# DECISION AGENT
# ============================================================

def decision_agent(results, scores: pd.DataFrame):

    console.print(
        "[bold magenta]Decision Agent Running[/bold magenta]"
    )

    # Rank primarily by RMSE,
    # with MAE as a secondary signal.

    ranked_scores = scores.sort_values(
        by=["RMSE", "MAE"],
        ascending=[True, True]
    ).reset_index(drop=True)

    best_model = ranked_scores.iloc[0]["Model"]

    print("\nModel Ranking:")
    display(ranked_scores)

    print(f"\n🏆 Best Model: {best_model}")

    return best_model, ranked_scores


# In[26]:


def final_forecast_agent(history, best_model, horizon):

    console.print(
        "[bold red]Final Forecast Agent Running[/bold red]"
    )

    # Forecast using the selected best model
    forecast = fit_full_and_forecast(
        best_model,
        history,
        horizon
    )

    # Create forecast dates
    last_date = history.index[-1]

    forecast_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon
    )

    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Forecast": forecast
    })

    forecast_df.set_index("Date", inplace=True)

    print(f"\n🏆 Selected Model: {best_model}")
    print(f"📈 Forecast Horizon: {horizon} trading days")

    display(forecast_df)

    return forecast_df


# In[27]:


print("validation_results:", "validation_results" in globals())
print("scores:", "scores" in globals())
print("best_model:", "best_model" in globals())


# In[28]:


# ============================================================
# DECISION AGENT
# ============================================================

def decision_agent(results, scores):

    console.print(
        "[bold magenta]Decision Agent Running[/bold magenta]"
    )

    # Rank models by RMSE, then MAE
    ranked_scores = scores.sort_values(
        by=["RMSE", "MAE"],
        ascending=[True, True]
    ).reset_index(drop=True)

    # Select best-performing model
    best_model = ranked_scores.iloc[0]["Model"]

    print("\nModel Ranking:")
    display(ranked_scores)

    print(f"\n🏆 Best Model: {best_model}")

    return best_model, ranked_scores


# In[29]:


best_model, ranked_scores = decision_agent(
    validation_results,
   scores
)


# In[30]:


print("Best model:", best_model)
print("\nRanked models:")
display(ranked_scores)


# In[31]:


def final_forecast_agent(history, best_model, horizon):

    console.print(
        "[bold red]Final Forecast Agent Running[/bold red]"
    )

    forecast = fit_full_and_forecast(
        best_model,
        history,
        horizon
    )

    last_date = history.index[-1]

    forecast_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon
    )

    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Forecast": forecast
    })

    forecast_df.set_index("Date", inplace=True)
    print(f"\n Company: {SYMBOL}")
    print(f"\n🏆 Selected Model: {best_model}")
    print(f"📈 Forecast Horizon: {horizon} trading days")

    display(forecast_df)

    return forecast_df, forecast


# In[32]:


forecast_df, forecast_matrix = final_forecast_agent(
    nse_data["close"],
    best_model,
    FORECAST_DAYS
)
