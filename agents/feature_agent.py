import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import N_LAGS


def make_features(close):

    df = pd.DataFrame()

    df["close"] = close

    for lag in range(
            1,
            N_LAGS + 1):

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