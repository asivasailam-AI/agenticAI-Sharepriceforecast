import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def split_by_symbol(feature_df, test_size):

    train_parts = []
    test_parts = []

    for symbol, symbol_df in feature_df.groupby(
            "symbol",
            sort=False):
        split_idx = int(
            len(symbol_df) * (1 - test_size)
        )
        if split_idx == 0 or split_idx == len(symbol_df):
            raise ValueError(
                f"Not enough rows to create train and test sets for {symbol}."
            )

        train_parts.append(
            symbol_df.iloc[:split_idx]
        )
        test_parts.append(
            symbol_df.iloc[split_idx:]
        )

    if not train_parts:
        raise ValueError("No ticker data is available for model evaluation.")

    return (
        pd.concat(train_parts),
        pd.concat(test_parts)
    )


def forecast_arima_test(train_df, test_df, order=(5, 1, 0)):

    actual_parts = []
    prediction_parts = []

    for symbol, symbol_test in test_df.groupby(
            "symbol",
            sort=False):
        symbol_train = train_df.loc[
            train_df["symbol"] == symbol
        ]
        if symbol_train.empty:
            raise ValueError(
                f"No training data is available for {symbol}."
            )

        fitted_model = ARIMA(
            symbol_train["target"].to_numpy(),
            order=order
        ).fit()

        prediction = fitted_model.forecast(
            steps=len(symbol_test)
        )
        actual_parts.append(
            symbol_test["target"].to_numpy()
        )
        prediction_parts.append(
            np.asarray(prediction)
        )

    return (
        np.concatenate(actual_parts),
        np.concatenate(prediction_parts)
    )
