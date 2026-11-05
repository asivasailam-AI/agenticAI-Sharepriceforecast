import numpy as np


def evaluate(
        actual,
        predicted):

    rmse = np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )

    mae = np.mean(
        np.abs(
            actual - predicted
        )
    )

    mape = (
        np.mean(
            np.abs(
                (actual - predicted)
                / actual
            )
        ) * 100
    )

    return {

        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape
    }