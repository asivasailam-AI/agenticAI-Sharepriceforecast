import numpy as np


def forecast_price(
        model,
        feature_row):

    prediction = model.predict(
        feature_row
    )

    return float(
        np.asarray(
            prediction
        )[0]
    )