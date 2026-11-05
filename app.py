import base64
from io import BytesIO

from flask import Flask
from flask import render_template
from flask import request

import pandas as pd
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from agents.data_agent import load_stock_data
from agents.feature_agent import make_features
from agents.decision_agent import DecisionAgent

from services.predictor import (
    get_model,
    get_feature_columns,
    encode_symbol,
    get_model_metrics,
    get_model_rankings
)

app = Flask(__name__)

decision_agent = DecisionAgent()


def create_forecast_chart(symbol, current_price, forecast_data):

    labels = ["Current"] + [
        row["date"]
        for row in forecast_data
    ]
    prices = [current_price] + [
        row["price"]
        for row in forecast_data
    ]

    figure = Figure(figsize=(9, 4.5), constrained_layout=True)
    axis = figure.subplots()
    positions = range(len(labels))

    axis.plot(
        positions,
        prices,
        color="#2563eb",
        marker="o",
        linewidth=2,
        label="Price"
    )
    axis.scatter(
        [0],
        [current_price],
        color="#14b8a6",
        s=70,
        zorder=3,
        label="Current price"
    )
    axis.set_xticks(
        list(positions),
        labels,
        rotation=30,
        ha="right"
    )
    axis.set_title(f"{symbol} Price Forecast")
    axis.set_xlabel("Date")
    axis.set_ylabel("Price (INR)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()

    image = BytesIO()
    FigureCanvasAgg(figure).print_png(image)
    encoded_image = base64.b64encode(
        image.getvalue()
    ).decode("ascii")
    return f"data:image/png;base64,{encoded_image}"


# =====================================================
# FORECAST ENGINE
# =====================================================

def generate_forecast(
        symbol,
        model_name,
        close_history,
        forecast_days=5):

    model = get_model(model_name)

    feature_columns = get_feature_columns()

    features = make_features(
        close_history
    )

    latest = (
        features
        .tail(1)
        .copy()
    )

    if "close" in latest.columns:

        latest = latest.drop(
            columns=["close"]
        )

    if "ticker_encoded" in feature_columns:
        latest["ticker_encoded"] = encode_symbol(symbol)

    latest = latest[
        feature_columns
    ]

    current_price = float(
        close_history.iloc[-1]
    )

    model_prediction = float(
        model.predict(latest)[0]
    )

    print("\n====================")
    print("SYMBOL:", symbol)
    print("MODEL:", model_name)
    print("CURRENT PRICE:", current_price)
    print("RAW MODEL PREDICTION:", model_prediction)
    print("====================\n")

    # ---------------------------------------
    # Safety stabilisation
    # ---------------------------------------

    if model_prediction <= 0:

        model_prediction = current_price

    max_move = current_price * 0.15

    upper_limit = (
        current_price + max_move
    )

    lower_limit = (
        current_price - max_move
    )

    model_prediction = min(
        model_prediction,
        upper_limit
    )

    model_prediction = max(
        model_prediction,
        lower_limit
    )

    forecasts = []

    step_change = (

        model_prediction
        -
        current_price

    ) / forecast_days

    for day in range(
            1,
            forecast_days + 1):

        price = (

            current_price

            +

            step_change * day

        )

        forecasts.append(

            round(
                float(price),
                2
            )

        )

    return forecasts


# =====================================================
# HOME
# =====================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =====================================================
# PREDICT
# =====================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        symbol = (

            request.form["symbol"]

            .upper()

            .strip()

        )

        forecast_days = int(
            request.form.get(
                "forecast_days",
                5
            )
        )

        today = (

            pd.Timestamp.today()

            .strftime(
                "%d-%m-%Y"
            )

        )

        data = load_stock_data(

            symbol,

            "01-01-2021",

            today

        )

        if data.empty:

            raise ValueError(
                f"No NSE data found "
                f"for {symbol}"
            )

        current_price = float(
            data["close"].iloc[-1]
        )

        print(
            f"\nCurrent NSE Price "
            f"for {symbol}: "
            f"{current_price}"
        )

        # --------------------------------
        # Decision Agent
        # --------------------------------

        selected_model = (
            decision_agent
            .choose_model(symbol)
        )

        performance_metrics = get_model_metrics(
            selected_model
        )
        model_rankings = get_model_rankings()

        print(
            "Selected Model:",
            selected_model
        )

        # --------------------------------
        # Forecast
        # --------------------------------

        forecast_prices = (

            generate_forecast(

                symbol,

                selected_model,

                data["close"],

                forecast_days

            )

        )

        # --------------------------------
        # Future Dates
        # --------------------------------

        future_dates = pd.bdate_range(

            start=(

                data.index[-1]

                +

                pd.offsets.BDay(1)

            ),

            periods=forecast_days

        )

        forecast_data = []

        for dt, price in zip(

                future_dates,

                forecast_prices):

            forecast_data.append({

                "date":

                    dt.strftime(
                        "%d-%m-%Y"
                    ),

                "price":

                    round(
                        float(price),
                        2
                    )

            })

        chart_image = create_forecast_chart(
            symbol,
            current_price,
            forecast_data
        )

        return render_template(

            "predict.html",

            symbol=symbol,

            selected_model=selected_model,

            current_price=round(
                current_price,
                2
            ),

            forecast_days=forecast_days,

            forecast_data=forecast_data,

            chart_image=chart_image,

            performance_metrics=performance_metrics,

            model_rankings=model_rankings

        )

    except Exception as e:

        print(
            "\nAPPLICATION ERROR:",
            str(e)
        )

        return render_template(

            "predict.html",

            symbol="--",

            selected_model="--",

            current_price=0,

            forecast_days=0,

            forecast_data=[],

            chart_image=None,

            performance_metrics=None,

            model_rankings=[],

            error=str(e)

        )


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )