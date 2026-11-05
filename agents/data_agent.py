import pandas as pd
from nselib import capital_market as cm


def load_stock_data(
        symbol,
        from_date,
        to_date):

    raw = cm.price_volume_data(
        symbol=symbol,
        from_date=from_date,
        to_date=to_date
    )

    raw.columns = [
        str(x).lower().strip()
        for x in raw.columns
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