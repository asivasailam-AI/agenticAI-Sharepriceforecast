import json
from pathlib import Path

from services.predictor import (
    SUPPORTED_MODEL_NAMES,
    get_model_rankings
)


class DecisionAgent:

    def __init__(self):
        model_dir = Path(__file__).resolve().parents[1] / "models"
        self.leaderboard_path = model_dir / "leaderboard.json"

    def choose_model(self, symbol):
        symbol = symbol.upper().strip()
        rankings = get_model_rankings(symbol)
        for ranking in rankings:
            model_name = ranking["Model"]
            if model_name in SUPPORTED_MODEL_NAMES:
                return model_name

        with open(self.leaderboard_path, "r") as f:
            registry = json.load(f)

        model_name = registry.get(symbol)
        if model_name not in SUPPORTED_MODEL_NAMES:
            for ranking in get_model_rankings():
                candidate = ranking["Model"]
                if candidate in SUPPORTED_MODEL_NAMES:
                    return candidate

            model_name = registry.get("GLOBAL_MODEL")
        if model_name not in SUPPORTED_MODEL_NAMES:
            raise ValueError(
                f"No supported model is available for {symbol!r}. "
                "Train the models to generate a valid scorecard or leaderboard."
            )
        return model_name
