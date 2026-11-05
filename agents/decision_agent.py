import json
from pathlib import Path


class DecisionAgent:

    def __init__(self):
        model_dir = Path(__file__).resolve().parents[1] / "models"
        with open(model_dir / "leaderboard.json", "r") as f:
            self.registry = json.load(f)

    def choose_model(self, symbol):
        symbol = symbol.upper().strip()
        return self.registry.get(
            symbol,
            self.registry["GLOBAL_MODEL"]
        )
