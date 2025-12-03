
# random agent 

import random
from .engine import SushiGoGame
from .actions import Action

class RandomAgent:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def select_action(self, game: SushiGoGame, player_id: int) -> Action:
        valid = game.get_valid_actions(player_id)
        return self.rng.choice(valid)
