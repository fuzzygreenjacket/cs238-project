"""
Random Agent - Baseline agent that selects actions uniformly at random.

Used as a control/baseline for comparing more sophisticated strategies.
"""

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sushigo.engine import SushiGoGame
    from sushigo.actions import Action


class RandomAgent:
    """Agent that selects valid actions uniformly at random."""
    
    def __init__(self, seed=None):
        """
        Initialize random agent.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
    
    def select_action(self, game: 'SushiGoGame', player_id: int) -> 'Action':
        """
        Select a random valid action.
        
        Args:
            game: Current game state
            player_id: This agent's player ID
            
        Returns:
            A randomly chosen valid action
        """
        valid_actions = game.get_valid_actions(player_id)
        if not valid_actions:
            return None
        return self.rng.choice(valid_actions)