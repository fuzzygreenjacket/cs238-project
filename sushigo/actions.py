from dataclasses import dataclass
from typing import Optional


# what is a player doing in this turn? 
# a player can either play one card fro their deck 
# if they have chopsticks, play two cards 


# ex use: Action(use_chopsticks=False, first_index=2) to play the card at idx 2 of the deck 
# Action(use_chopsticks=True, first_index=1, second_index=4) using chopsticks, play card at idx 1 and 4 

# in our engine, all players act simultaneously mimicking a real sushi go game 
# - Everyone looks at their hand privately.
# - Everyone secretly chooses 1 card.
# - Everyone reveals simultaneously.
# - THEN you update your beliefs / information

@dataclass(frozen=True)
class Action:
    """
    use chopticks: true if player has chopsticks and wants to use. allows player to play 2 cards 
    first_idx = card they want to play 
    second_idx = optional second card if chopsticks 
    """
    use_chopsticks: bool
    first_index: int
    second_index: Optional[int] = None
