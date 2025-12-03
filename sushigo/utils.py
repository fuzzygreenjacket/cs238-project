import random
from typing import List
from .cards import CardType

# func to build the initial deck 
# deck is outputted as an array 
def build_deck() -> List[CardType]:
    deck = []
    deck += [CardType.TEMPURA] * 14
    deck += [CardType.SASHIMI] * 14
    deck += [CardType.DUMPLING] * 14

    deck += [CardType.MAKI_2] * 12
    deck += [CardType.MAKI_3] * 8
    deck += [CardType.MAKI_1] * 6

    deck += [CardType.SALMON_NIGIRI] * 10
    deck += [CardType.SQUID_NIGIRI] * 5
    deck += [CardType.EGG_NIGIRI] * 5

    deck += [CardType.PUDDING] * 10
    deck += [CardType.WASABI] * 6
    deck += [CardType.CHOPSTICKS] * 4

    assert len(deck) == 108
    return deck

# depending on the # of players, we allocate each player a hand 
def initial_hand_size(n_players: int) -> int:
    if n_players == 2: return 10
    if n_players == 3: return 9
    if n_players == 4: return 8
    if n_players == 5: return 7
    raise ValueError("Invalid player count.")
