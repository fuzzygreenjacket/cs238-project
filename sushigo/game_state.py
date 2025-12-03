from dataclasses import dataclass
from typing import List
from .cards import CardType


@dataclass(frozen=True)
class PublicState:
    num_players: int
    round_index: int
    hands_sizes: List[int]
    tableaus: List[List[CardType]]
    puddings: List[int]
    scores: List[int]
