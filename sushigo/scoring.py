from typing import List
from .cards import CardType


# tempura, sashimi, dumplings 

def score_tempura(tb: List[CardType]) -> int:
    count = sum(1 for c in tb if c == CardType.TEMPURA)
    return (count // 2) * 5


def score_sashimi(tb: List[CardType]) -> int:
    count = sum(1 for c in tb if c == CardType.SASHIMI)
    return (count // 3) * 10


def score_dumplings(tb: List[CardType]) -> int:
    count = sum(1 for c in tb if c == CardType.DUMPLING)
    if count == 0:
        return 0
    elif count == 1:
        return 1
    elif count == 2:
        return 3
    elif count == 3:
        return 6
    elif count == 4:
        return 10
    return 15


# -nigiri, wasabi 

def score_nigiri_and_wasabi(tb: List[CardType]) -> int:
    score = 0
    wasabi_available = 0

    for card in tb:
        if card == CardType.WASABI:
            wasabi_available += 1

        elif card in (CardType.EGG_NIGIRI, CardType.SALMON_NIGIRI, CardType.SQUID_NIGIRI):
            if card == CardType.EGG_NIGIRI:
                base = 1
            elif card == CardType.SALMON_NIGIRI:
                base = 2
            else:
                base = 3

            if wasabi_available > 0:
                score += base * 3
                wasabi_available -= 1
            else:
                score += base

    return score


# maki 

def maki_icons(card: CardType) -> int:
    if card == CardType.MAKI_1:
        return 1
    if card == CardType.MAKI_2:
        return 2
    if card == CardType.MAKI_3:
        return 3
    return 0


def score_maki(tableaus: List[List[CardType]]) -> List[int]:
    totals = [sum(maki_icons(c) for c in tb) for tb in tableaus]
    scores = [0] * len(tableaus)

    max_total = max(totals)
    if max_total == 0:
        return scores

    most_players = [p for p, t in enumerate(totals) if t == max_total]

    if len(most_players) == 1:
        winner = most_players[0]
        scores[winner] += 6

        remaining = [(p, t) for p, t in enumerate(totals) if p != winner and t > 0]
        if remaining:
            second_total = max(t for _, t in remaining)
            second_players = [p for p, t in remaining if t == second_total]
            share = 3 // len(second_players)
            for p in second_players:
                scores[p] += share

    else:
        share = 6 // len(most_players)
        for p in most_players:
            scores[p] += share

    return scores
