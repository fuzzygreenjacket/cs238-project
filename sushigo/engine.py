import random
from typing import Dict, List
from .cards import CardType
from .actions import Action
from .game_state import PublicState
from .scoring import (
    score_tempura, score_sashimi, score_dumplings,
    score_nigiri_and_wasabi, score_maki
)
from .utils import build_deck, initial_hand_size


class SushiGoGame:
    NUM_ROUNDS = 3

    def __init__(self, num_players: int, seed=None):
        if num_players < 2 or num_players > 5:
            raise ValueError("Supports 2–5 players.")
        self.num_players = num_players
        self.rng = random.Random(seed)

        print("\n=== INITIALIZING SUSHI GO GAME ===")
        print(f"Number of players: {self.num_players}")
        print(f"Random seed: {seed}")

        self.reset()

    # reset + setup 

    def reset(self):
        print("\n=== RESETTING GAME ===")
        self.deck = build_deck()
        print(f"Built deck with {len(self.deck)} cards")

        self.rng.shuffle(self.deck)
        print(f"Deck shuffled. Top 5 cards: {[c.name for c in self.deck[:5]]}")

        self.round_index = 0

        self.hands = [[] for _ in range(self.num_players)]
        self.tableaus = [[] for _ in range(self.num_players)]
        self.puddings = [0] * self.num_players
        self.scores = [0] * self.num_players

        self._deal_round()

    def _deal_round(self):
        n = initial_hand_size(self.num_players)
        print(f"\n=== DEALING ROUND {self.round_index + 1} ===")
        print(f"Each player receives {n} cards")

        for p in range(self.num_players):
            self.hands[p] = [self.deck.pop() for _ in range(n)]
            print(f"Player {p} initial hand: {[c.name for c in self.hands[p]]}")

    # query helpers 
    def get_player_hand(self, p):
        return list(self.hands[p])

    def available_chopsticks(self, p):
        return sum(c == CardType.CHOPSTICKS for c in self.tableaus[p])

    def get_valid_actions(self, p):
        hand = self.hands[p]
        if not hand:
            return []

        actions = [Action(False, i) for i in range(len(hand))]

        if self.available_chopsticks(p) > 0 and len(hand) >= 2:
            for i in range(len(hand)):
                for j in range(i + 1, len(hand)):
                    actions.append(Action(True, i, j))

        return actions
        
    def is_game_over(self) -> bool:
        """Returns True when all 3 rounds have finished."""
        return self.round_index >= self.NUM_ROUNDS


    def clone_public_state(self):
        return PublicState(
            num_players=self.num_players,
            round_index=self.round_index,
            hands_sizes=[len(h) for h in self.hands],
            tableaus=[list(tb) for tb in self.tableaus],
            puddings=list(self.puddings),
            scores=list(self.scores)
        )

    # turn mechanics 
    def play_turn(self, actions: Dict[int, Action]):
        print("\n==================== NEW TURN ====================")

        old_hands = [list(h) for h in self.hands]
        new_hands = [[] for _ in range(self.num_players)]

        # process layer action 
        for p in range(self.num_players):
            hand = old_hands[p]
            if not hand:
                print(f"Player {p} has no cards to play.")
                continue

            action = actions[p]
            print(f"\nPlayer {p} starting hand: {[c.name for c in hand]}")
            print(f"Player {p} chose action: {action}")

            if not action.use_chopsticks:
                chosen = [action.first_index]
            else:
                chosen = sorted([action.first_index, action.second_index])
                print(f"Player {p} USES CHOPSTICKS!")

            chosen_cards = [hand[i] for i in chosen]
            print(f"Player {p} played: {[c.name for c in chosen_cards]}")

            remaining = [c for i, c in enumerate(hand) if i not in chosen]

            if action.use_chopsticks:
                self._consume_chopsticks(p, remaining)
                print("Chopsticks put back into remaining hand for passing")

            self.tableaus[p].extend(chosen_cards)
            print(f"Player {p} tableau now: {[c.name for c in self.tableaus[p]]}")

            new_hands[p] = remaining
            print(f"Player {p} hand after play: {[c.name for c in remaining]}")

        # pass hands to the left 
        print("\n--- PASSING HANDS TO NEXT PLAYER (left) ---")
        passed = [[] for _ in range(self.num_players)]

        for p in range(self.num_players):
            next_p = (p + 1) % self.num_players
            passed[next_p] = new_hands[p]
            print(f"Hand from Player {p} -> Player {next_p}")

        self.hands = passed

        for p in range(self.num_players):
            print(f"Player {p} receives new hand: {[c.name for c in self.hands[p]]}")

        # check if round is over 
        if all(len(h) == 0 for h in self.hands):
            print("\n=========== ROUND COMPLETE — SCORING ===========")
            self._score_round()

            print(f"Scores so far: {self.scores}")
            print(f"Puddings so far: {self.puddings}")

            self.round_index += 1

            if self.round_index < self.NUM_ROUNDS:
                self._deal_round()
            else:
                print("\n=========== ALL ROUNDS COMPLETE — FINAL PUDDING SCORING ===========")
                self._score_puddings()
                print(f"\nFINAL SCORES: {self.scores}")
                print(f"FINAL PUDDINGS: {self.puddings}")

    # chopsticks handling 

    def _consume_chopsticks(self, p, remaining_hand):
        tb = self.tableaus[p]
        for i, card in enumerate(tb):
            if card == CardType.CHOPSTICKS:
                print(f"Player {p} consumes a Chopsticks from tableau")
                del tb[i]
                remaining_hand.append(CardType.CHOPSTICKS)
                return
        raise RuntimeError("Chopsticks use mismatch.")

    #  round scoring 
    def _score_round(self):
        print("\n===== SCORING ROUND =====")
        round_tb = [[] for _ in range(self.num_players)]

        for p in range(self.num_players):
            print(f"\n-- Extracting puddings + round cards for Player {p} --")
            for c in self.tableaus[p]:
                if c == CardType.PUDDING:
                    self.puddings[p] += 1
                    print(f"Player {p} gains a Pudding (total: {self.puddings[p]})")
                else:
                    round_tb[p].append(c)
            self.tableaus[p] = []

        print("\nScoring Maki rolls...")
        maki_scores = score_maki(round_tb)
        print(f"Maki scores: {maki_scores}")

        for p in range(self.num_players):
            tb = round_tb[p]
            tempura = score_tempura(tb)
            sashimi = score_sashimi(tb)
            dumplings = score_dumplings(tb)
            nigiri = score_nigiri_and_wasabi(tb)

            total = tempura + sashimi + dumplings + nigiri + maki_scores[p]

            print(f"\nPlayer {p} round details:")
            print(f"  Tempura:   {tempura}")
            print(f"  Sashimi:   {sashimi}")
            print(f"  Dumplings: {dumplings}")
            print(f"  Nigiri/W:  {nigiri}")
            print(f"  Maki:      {maki_scores[p]}")
            print(f"  TOTAL:     {total}")

            self.scores[p] += total

    # final pudding scoring 
    def _score_puddings(self):
        print("\n===== FINAL PUDDING SCORING =====")
        counts = self.puddings

        print(f"Pudding counts: {counts}")
        if all(c == counts[0] for c in counts):
            print("All players tied in puddings — no scoring changes.")
            return

        max_c = max(counts)
        min_c = min(counts)

        most = [p for p, c in enumerate(counts) if c == max_c]
        share = 6 // len(most)
        print(f"Players with MOST puddings: {most}, each gets +{share}")
        for p in most:
            self.scores[p] += share

        if self.num_players > 2:
            least = [p for p, c in enumerate(counts) if c == min_c]
            penalty = 6 // len(least)
            print(f"Players with LEAST puddings: {least}, each loses -{penalty}")
            for p in least:
                self.scores[p] -= penalty
