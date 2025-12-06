"""
Q-Learning Agent

This agent learns optimal play through experience by maintaining Q-values
Q(s,a) that estimate the expected future score for taking action a in state s.

Strategy:
- Uses epsilon-greedy exploration (random actions with probability epsilon)
- Updates Q-values after each round using temporal difference learning
- State representation: compact feature vector (card counts, round info)
- Learns over many games without needing a model of the environment

The agent improves over time as it learns patterns like:
"If I have one Tempura, taking another is usually good"
"Wasabi is more valuable early in the round"
"""

import random
import pickle
from typing import TYPE_CHECKING, Dict, Tuple
from collections import Counter, defaultdict

if TYPE_CHECKING:
    from sushigo.engine import SushiGoGame
    from sushigo.actions import Action


class QLearningAgent:
    """Q-Learning agent that learns from experience."""

    def __init__(self, seed=None, learning_rate=0.1, discount_factor=0.95,
                 epsilon=0.2, epsilon_decay=0.9995, min_epsilon=0.05):
        """
        Initialize Q-Learning agent.
        """
        self.rng = random.Random(seed)

        # Q-learning parameters
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Q-table: maps (state, action_type) -> expected value
        self.q_table = defaultdict(float)

        # Track episode history for learning: list of (state, action_type)
        self.episode_history = []
        self.previous_score = 0

        # For debugging / diagnostics
        self.recent_rewards = []  # collect round rewards (shaped) for statistics

        # Statistics
        self.games_played = 0
        self.total_reward = 0

    # ---------- Action selection ----------
    def select_action(self, game: 'SushiGoGame', player_id: int) -> 'Action':
        """
        Select action using epsilon-greedy policy.
        """
        hand = game.get_player_hand(player_id)
        if not hand:
            return None

        valid_actions = game.get_valid_actions(player_id)
        if not valid_actions:
            return None

        public_state = game.clone_public_state()
        state = self._extract_state_features(hand, public_state, player_id)

        # Epsilon-greedy: explore with probability epsilon
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(valid_actions)
        else:
            action = self._select_best_action(state, valid_actions, hand)

        # Store this step for later learning (reward filled later)
        action_type = self._action_to_type(action, hand)
        self.episode_history.append((state, action_type))

        return action

    # ---------- Learning hooks ----------
    def update_after_round(self, game: 'SushiGoGame', player_id: int):
        """
        Update Q-values after a round completes.

        Reward shaping added: round score gain + small set-completion bonus.
        The shaped round reward is distributed across the actions taken in that round
        with light recency weighting (later actions get slightly more credit).
        """
        public_state = game.clone_public_state()
        current_score = public_state.scores[player_id]

        # Basic shaped reward: score delta this round
        round_reward = current_score - self.previous_score

        # Detect simple "set completion" by comparing counts of tableau cards before/after
        # We don't have previous tableau stored per-round here, so approximate:
        # If this round we gained >=1 Tempura or Sashimi or completed dumpling bucket, give small bonus.
        # (Assumes public_state.tableaus exist and are lists of cards.)
        try:
            my_tableau = public_state.tableaus[player_id]
            # a small heuristic bonus if tableau grew noticeably this round
            # (this is cheap and helps reward useful collection actions)
            if len(my_tableau) > 0:
                set_bonus = 0
                # small bonus if we increased total tableau size relative to previous_score trick
                # (not perfect, but gives more frequent signal)
                set_bonus = 1 if round_reward > 0 else 0
            else:
                set_bonus = 0
        except Exception:
            set_bonus = 0

        shaped_reward = round_reward + set_bonus

        # Diagnostics
        self.recent_rewards.append(shaped_reward)

        # Update running totals
        self.previous_score = current_score
        self.total_reward += shaped_reward

        # Update Q-values for actions taken this round
        self._update_q_values(shaped_reward)

        # Clear episode history for next round
        self.episode_history = []

    def update_after_game(self):
        """
        Called after a complete game finishes.
        Updates statistics and decays epsilon.
        """
        self.games_played += 1

        # Decay exploration rate
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        # Reset for next game
        self.previous_score = 0
        self.episode_history = []

    # ---------- State / Action helpers ----------
    def _extract_state_features(self, hand, public_state, player_id) -> Tuple:
        """
        Extract a compact state representation for Q-learning.

        Keep similar to previous but stable and hashable.
        """
        from sushigo.cards import CardType

        # Count cards in hand
        hand_counts = Counter(hand)

        # Count cards in tableau
        my_tableau = public_state.tableaus[player_id]
        tableau_counts = Counter(my_tableau)

        # Calculate relative score
        my_score = public_state.scores[player_id]
        avg_score = sum(public_state.scores) / len(public_state.scores)
        score_diff = int((my_score - avg_score) / 5)  # Discretize to buckets of 5

        features = (
            min(hand_counts[CardType.TEMPURA], 3),
            min(hand_counts[CardType.SASHIMI], 3),
            min(hand_counts[CardType.DUMPLING], 3),
            min(hand_counts[CardType.MAKI_1] + hand_counts[CardType.MAKI_2] +
                hand_counts[CardType.MAKI_3], 4),
            min(hand_counts[CardType.SQUID_NIGIRI], 2),
            min(hand_counts[CardType.SALMON_NIGIRI], 2),
            min(hand_counts[CardType.EGG_NIGIRI], 2),
            min(hand_counts[CardType.WASABI], 2),
            min(hand_counts[CardType.CHOPSTICKS], 2),
            min(hand_counts[CardType.PUDDING], 2),

            min(tableau_counts[CardType.TEMPURA], 4),
            min(tableau_counts[CardType.SASHIMI], 4),
            min(tableau_counts[CardType.DUMPLING], 5),
            min(tableau_counts[CardType.WASABI], 2),

            public_state.round_index,
            min(len(hand), 10),
            max(-3, min(3, score_diff)),
            min(public_state.puddings[player_id], 4)
        )

        return features

    def _action_to_type(self, action, hand) -> Tuple:
        """
        Convert action to a hashable type representation.
        """
        from sushigo.cards import CardType

        if not action.use_chopsticks:
            card = hand[action.first_index]
            return ('single', card.name)
        else:
            card1 = hand[action.first_index]
            card2 = hand[action.second_index]
            cards = tuple(sorted([card1.name, card2.name]))
            return ('chopsticks', cards[0], cards[1])

    def _select_best_action(self, state, valid_actions, hand):
        """
        Select action with highest Q-value (greedy).
        """
        best_value = float('-inf')
        best_actions = []

        for action in valid_actions:
            action_type = self._action_to_type(action, hand)
            q_value = self.q_table[(state, action_type)]

            if q_value > best_value:
                best_value = q_value
                best_actions = [action]
            elif q_value == best_value:
                best_actions.append(action)

        # If all Q-values are zero (cold start), break ties randomly among valid_actions
        if not best_actions:
            return self.rng.choice(valid_actions)

        return self.rng.choice(best_actions)

    # ---------- Q-value update ----------
    def _update_q_values(self, shaped_round_reward):
        """
        Update Q-values for actions in the episode. We distribute shaped_round_reward
        across the actions taken this round, weighting later actions slightly more.
        This helps with recency credit assignment without a full n-step implementation.
        """
        if not self.episode_history:
            return

        n = len(self.episode_history)
        # Recency weights: linear from 1..n, normalized
        weights = [i + 1 for i in range(n)]
        wsum = sum(weights)
        # reward per action with weights
        for (state, action_type), w in zip(self.episode_history, weights):
            reward_share = shaped_round_reward * (w / wsum)

            current_q = self.q_table[(state, action_type)]
            # Standard incremental update toward observed reward share
            new_q = current_q + self.alpha * (reward_share + 0.0 - current_q)
            self.q_table[(state, action_type)] = new_q

    # ---------- Save / Load ----------
    def save_q_table(self, filepath):
        """Save Q-table to file for later use."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'games_played': self.games_played,
                'epsilon': self.epsilon
            }, f)
        print(f"Q-table saved to {filepath}")

    def load_q_table(self, filepath):
        """Load Q-table from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = defaultdict(float, data['q_table'])
            self.games_played = data.get('games_played', 0)
            self.epsilon = data.get('epsilon', self.epsilon)
        print(f"Q-table loaded from {filepath}")
        print(f"  Games played: {self.games_played}")
        print(f"  Current epsilon: {self.epsilon:.4f}")
        print(f"  Q-table size: {len(self.q_table)} state-action pairs")

    # ---------- Utilities ----------
    def set_eval_mode(self, epsilon=0.0):
        """
        Temporary helper to set exploration rate for evaluation.
        Use this before running an evaluation (and restore epsilon afterwards).
        """
        self.epsilon = epsilon

    def get_stats(self):
        """Return training statistics."""
        return {
            'games_played': self.games_played,
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'total_reward': self.total_reward,
            'recent_reward_mean': (sum(self.recent_rewards) / len(self.recent_rewards)) if self.recent_rewards else 0.0
        }
