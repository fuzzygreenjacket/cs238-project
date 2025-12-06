"""
Adaptive Meta-Strategy Agent

This agent does not follow a single fixed style. Instead, it dynamically
chooses among several sub-strategies based on:
- Game state (round, cards remaining)
- Score differential (leading, trailing, tied)
- Round phase (early, mid, late)

It adapts its personality as the game evolves, switching between aggressive,
defensive, safe, catch-up, and opportunistic modes.
"""

import random
from typing import TYPE_CHECKING, Dict, Any
from collections import Counter

if TYPE_CHECKING:
    from sushigo.engine import SushiGoGame
    from sushigo.actions import Action
    from sushigo.cards import CardType


class MetaStrategyAgent:
    """Adaptive agent that switches strategies based on game situation."""
    
    def __init__(self, seed=None):
        """
        Initialize Meta-Strategy agent.
        
        Args:
            seed: Random seed for tie-breaking
        """
        self.rng = random.Random(seed)
        # Track opponent patterns (could be used for future enhancements)
        self.opponent_preferences = {}
        
    def select_action(self, game: 'SushiGoGame', player_id: int) -> 'Action':
        """
        Dynamically choose strategy and select best action.
        
        Args:
            game: Current game state
            player_id: This agent's player ID
            
        Returns:
            Best action for the dynamically chosen strategy
        """
        hand = game.get_player_hand(player_id)
        if not hand:
            return None
            
        public_state = game.clone_public_state()
        
        # Analyze current situation
        situation = self._analyze_situation(player_id, public_state)
        
        # Choose appropriate strategy
        strategy = self._choose_meta_strategy(
            situation, hand, public_state.tableaus[player_id]
        )
        
        # Get valid actions
        valid_actions = game.get_valid_actions(player_id)
        if not valid_actions:
            return None
        
        # Select best action for chosen strategy
        best_actions = []
        best_score = float('-inf')
        
        for action in valid_actions:
            score = self._score_action(
                action, hand, public_state.tableaus[player_id],
                strategy, situation
            )
            
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)
        
        return self.rng.choice(best_actions)
    
    def _analyze_situation(self, player_id: int, public_state) -> Dict[str, Any]:
        """
        Analyze current game state to inform strategy selection.
        
        Args:
            player_id: This agent's player ID
            public_state: Current public game state
            
        Returns:
            Dictionary containing situation analysis
        """
        my_score = public_state.scores[player_id]
        my_pudding = public_state.puddings[player_id]
        
        # Calculate relative position
        scores = public_state.scores
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        
        score_diff = my_score - avg_score
        is_leading = my_score == max_score
        is_behind = my_score < avg_score - 5
        
        # Determine round phase (based on hand sizes)
        hand_sizes = public_state.hands_sizes
        avg_hand_size = sum(hand_sizes) / len(hand_sizes) if hand_sizes else 0
        
        is_early_round = avg_hand_size > 6
        is_mid_round = 3 <= avg_hand_size <= 6
        is_late_round = avg_hand_size < 3
        
        is_final_round = public_state.round_index == 2
        
        return {
            'my_score': my_score,
            'my_pudding': my_pudding,
            'score_diff': score_diff,
            'is_leading': is_leading,
            'is_behind': is_behind,
            'is_early_round': is_early_round,
            'is_mid_round': is_mid_round,
            'is_late_round': is_late_round,
            'is_final_round': is_final_round,
            'round_index': public_state.round_index,
            'avg_hand_size': avg_hand_size
        }
    
    def _choose_meta_strategy(self, situation: Dict[str, Any], hand, my_tableau) -> str:
        """
        Select strategy based on situation analysis.
        
        Strategies:
        - 'aggressive': High-risk, high-reward (when behind)
        - 'safe': Lock in guaranteed points (late round or leading)
        - 'catch_up': Balanced risk-reward (when behind)
        - 'defensive': Maintain lead, block opponents (when leading)
        - 'opportunistic': Take best opportunities (default/early)
        - 'pudding_focus': Prioritize puddings (final round)
        
        Args:
            situation: Analyzed game situation
            hand: Current hand
            my_tableau: Current tableau
            
        Returns:
            Strategy name
        """
        from sushigo.cards import CardType
        
        tableau_counts = Counter(my_tableau)
        hand_counts = Counter(hand)
        
        # CATCH-UP mode: Behind and need high-variance plays
        if situation['is_behind'] and situation['is_mid_round']:
            # Look for high-risk, high-reward opportunities
            if (tableau_counts[CardType.SASHIMI] >= 1 and 
                hand_counts[CardType.SASHIMI] >= 1):
                return 'aggressive'  # Go for sashimi triple
            if hand_counts[CardType.MAKI_3] >= 1:
                return 'aggressive'  # Push for maki majority
            return 'catch_up'
        
        # DEFENSIVE mode: Leading and want to maintain lead safely
        if situation['is_leading'] and situation['is_late_round']:
            return 'defensive'
        
        # SAFE mode: Late in round, lock in guaranteed points
        if situation['is_late_round']:
            # Check for near-complete sets
            if tableau_counts[CardType.TEMPURA] % 2 == 1:
                return 'safe'  # Complete tempura pairs
            if tableau_counts[CardType.SASHIMI] == 2:
                return 'safe'  # Complete sashimi triple
            return 'safe'
        
        # PUDDING FOCUS: Final round and trailing in puddings
        if situation['is_final_round']:
            pudding_diff = situation['my_pudding']
            if pudding_diff < 2:  # Need more puddings
                return 'pudding_focus'
        
        # OPPORTUNISTIC mode: Early round, look for best opportunities
        if situation['is_early_round']:
            # Check for strong starting opportunities
            if (hand_counts[CardType.WASABI] >= 1 and 
                hand_counts[CardType.SQUID_NIGIRI] >= 1):
                return 'opportunistic'
            if hand_counts[CardType.SASHIMI] >= 2:
                return 'opportunistic'
            if hand_counts[CardType.TEMPURA] >= 2:
                return 'opportunistic'
        
        # Default: balanced play
        return 'opportunistic'
    
    def _score_action(self, action, hand, my_tableau, strategy: str, 
                     situation: Dict[str, Any]) -> float:
        """
        Score action based on chosen meta-strategy.
        
        Args:
            action: Action to evaluate
            hand: Current hand
            my_tableau: Current tableau
            strategy: Chosen strategy name
            situation: Analyzed situation
            
        Returns:
            Score for this action
        """
        from sushigo.cards import CardType
        
        # Get cards from action
        if not action.use_chopsticks:
            cards = [hand[action.first_index]]
        else:
            cards = [hand[action.first_index], hand[action.second_index]]
        
        tableau_counts = Counter(my_tableau)
        total_score = 0
        
        for card in cards:
            base_value = self._get_base_value(card, tableau_counts)
            
            # Apply strategy modifiers
            if strategy == 'aggressive':
                # Heavily favor high-value sets, risk it all
                if card == CardType.SASHIMI:
                    base_value *= 2.5
                elif card == CardType.MAKI_3:
                    base_value *= 2.0
                elif card in [CardType.SQUID_NIGIRI, CardType.WASABI]:
                    base_value *= 1.8
                else:
                    base_value *= 0.5  # Deprioritize safe plays
                    
            elif strategy == 'safe':
                # Favor completing existing sets
                if card == CardType.TEMPURA and tableau_counts[CardType.TEMPURA] % 2 == 1:
                    base_value *= 3.0  # Complete pair!
                elif card == CardType.SASHIMI and tableau_counts[CardType.SASHIMI] == 2:
                    base_value *= 3.5  # Complete triple!
                elif card == CardType.DUMPLING:
                    base_value *= 1.5  # Safe steady points
                else:
                    base_value *= 0.8  # Slightly deprioritize new ventures
                    
            elif strategy == 'catch_up':
                # Balance between risk and reward
                if card == CardType.SASHIMI:
                    base_value *= 1.8
                elif card in [CardType.MAKI_2, CardType.MAKI_3]:
                    base_value *= 1.6
                elif card == CardType.DUMPLING:
                    base_value *= 1.3
                    
            elif strategy == 'defensive':
                # Take what opponents might want, block them
                if card == CardType.MAKI_3:
                    base_value *= 2.0  # Block maki majority
                elif card == CardType.SQUID_NIGIRI:
                    base_value *= 1.5  # Deny best nigiri
                elif card == CardType.DUMPLING:
                    base_value *= 1.4  # Steady points
                    
            elif strategy == 'pudding_focus':
                if card == CardType.PUDDING:
                    base_value = 100  # Top priority!
                else:
                    base_value *= 0.5  # Everything else secondary
                    
            elif strategy == 'opportunistic':
                # Look for synergies and strong starts
                if card == CardType.WASABI:
                    base_value *= 1.8
                elif card == CardType.SQUID_NIGIRI and tableau_counts[CardType.WASABI] > 0:
                    base_value *= 2.5
                elif card == CardType.SASHIMI and tableau_counts[CardType.SASHIMI] >= 1:
                    base_value *= 2.0
                elif card == CardType.TEMPURA and tableau_counts[CardType.TEMPURA] >= 1:
                    base_value *= 1.8
            
            total_score += base_value
        
        # Chopsticks bonus varies by strategy
        if action.use_chopsticks:
            chopsticks_bonus = {
                'aggressive': 40,
                'safe': 20,
                'catch_up': 35,
                'defensive': 15,
                'pudding_focus': 25,
                'opportunistic': 30
            }.get(strategy, 25)
            total_score += chopsticks_bonus
        
        return total_score
    
    def _get_base_value(self, card, tableau_counts) -> float:
        """
        Get base expected value of a card.
        
        Args:
            card: Card type
            tableau_counts: Counter of cards in tableau
            
        Returns:
            Base expected value
        """
        from sushigo.cards import CardType
        
        if card == CardType.TEMPURA:
            return 5.0 if tableau_counts[CardType.TEMPURA] % 2 == 1 else 2.5
            
        elif card == CardType.SASHIMI:
            count = tableau_counts[CardType.SASHIMI]
            if count == 2:
                return 10.0
            elif count == 1:
                return 5.0
            else:
                return 2.0
                
        elif card == CardType.DUMPLING:
            return max(5 - tableau_counts[CardType.DUMPLING] * 0.5, 2)
            
        elif card == CardType.SQUID_NIGIRI:
            return 9.0 if tableau_counts[CardType.WASABI] > 0 else 3.0
            
        elif card == CardType.SALMON_NIGIRI:
            return 6.0 if tableau_counts[CardType.WASABI] > 0 else 2.0
            
        elif card == CardType.EGG_NIGIRI:
            return 3.0 if tableau_counts[CardType.WASABI] > 0 else 1.0
            
        elif card == CardType.WASABI:
            return 4.0
            
        elif card == CardType.MAKI_3:
            return 5.0
        elif card == CardType.MAKI_2:
            return 3.5
        elif card == CardType.MAKI_1:
            return 2.0
            
        elif card == CardType.PUDDING:
            return 3.0
            
        elif card == CardType.CHOPSTICKS:
            return 2.5
            
        else:
            return 1.0