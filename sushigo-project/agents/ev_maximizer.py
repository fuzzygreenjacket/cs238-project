"""
Expected-Value Maximizer Agent

This agent maintains beliefs about remaining cards and picks the action
with the highest expected contribution to its final score.

Strategy:
- Tracks all visible cards (own hand + all tableaus)
- Calculates expected value for each possible action
- Selects the action with maximum EV
- Does not explicitly model opponent behavior
"""

import random
from typing import TYPE_CHECKING
from collections import Counter

if TYPE_CHECKING:
    from sushigo.engine import SushiGoGame
    from sushigo.actions import Action
    from sushigo.cards import CardType


class EVMaximizerAgent:
    """Agent that maximizes expected value of each action."""
    
    def __init__(self, seed=None):
        """
        Initialize EV Maximizer agent.
        
        Args:
            seed: Random seed for tie-breaking
        """
        self.rng = random.Random(seed)
        self.seen_cards = Counter()
        
    def select_action(self, game: 'SushiGoGame', player_id: int) -> 'Action':
        """
        Select action that maximizes expected value.
        
        Args:
            game: Current game state
            player_id: This agent's player ID
            
        Returns:
            Action with highest expected value
        """
        hand = game.get_player_hand(player_id)
        if not hand:
            return None
            
        public_state = game.clone_public_state()
        
        # Update belief about remaining cards
        self._update_beliefs(hand, public_state)
        
        # Get valid actions
        valid_actions = game.get_valid_actions(player_id)
        if not valid_actions:
            return None
            
        # Evaluate each action
        best_actions = []
        best_ev = float('-inf')
        
        for action in valid_actions:
            ev = self._evaluate_action(
                action, 
                hand, 
                public_state.tableaus[player_id],
                public_state.round_index
            )
            
            if ev > best_ev:
                best_ev = ev
                best_actions = [action]
            elif ev == best_ev:
                best_actions.append(action)
        
        # Random tie-breaking
        return self.rng.choice(best_actions)
    
    def _update_beliefs(self, hand, public_state):
        """Update belief about card distribution based on visible info."""
        self.seen_cards.clear()
        
        # Count cards in own hand
        for card in hand:
            self.seen_cards[card] += 1
            
        # Count cards in all tableaus
        for tableau in public_state.tableaus:
            for card in tableau:
                self.seen_cards[card] += 1
    
    def _evaluate_action(self, action, hand, my_tableau, round_index):
        """
        Calculate expected value of taking this action.
        
        Args:
            action: Action to evaluate
            hand: Current hand
            my_tableau: Current tableau
            round_index: Current round (0-2)
            
        Returns:
            Expected value score for this action
        """
        # Determine which card(s) we're evaluating
        if not action.use_chopsticks:
            cards = [hand[action.first_index]]
        else:
            cards = [hand[action.first_index], hand[action.second_index]]
        
        total_ev = 0
        
        for card in cards:
            ev = self._card_expected_value(card, my_tableau, round_index)
            total_ev += ev
            
        # Bonus for using chopsticks (card advantage)
        if action.use_chopsticks:
            total_ev += 1.0
            
        return total_ev
    
    def _card_expected_value(self, card, my_tableau, round_index):
        """
        Estimate expected value contribution of a single card.
        
        Args:
            card: Card type to evaluate
            my_tableau: Current tableau
            round_index: Current round (0-2)
            
        Returns:
            Expected value for this card
        """
        from sushigo.cards import CardType
        
        tableau_counts = Counter(my_tableau)
        
        if card == CardType.TEMPURA:
            # Worth 5 if we complete a pair, 0 otherwise
            tempura_count = tableau_counts[CardType.TEMPURA]
            if tempura_count == 1:
                return 5.0  # Complete the pair!
            elif tempura_count == 0:
                return 2.5  # ~50% chance of completing
            else:
                return 0.5  # Already have complete sets
                
        elif card == CardType.SASHIMI:
            # Worth 10 for sets of 3
            sashimi_count = tableau_counts[CardType.SASHIMI]
            if sashimi_count == 2:
                return 10.0  # Complete the triple!
            elif sashimi_count == 1:
                return 3.3  # ~33% chance of completing
            elif sashimi_count == 0:
                return 2.0  # Need 2 more
            else:
                return 0.5  # Already have sets
                
        elif card == CardType.DUMPLING:
            # Dumplings score: 1,3,6,10,15 for 1-5+
            # Marginal values: +1, +2, +3, +4, +5
            dumpling_count = tableau_counts[CardType.DUMPLING]
            marginal_values = [1, 2, 3, 4, 5, 5, 5]
            return marginal_values[min(dumpling_count, 6)]
            
        elif card in [CardType.SQUID_NIGIRI, CardType.SALMON_NIGIRI, CardType.EGG_NIGIRI]:
            # Check if we have wasabi available
            wasabi_count = tableau_counts[CardType.WASABI]
            nigiri_base = {
                CardType.SQUID_NIGIRI: 3,
                CardType.SALMON_NIGIRI: 2,
                CardType.EGG_NIGIRI: 1
            }[card]
            
            # If we have wasabi, triple the value
            if wasabi_count > 0:
                return nigiri_base * 3
            else:
                return nigiri_base
                
        elif card == CardType.WASABI:
            # Worth 0 alone, but enables tripling future nigiri
            # Expected value depends on seeing good nigiri
            return 4.0  # Expected bonus for future nigiri
            
        elif card == CardType.MAKI_1:
            return 1.5  # Baseline maki value
        elif card == CardType.MAKI_2:
            return 3.0
        elif card == CardType.MAKI_3:
            return 4.5
            
        elif card == CardType.PUDDING:
            # Pudding scores at game end
            # More valuable in final round
            if round_index == 2:  # Final round (0-indexed)
                return 4.0
            else:
                return 2.0
                
        elif card == CardType.CHOPSTICKS:
            # Enables taking 2 cards in future turn
            return 2.0  # Value for flexibility
            
        else:
            return 1.0  # Default for unknown cards