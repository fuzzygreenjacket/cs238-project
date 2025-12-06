"""
Set-Completion Optimizer Agent

This agent commits early to high-value sets and prioritizes completing them.
It plays like a human who picks a strategy early and sticks with it.

Strategy:
- Analyze initial hand to choose a primary goal
- Commit to that goal for the round
- Heavily prioritize cards that advance the chosen set
- Only deviate when completion is impossible or highly unlikely
"""

import random
from typing import TYPE_CHECKING, Optional
from collections import Counter

if TYPE_CHECKING:
    from sushigo.engine import SushiGoGame
    from sushigo.actions import Action
    from sushigo.cards import CardType


class SetCompletionAgent:
    """Agent that commits to completing high-value card sets."""
    
    def __init__(self, seed=None):
        """
        Initialize Set-Completion agent.
        
        Args:
            seed: Random seed for strategy selection tie-breaking
        """
        self.rng = random.Random(seed)
        # Track commitment per round
        self.committed_strategy = None
        self.current_round = -1
        
    def select_action(self, game: 'SushiGoGame', player_id: int) -> 'Action':
        """
        Select action that best advances current set-completion strategy.
        
        Args:
            game: Current game state
            player_id: This agent's player ID
            
        Returns:
            Action that best advances the committed strategy
        """
        hand = game.get_player_hand(player_id)
        if not hand:
            return None
            
        public_state = game.clone_public_state()
        
        # Reset strategy on new round
        if public_state.round_index != self.current_round:
            self.current_round = public_state.round_index
            self.committed_strategy = None
        
        my_tableau = public_state.tableaus[player_id]
        
        # Decide strategy if not committed yet
        if self.committed_strategy is None:
            self.committed_strategy = self._choose_strategy(hand, my_tableau)
        
        # Get valid actions
        valid_actions = game.get_valid_actions(player_id)
        if not valid_actions:
            return None
        
        # Score each action based on committed strategy
        best_actions = []
        best_score = float('-inf')
        
        for action in valid_actions:
            score = self._score_action_for_strategy(
                action, hand, my_tableau, self.committed_strategy
            )
            
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)
        
        return self.rng.choice(best_actions)
    
    def _choose_strategy(self, hand, my_tableau) -> str:
        """
        Choose a strategy to commit to based on hand and tableau.
        
        Strategies: 'tempura', 'sashimi', 'dumpling', 'maki', 'nigiri', 'pudding'
        
        Args:
            hand: Current hand
            my_tableau: Current tableau
            
        Returns:
            Strategy name to commit to
        """
        from sushigo.cards import CardType
        
        hand_counts = Counter(hand)
        tableau_counts = Counter(my_tableau)
        
        # Priority 1: Near-complete sets in tableau + card in hand
        if tableau_counts[CardType.TEMPURA] >= 1 and hand_counts[CardType.TEMPURA] >= 1:
            return 'tempura'
        
        if tableau_counts[CardType.SASHIMI] >= 1 and hand_counts[CardType.SASHIMI] >= 1:
            return 'sashimi'
        
        # Priority 2: Multiple cards in hand suggest strong opportunity
        if hand_counts[CardType.TEMPURA] >= 2:
            return 'tempura'
        
        if hand_counts[CardType.SASHIMI] >= 2:
            return 'sashimi'
        
        if hand_counts[CardType.DUMPLING] >= 2:
            return 'dumpling'
        
        # Priority 3: Maki majority potential
        total_maki = (hand_counts[CardType.MAKI_1] + 
                     hand_counts[CardType.MAKI_2] + 
                     hand_counts[CardType.MAKI_3])
        if total_maki >= 2:
            return 'maki'
        
        # Priority 4: Nigiri + Wasabi combo
        nigiri_types = [CardType.SQUID_NIGIRI, CardType.SALMON_NIGIRI, CardType.EGG_NIGIRI]
        total_nigiri = sum(hand_counts[n] for n in nigiri_types)
        
        if (hand_counts[CardType.WASABI] >= 1 and total_nigiri >= 1) or total_nigiri >= 3:
            return 'nigiri'
        
        # Default: Dumpling (steady accumulation strategy)
        return 'dumpling'
    
    def _score_action_for_strategy(self, action, hand, my_tableau, strategy: str) -> float:
        """
        Score an action based on how well it advances the committed strategy.
        
        Args:
            action: Action to evaluate
            hand: Current hand
            my_tableau: Current tableau
            strategy: Committed strategy name
            
        Returns:
            Score for this action (higher = better for strategy)
        """
        from sushigo.cards import CardType
        
        # Get cards from action
        if not action.use_chopsticks:
            cards = [hand[action.first_index]]
        else:
            cards = [hand[action.first_index], hand[action.second_index]]
        
        total_score = 0
        tableau_counts = Counter(my_tableau)
        
        for card in cards:
            score = 0
            
            if strategy == 'tempura':
                if card == CardType.TEMPURA:
                    # High priority to complete pairs
                    if tableau_counts[CardType.TEMPURA] % 2 == 1:
                        score = 100  # COMPLETE THE PAIR!
                    else:
                        score = 50  # Start new pair
                elif card == CardType.CHOPSTICKS:
                    score = 20  # Useful for flexibility
                else:
                    score = 1  # Very low priority
                    
            elif strategy == 'sashimi':
                if card == CardType.SASHIMI:
                    sashimi_count = tableau_counts[CardType.SASHIMI]
                    if sashimi_count == 2:
                        score = 150  # COMPLETE THE TRIPLE!
                    elif sashimi_count == 1:
                        score = 80  # Need one more
                    else:
                        score = 40  # Start the set
                elif card == CardType.CHOPSTICKS:
                    score = 20
                else:
                    score = 1
                    
            elif strategy == 'dumpling':
                if card == CardType.DUMPLING:
                    # Always good, diminishing returns
                    dumpling_count = tableau_counts[CardType.DUMPLING]
                    score = max(60 - dumpling_count * 5, 20)
                elif card == CardType.CHOPSTICKS:
                    score = 15
                else:
                    score = 1
                    
            elif strategy == 'maki':
                if card in [CardType.MAKI_1, CardType.MAKI_2, CardType.MAKI_3]:
                    # Prioritize high-value maki
                    maki_values = {
                        CardType.MAKI_3: 80,
                        CardType.MAKI_2: 60,
                        CardType.MAKI_1: 40
                    }
                    score = maki_values[card]
                elif card == CardType.CHOPSTICKS:
                    score = 25  # Good for grabbing multiple maki
                else:
                    score = 1
                    
            elif strategy == 'nigiri':
                # Prioritize wasabi first, then best nigiri
                if card == CardType.WASABI:
                    # Only valuable if we don't have unused wasabi
                    if tableau_counts[CardType.WASABI] == 0:
                        score = 90
                    else:
                        score = 10
                elif card == CardType.SQUID_NIGIRI:
                    if tableau_counts[CardType.WASABI] > 0:
                        score = 100  # Best nigiri on wasabi!
                    else:
                        score = 50
                elif card == CardType.SALMON_NIGIRI:
                    if tableau_counts[CardType.WASABI] > 0:
                        score = 80
                    else:
                        score = 35
                elif card == CardType.EGG_NIGIRI:
                    if tableau_counts[CardType.WASABI] > 0:
                        score = 60
                    else:
                        score = 20
                elif card == CardType.CHOPSTICKS:
                    score = 15
                else:
                    score = 1
                    
            elif strategy == 'pudding':
                if card == CardType.PUDDING:
                    score = 70
                elif card == CardType.CHOPSTICKS:
                    score = 15
                else:
                    score = 1
                    
            total_score += score
        
        # Bonus for using chopsticks when committed
        if action.use_chopsticks:
            total_score += 30  # Big bonus for getting 2 cards toward goal
            
        return total_score