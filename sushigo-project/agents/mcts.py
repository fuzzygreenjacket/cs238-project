"""
Monte Carlo Tree Search (MCTS) Agent

This agent uses MCTS to search through possible future game states and
select the action that leads to the best expected outcome.

Strategy:
1. Selection: Traverse tree using UCB1 to balance exploration/exploitation
2. Expansion: Add new child nodes for unexplored actions
3. Simulation: Randomly play out the game from new node
4. Backpropagation: Update all nodes in path with simulation result

Unlike heuristic agents, MCTS actually plans ahead by simulating games.
"""

import random
import math
import time
from typing import TYPE_CHECKING, Optional, List, Dict
from collections import defaultdict
import copy

if TYPE_CHECKING:
    from sushigo.engine import SushiGoGame
    from sushigo.actions import Action


class MCTSNode:
    """Node in the MCTS search tree."""
    
    def __init__(self, state_hash, parent=None, action=None):
        """
        Initialize MCTS node.
        
        Args:
            state_hash: Hashable representation of game state
            parent: Parent node in tree
            action: Action that led to this node
        """
        self.state_hash = state_hash
        self.parent = parent
        self.action = action
        
        # MCTS statistics
        self.visits = 0
        self.total_value = 0.0
        
        # Children
        self.children = []
        self.untried_actions = []
        
    def ucb1_score(self, exploration_constant=1.414):
        """
        Calculate UCB1 score for this node.
        
        UCB1 = exploitation + exploration
             = (average value) + C * sqrt(ln(parent visits) / visits)
        
        Args:
            exploration_constant: C parameter (√2 is theoretically optimal)
            
        Returns:
            UCB1 score
        """
        if self.visits == 0:
            return float('inf')  # Unvisited nodes have infinite priority
        
        exploitation = self.total_value / self.visits
        
        if self.parent is None:
            exploration = 0
        else:
            exploration = exploration_constant * math.sqrt(
                math.log(self.parent.visits) / self.visits
            )
        
        return exploitation + exploration
    
    def best_child(self, exploration_constant=1.414):
        """Select child with highest UCB1 score."""
        return max(self.children, key=lambda c: c.ucb1_score(exploration_constant))
    
    def most_visited_child(self):
        """Select child with most visits (used for final action selection)."""
        return max(self.children, key=lambda c: c.visits)
    
    def update(self, value):
        """Update node statistics after simulation."""
        self.visits += 1
        self.total_value += value
    
    def add_child(self, state_hash, action):
        """Add a child node."""
        child = MCTSNode(state_hash, parent=self, action=action)
        self.children.append(child)
        return child


class MCTSAgent:
    """Monte Carlo Tree Search agent."""
    
    def __init__(self, seed=None, num_simulations=500, exploration_constant=1.414,
                 time_limit=None):
        """
        Initialize MCTS agent.
        
        Args:
            seed: Random seed
            num_simulations: Number of MCTS simulations per move
            exploration_constant: UCB1 exploration parameter (√2 ≈ 1.414)
            time_limit: Optional time limit in seconds per move
        """
        self.rng = random.Random(seed)
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.time_limit = time_limit
        
        # Statistics
        self.total_simulations = 0
        self.total_time = 0.0
        
    def select_action(self, game: 'SushiGoGame', player_id: int) -> 'Action':
        """
        Select action using MCTS.
        
        Args:
            game: Current game state
            player_id: This agent's player ID
            
        Returns:
            Best action according to MCTS
        """
        start_time = time.time()
        
        hand = game.get_player_hand(player_id)
        if not hand:
            return None
        
        valid_actions = game.get_valid_actions(player_id)
        if not valid_actions:
            return None
        
        # If only one action, no need to search
        if len(valid_actions) == 1:
            return valid_actions[0]
        
        # Filter out chopsticks actions to avoid state inconsistency issues
        # (Chopsticks actions create problems when reusing tree nodes across simulations)
        non_chopsticks_actions = [a for a in valid_actions if not a.use_chopsticks]
        if not non_chopsticks_actions:
            # If all actions require chopsticks, just pick randomly
            return self.rng.choice(valid_actions)
        
        # Use only non-chopsticks actions for search
        search_actions = non_chopsticks_actions
        
        # Create root node
        root_state_hash = self._hash_state(game, player_id)
        root = MCTSNode(root_state_hash)
        root.untried_actions = search_actions.copy()
        
        # Run MCTS simulations
        simulations_run = 0
        while simulations_run < self.num_simulations:
            # Check time limit
            if self.time_limit and (time.time() - start_time) > self.time_limit:
                break
            
            # Clone game for simulation
            sim_game = self._clone_game(game)
            
            # Run one MCTS iteration
            self._mcts_iteration(root, sim_game, player_id)
            
            simulations_run += 1
        
        # Select best action (most visited child)
        if not root.children:
            # No children created, return random valid action
            return self.rng.choice(search_actions)
        
        best_child = root.most_visited_child()
        
        # Update statistics
        self.total_simulations += simulations_run
        elapsed = time.time() - start_time
        self.total_time += elapsed
        
        return best_child.action
    
    def _mcts_iteration(self, root: MCTSNode, game: 'SushiGoGame', player_id: int):
        """
        Run one MCTS iteration: Selection, Expansion, Simulation, Backpropagation.
        
        Args:
            root: Root node of search tree
            game: Game state (will be modified)
            player_id: This agent's player ID
        """
        # Phase 1: Selection - traverse tree to leaf
        node = root
        path = [node]
        
        while not game.is_game_over() and len(node.untried_actions) == 0 and len(node.children) > 0:
            node = node.best_child(self.exploration_constant)
            path.append(node)
            
            # Apply action to game
            if node.action is not None:
                self._apply_action_to_game(game, player_id, node.action)
        
        # Phase 2: Expansion - add new child if possible
        if not game.is_game_over() and len(node.untried_actions) > 0:
            # Pick random untried action
            action = self.rng.choice(node.untried_actions)
            node.untried_actions.remove(action)
            
            # Apply action
            self._apply_action_to_game(game, player_id, action)
            
            # Create child node
            new_state_hash = self._hash_state(game, player_id)
            child = node.add_child(new_state_hash, action)
            
            # Get child's possible actions (excluding chopsticks)
            if not game.is_game_over():
                all_actions = game.get_valid_actions(player_id)
                child.untried_actions = [a for a in all_actions if not a.use_chopsticks]
            
            node = child
            path.append(node)
        
        # Phase 3: Simulation - play out randomly to end
        final_score = self._simulate_game(game, player_id)
        
        # Phase 4: Backpropagation - update all nodes in path
        for node in path:
            node.update(final_score)
    
    def _simulate_game(self, game: 'SushiGoGame', player_id: int) -> float:
        """
        Simulate game to completion using random policy.
        
        Args:
            game: Game state (will be modified)
            player_id: This agent's player ID
            
        Returns:
            Final score for player_id
        """
        # Play randomly until game ends
        while not game.is_game_over():
            actions = {}
            
            for p in range(game.num_players):
                if game.hands[p]:
                    valid_actions = game.get_valid_actions(p)
                    if valid_actions:
                        # Prefer non-chopsticks actions to avoid errors
                        non_chopsticks_actions = [a for a in valid_actions if not a.use_chopsticks]
                        if non_chopsticks_actions:
                            actions[p] = self.rng.choice(non_chopsticks_actions)
                        else:
                            actions[p] = self.rng.choice(valid_actions)
            
            if actions:
                game.play_turn(actions)
        
        # Return final score
        return game.scores[player_id]
    
    def _apply_action_to_game(self, game: 'SushiGoGame', player_id: int, action):
        """
        Apply a single action to the game state.
        This requires simulating all players' actions.
        
        Args:
            game: Game state (will be modified)
            player_id: Player making the action
            action: Action to apply
        """
        # Need to simulate other players' actions too
        actions = {}
        
        for p in range(game.num_players):
            if p == player_id:
                actions[p] = action
            else:
                # Other players play randomly in our simulation
                if game.hands[p]:
                    valid_actions = game.get_valid_actions(p)
                    if valid_actions:
                        # Filter out chopsticks actions for simulated players
                        # to avoid chopsticks mismatch errors
                        non_chopsticks_actions = [a for a in valid_actions if not a.use_chopsticks]
                        if non_chopsticks_actions:
                            actions[p] = self.rng.choice(non_chopsticks_actions)
                        else:
                            # If only chopsticks actions available, use one
                            actions[p] = self.rng.choice(valid_actions)
        
        if actions:
            game.play_turn(actions)
    
    def _hash_state(self, game: 'SushiGoGame', player_id: int):
        """
        Create hashable representation of game state from player's perspective.
        
        Args:
            game: Game state
            player_id: This agent's player ID
            
        Returns:
            Hashable state representation
        """
        public_state = game.clone_public_state()
        
        # Include: hand, tableau, round, scores, puddings
        hand_tuple = tuple(sorted([c.name for c in game.get_player_hand(player_id)]))
        tableau_tuple = tuple(sorted([c.name for c in public_state.tableaus[player_id]]))
        
        state = (
            hand_tuple,
            tableau_tuple,
            public_state.round_index,
            tuple(public_state.scores),
            tuple(public_state.puddings),
            tuple(public_state.hands_sizes)
        )
        
        return state
    
    def _clone_game(self, game: 'SushiGoGame') -> 'SushiGoGame':
        """
        Create a deep copy of the game state for simulation.
        
        Args:
            game: Game to clone
            
        Returns:
            Cloned game
        """
        # Deep copy the entire game state
        cloned = copy.deepcopy(game)
        return cloned
    
    def get_stats(self):
        """Return agent statistics."""
        avg_time = self.total_time / max(1, self.total_simulations / self.num_simulations)
        return {
            'total_simulations': self.total_simulations,
            'total_time': self.total_time,
            'avg_time_per_decision': avg_time,
            'simulations_per_decision': self.num_simulations
        }


class FastMCTSAgent(MCTSAgent):
    """
    Faster MCTS agent with reduced simulation count.
    Good for quick games and testing.
    """
    
    def __init__(self, seed=None):
        super().__init__(
            seed=seed,
            num_simulations=100,  # Fewer simulations
            exploration_constant=1.414,
            time_limit=1.0  # 1 second time limit
        )


class StrongMCTSAgent(MCTSAgent):
    """
    Stronger MCTS agent with more simulations.
    Takes longer but plays better.
    """
    
    def __init__(self, seed=None):
        super().__init__(
            seed=seed,
            num_simulations=1000,  # More simulations
            exploration_constant=1.414,
            time_limit=5.0  # 5 second time limit
        )