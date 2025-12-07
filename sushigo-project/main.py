"""
Main script for running Sushi Go simulations.

Demonstrates how to run games with different agent combinations.
For comprehensive statistics and visualizations, use tournament_stats.py
"""

from sushigo.engine import SushiGoGame
from agents import (
    RandomAgent, 
    EVMaximizerAgent, 
    SetCompletionAgent, 
    MetaStrategyAgent,
    QLearningAgent,
    FastMCTSAgent
)


def run_single_game():
    """Run a single game with mixed agents."""
    print("=" * 60)
    print("RUNNING SINGLE GAME WITH MIXED AGENTS")
    print("=" * 60)
    
    game = SushiGoGame(num_players=4, seed=123)
    
    # Mix of different agent types
    # Note: Q-Learning agent is untrained (will play randomly)
    # To use a trained agent, load it first: q_agent.load_q_table("q_table_final.pkl")
    agents = [
        RandomAgent(seed=0),
        EVMaximizerAgent(seed=1),
        SetCompletionAgent(seed=2),
        MetaStrategyAgent(seed=3)
    ]
    
    agent_names = [
        "Random",
        "EV-Maximizer", 
        "Set-Completion",
        "Meta-Strategy"
    ]
    
    print("\nPlayers:")
    for i, name in enumerate(agent_names):
        print(f"  Player {i}: {name}")
    
    # Game loop
    while not game.is_game_over():
        actions = {}
        for p in range(game.num_players):
            if game.hands[p]:
                actions[p] = agents[p].select_action(game, p)
        game.play_turn(actions)
    
    # Print final results
    print("\n" + "=" * 60)
    print("GAME OVER - FINAL RESULTS")
    print("=" * 60)
    for i, name in enumerate(agent_names):
        print(f"  {name:20s} (Player {i}): {game.scores[i]:3d} points")
    
    winner = game.scores.index(max(game.scores))
    print(f"\nWinner: {agent_names[winner]}")


def run_agent_tournament(num_games=10):
    """
    Run a tournament where each agent type plays multiple games.
    
    Args:
        num_games: Number of games to run
    """
    print("\n" + "=" * 60)
    print(f"RUNNING TOURNAMENT ({num_games} games)")
    print("=" * 60)
    
    agent_types = [
        ("Random", RandomAgent),
        ("EV-Maximizer", EVMaximizerAgent),
        ("Set-Completion", SetCompletionAgent),
        ("Meta-Strategy", MetaStrategyAgent),
        ("MCTS (Fast)", FastMCTSAgent)  # Added MCTS
        # Note: Q-Learning excluded unless you have a trained model
        # To include: ("Q-Learning", lambda seed: load_trained_q_agent())
    ]
    
    # Track statistics
    total_scores = {name: 0 for name, _ in agent_types}
    wins = {name: 0 for name, _ in agent_types}
    
    print("\nNote: MCTS will make games slower (searching 100 future states per move)")
    print()
    
    for game_num in range(num_games):
        print(f"Game {game_num + 1}/{num_games}...", end=" ", flush=True)
        
        game = SushiGoGame(num_players=5, seed=game_num)
        agents = [AgentClass(seed=game_num + i) for i, (_, AgentClass) in enumerate(agent_types)]
        
        # Run game
        while not game.is_game_over():
            actions = {}
            for p in range(game.num_players):
                if game.hands[p]:
                    actions[p] = agents[p].select_action(game, p)
            game.play_turn(actions)
        
        # Record results
        for i, (name, _) in enumerate(agent_types):
            score = game.scores[i]
            total_scores[name] += score
            
        winner_idx = game.scores.index(max(game.scores))
        winner_name = agent_types[winner_idx][0]
        wins[winner_name] += 1
        
        print(f"Winner: {winner_name}")
    
    # Print tournament results
    print("\n" + "=" * 60)
    print("TOURNAMENT RESULTS")
    print("=" * 60)
    print(f"{'Agent':<20s} {'Wins':<10s} {'Avg Score':<10s} {'Total':<10s}")
    print("-" * 60)
    
    for name, _ in agent_types:
        avg_score = total_scores[name] / num_games
        print(f"{name:<20s} {wins[name]:<10d} {avg_score:<10.1f} {total_scores[name]:<10d}")
    
    # Determine overall winner
    print("-" * 60)
    best_agent = max(agent_types, key=lambda x: wins[x[0]])
    print(f"Tournament winner: {best_agent[0]} with {wins[best_agent[0]]} wins")


def run_all_random_baseline():
    """Run a baseline game with all random agents."""
    print("\n" + "=" * 60)
    print("BASELINE: ALL RANDOM AGENTS")
    print("=" * 60)
    
    game = SushiGoGame(num_players=4, seed=42)
    agents = [RandomAgent(seed=i) for i in range(4)]
    
    while not game.is_game_over():
        actions = {}
        for p in range(game.num_players):
            if game.hands[p]:
                actions[p] = agents[p].select_action(game, p)
        game.play_turn(actions)
    
    print("\nFinal scores:")
    for i in range(4):
        print(f"  Player {i}: {game.scores[i]} points")


def run_with_trained_qlearning():
    """
    Run tournament including a trained Q-Learning agent.
    Requires q_table_final.pkl to exist (run train_qlearning.py first).
    """
    print("\n" + "=" * 60)
    print("TOURNAMENT WITH TRAINED Q-LEARNING AGENT")
    print("=" * 60)
    
    # Try to load trained Q-learning agent
    try:
        q_agent = QLearningAgent(seed=42, epsilon=0.0)  # No exploration
        q_agent.load_q_table("q_table_final.pkl")
        print("✓ Loaded trained Q-Learning agent")
    except FileNotFoundError:
        print("✗ No trained Q-table found!")
        print("  Run 'python train_qlearning.py' first to train the agent.")
        return
    
    num_games = 50
    
    agent_types = [
        ("Q-Learning (Trained)", lambda seed: q_agent),
        ("Random", RandomAgent),
        ("EV-Maximizer", EVMaximizerAgent),
        ("Set-Completion", SetCompletionAgent),
        ("Meta-Strategy", MetaStrategyAgent)
    ]
    
    # Track statistics
    total_scores = {name: 0 for name, _ in agent_types}
    wins = {name: 0 for name, _ in agent_types}
    
    for game_num in range(num_games):
        if (game_num + 1) % 10 == 0:
            print(f"Game {game_num + 1}/{num_games}...")
        
        game = SushiGoGame(num_players=5, seed=game_num)
        agents = [AgentClass(seed=game_num + i) for i, (_, AgentClass) in enumerate(agent_types)]
        
        # Run game
        while not game.is_game_over():
            actions = {}
            for p in range(game.num_players):
                if game.hands[p]:
                    actions[p] = agents[p].select_action(game, p)
            game.play_turn(actions)
        
        # Record results
        for i, (name, _) in enumerate(agent_types):
            score = game.scores[i]
            total_scores[name] += score
            
        winner_idx = game.scores.index(max(game.scores))
        winner_name = agent_types[winner_idx][0]
        wins[winner_name] += 1
    
    # Print tournament results
    print("\n" + "=" * 60)
    print("TOURNAMENT RESULTS")
    print("=" * 60)
    print(f"{'Agent':<25s} {'Wins':<10s} {'Avg Score':<10s} {'Win %':<10s}")
    print("-" * 60)
    
    for name, _ in agent_types:
        avg_score = total_scores[name] / num_games
        win_pct = wins[name] / num_games * 100
        print(f"{name:<25s} {wins[name]:<10d} {avg_score:<10.1f} {win_pct:<10.1f}")
    
    print("-" * 60)
    best_agent = max(agent_types, key=lambda x: wins[x[0]])
    print(f"Tournament winner: {best_agent[0]} with {wins[best_agent[0]]} wins")


if __name__ == "__main__":
    # Run different test scenarios
    
    # 1. Single game with mixed agents (no Q-Learning)
    run_single_game()
    
    # 2. Tournament (multiple games, no Q-Learning)
    run_agent_tournament(num_games=20)
    
    # 3. Baseline (all random)
    # run_all_random_baseline()
    
    # 4. Tournament with trained Q-Learning agent (uncomment after training)
    # run_with_trained_qlearning()