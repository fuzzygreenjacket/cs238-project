"""
Test and benchmark MCTS agent.

This script evaluates MCTS performance and compares it with other agents.
"""

from sushigo.engine import SushiGoGame
from agents import (
    RandomAgent,
    EVMaximizerAgent,
    SetCompletionAgent,
    MetaStrategyAgent,
    MCTSAgent,
    FastMCTSAgent,
    StrongMCTSAgent
)
import time
import numpy as np


def test_mcts_basic():
    """Basic test: Can MCTS agent complete a game?"""
    print("=" * 70)
    print("BASIC MCTS TEST")
    print("=" * 70)
    
    game = SushiGoGame(num_players=4, seed=42)
    
    agents = [
        FastMCTSAgent(seed=0),
        RandomAgent(seed=1),
        RandomAgent(seed=2),
        RandomAgent(seed=3)
    ]
    
    agent_names = ["MCTS (Fast)", "Random", "Random", "Random"]
    
    print("\nPlayers:")
    for i, name in enumerate(agent_names):
        print(f"  Player {i}: {name}")
    
    print("\nPlaying game...")
    start_time = time.time()
    
    while not game.is_game_over():
        actions = {}
        for p in range(game.num_players):
            if game.hands[p]:
                actions[p] = agents[p].select_action(game, p)
        game.play_turn(actions)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("GAME COMPLETE")
    print("=" * 70)
    
    for i, name in enumerate(agent_names):
        print(f"  {name:20s}: {game.scores[i]:3d} points")
    
    winner = game.scores.index(max(game.scores))
    print(f"\nWinner: {agent_names[winner]}")
    print(f"Game time: {elapsed:.2f} seconds")
    
    # MCTS stats
    mcts_stats = agents[0].get_stats()
    print(f"\nMCTS Statistics:")
    print(f"  Total simulations: {mcts_stats['total_simulations']}")
    print(f"  Total time: {mcts_stats['total_time']:.2f}s")
    print(f"  Avg time per decision: {mcts_stats['avg_time_per_decision']:.3f}s")


def benchmark_mcts_variants():
    """Compare Fast vs Standard vs Strong MCTS."""
    print("\n" + "=" * 70)
    print("MCTS VARIANT COMPARISON")
    print("=" * 70)
    
    variants = [
        ("Fast MCTS (100 sims)", FastMCTSAgent),
        ("Standard MCTS (500 sims)", lambda seed: MCTSAgent(seed=seed, num_simulations=500)),
        ("Strong MCTS (1000 sims)", StrongMCTSAgent)
    ]
    
    num_games = 2  # Reduced from 5 to 2 for speed
    
    for variant_name, AgentClass in variants:
        print(f"\n{variant_name}:")
        
        scores = []
        times = []
        
        for game_num in range(num_games):
            game = SushiGoGame(num_players=4, seed=100 + game_num)
            
            mcts_agent = AgentClass(seed=0)
            agents = [mcts_agent] + [RandomAgent(seed=game_num + i) for i in range(3)]
            
            start = time.time()
            
            while not game.is_game_over():
                actions = {}
                for p in range(game.num_players):
                    if game.hands[p]:
                        actions[p] = agents[p].select_action(game, p)
                game.play_turn(actions)
            
            elapsed = time.time() - start
            
            scores.append(game.scores[0])
            times.append(elapsed)
        
        avg_score = np.mean(scores)
        avg_time = np.mean(times)
        
        print(f"  Average Score: {avg_score:.2f}")
        print(f"  Average Game Time: {avg_time:.2f}s")
        print(f"  Time per decision: ~{avg_time/30:.2f}s")  # ~30 decisions per game


def mcts_vs_heuristic_agents(num_games=5):  # Reduced from 20 to 5
    """
    Compare MCTS against all heuristic agents.
    
    Args:
        num_games: Number of games to play
    """
    print("\n" + "=" * 70)
    print(f"MCTS VS HEURISTIC AGENTS ({num_games} games)")
    print("=" * 70)
    print("Using Fast MCTS (100 simulations) for reasonable speed")
    print()
    
    agent_types = [
        ("MCTS", FastMCTSAgent),
        ("Random", RandomAgent),
        ("EV-Maximizer", EVMaximizerAgent),
        ("Set-Completion", SetCompletionAgent)
    ]
    
    # Track statistics
    total_scores = {name: 0 for name, _ in agent_types}
    wins = {name: 0 for name, _ in agent_types}
    
    start_time = time.time()
    
    for game_num in range(num_games):
        if (game_num + 1) % 5 == 0:
            print(f"Playing game {game_num + 1}/{num_games}...")
        
        game = SushiGoGame(num_players=4, seed=200 + game_num)
        agents = [AgentClass(seed=game_num + i) for i, (_, AgentClass) in enumerate(agent_types)]
        
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
    
    total_time = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 70)
    print("TOURNAMENT RESULTS")
    print("=" * 70)
    print(f"{'Agent':<20s} {'Wins':<10s} {'Avg Score':<12s} {'Win Rate':<10s}")
    print("-" * 70)
    
    for name, _ in agent_types:
        avg_score = total_scores[name] / num_games
        win_rate = wins[name] / num_games * 100
        print(f"{name:<20s} {wins[name]:<10d} {avg_score:<12.2f} {win_rate:<10.1f}%")
    
    print("-" * 70)
    print(f"Total tournament time: {total_time:.1f}s ({total_time/num_games:.1f}s per game)")


def test_mcts_decision_quality():
    """
    Test if MCTS makes good decisions in specific scenarios.
    """
    print("\n" + "=" * 70)
    print("MCTS DECISION QUALITY TEST")
    print("=" * 70)
    
    # Scenario 1: Should complete a set
    print("\nScenario 1: Should complete Tempura pair")
    game = SushiGoGame(num_players=4, seed=999)
    
    # Manually set up a scenario (this is a conceptual test)
    # In practice, you'd need to manipulate game state
    
    mcts = FastMCTSAgent(seed=0)
    action = mcts.select_action(game, 0)
    
    print(f"  MCTS chose action: {action}")
    print(f"  Note: Full decision quality testing requires scenario setup")


def compare_simulation_counts():
    """Show how simulation count affects performance."""
    print("\n" + "=" * 70)
    print("SIMULATION COUNT IMPACT")
    print("=" * 70)
    
    sim_counts = [50, 100, 200]  # Reduced from [50, 100, 200, 500]
    
    print(f"\n{'Simulations':<15s} {'Avg Score':<12s} {'Time/Game':<12s}")
    print("-" * 40)
    
    for sim_count in sim_counts:
        scores = []
        times = []
        
        for game_num in range(3):  # Reduced from 5 to 3
            game = SushiGoGame(num_players=4, seed=300 + game_num)
            
            mcts = MCTSAgent(seed=0, num_simulations=sim_count)
            agents = [mcts] + [RandomAgent(seed=game_num + i) for i in range(3)]
            
            start = time.time()
            
            while not game.is_game_over():
                actions = {}
                for p in range(game.num_players):
                    if game.hands[p]:
                        actions[p] = agents[p].select_action(game, p)
                game.play_turn(actions)
            
            elapsed = time.time() - start
            scores.append(game.scores[0])
            times.append(elapsed)
        
        avg_score = np.mean(scores)
        avg_time = np.mean(times)
        
        print(f"{sim_count:<15d} {avg_score:<12.2f} {avg_time:<12.2f}s")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MCTS TESTING SUITE")
    print("=" * 70)
    print("NOTE: This will take ~15-30 minutes to complete all tests.")
    print("For a quick test (~2-5 min), run: python quick_test_mcts.py")
    print("=" * 70)
    
    # Test 1: Basic functionality
    test_mcts_basic()
    
    # Test 2: Compare MCTS variants (COMMENT OUT if too slow)
    # benchmark_mcts_variants()
    
    # Test 3: MCTS vs heuristic agents
    mcts_vs_heuristic_agents(num_games=5)
    
    # Test 4: Simulation count impact (COMMENT OUT if too slow)
    # compare_simulation_counts()
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)