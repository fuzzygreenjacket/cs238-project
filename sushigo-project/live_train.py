"""
Live Training Tournament

Watch a Q-Learning agent improve in real-time as it competes against
other agents. The agent starts from a checkpoint (or from scratch) and
continues learning during the tournament.

Great for visualizations and understanding learning dynamics!
"""

from sushigo.engine import SushiGoGame
from agents import (
    RandomAgent, 
    EVMaximizerAgent, 
    SetCompletionAgent, 
    MetaStrategyAgent,
    QLearningAgent
)
import matplotlib.pyplot as plt
from collections import deque
import numpy as np



def run_live_training_tournament(num_games=10000, start_from_checkpoint=True, 
                                 checkpoint_path="q_table_checkpoint_1000.pkl"):
    """
    Run tournament where Q-Learning agent trains live against opponents.

    This variant uses:
      - larger moving average window
      - slower plotting & logging for long runs
      - periodic checkpointing
      - curriculum of opponents
    """
    print("=" * 70)
    print("LIVE TRAINING TOURNAMENT")
    print("=" * 70)
    print(f"Games: {num_games}")
    print(f"Start from checkpoint: {start_from_checkpoint}")
    print()

    # Create Q-learning agent (tighter decay so exploration drops over a long run)
    q_agent = QLearningAgent(
        seed=42,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.15,
        epsilon_decay=0.9997,  # decay a bit faster over many games
        min_epsilon=0.01        # allow near-deterministic policy later
    )

    # Try to load checkpoint
    if start_from_checkpoint:
        try:
            q_agent.load_q_table(checkpoint_path)
            print(f"✓ Loaded checkpoint from {checkpoint_path}")
            print(f"  Starting with {q_agent.games_played} games of experience")
            print()
        except FileNotFoundError:
            print(f"✗ Checkpoint not found at {checkpoint_path}")
            print("  Starting from scratch instead")
            print()
            start_from_checkpoint = False

    # Helper: curriculum selection of opponents
    def pick_opponents_for_game(game_num):
        # Phase 1: learn vs Random
        if game_num < int(num_games * 0.2):
            return [
                ("Random", RandomAgent(seed=100 + (game_num % 100))),
                ("Random", RandomAgent(seed=200 + (game_num % 100))),
                ("Random", RandomAgent(seed=300 + (game_num % 100))),
                ("Random", RandomAgent(seed=400 + (game_num % 100)))
            ]
        # Phase 2: mix in heuristics
        elif game_num < int(num_games * 0.6):
            return [
                ("Random", RandomAgent(seed=500 + (game_num % 100))),
                ("EV-Maximizer", EVMaximizerAgent(seed=600 + (game_num % 100))),
                ("Random", RandomAgent(seed=700 + (game_num % 100))),
                ("Set-Completion", SetCompletionAgent(seed=800 + (game_num % 100)))
            ]
        # Phase 3: harder mix and occasional self-play copies
        else:
            # Occasionally include a copy of the Q-agent for self-play
            if (game_num % 50) == 0:
                return [
                    ("Q-Copy", q_agent),  # self-play
                    ("Meta-Strategy", MetaStrategyAgent(seed=900 + (game_num % 100))),
                    ("EV-Maximizer", EVMaximizerAgent(seed=1000 + (game_num % 100))),
                    ("Random", RandomAgent(seed=1100 + (game_num % 100)))
                ]
            else:
                return [
                    ("EV-Maximizer", EVMaximizerAgent(seed=1200 + (game_num % 100))),
                    ("Meta-Strategy", MetaStrategyAgent(seed=1300 + (game_num % 100))),
                    ("Set-Completion", SetCompletionAgent(seed=1400 + (game_num % 100))),
                    ("Random", RandomAgent(seed=1500 + (game_num % 100)))
                ]

    # Track statistics over time (larger window for long runs)
    window_size = 200
    q_scores = deque(maxlen=window_size)
    q_wins = deque(maxlen=window_size)
    opponent_avg_scores = {}

    # For plotting: collect less often to limit array sizes
    plot_interval = 200
    game_numbers = []
    avg_scores = []
    win_rates = []
    epsilon_values = []

    # Periodic checkpointing and printing
    print_interval = 500
    checkpoint_interval = 2000

    # Total statistics
    total_stats = {"Q-Learning": {"score": 0, "wins": 0}}
    # opponents names will be added per-game by curriculum, but initialize defaults
    default_opponent_names = ["Random", "EV-Maximizer", "Set-Completion", "Meta-Strategy"]
    for name in default_opponent_names:
        total_stats[name] = {"score": 0, "wins": 0}

    print("Starting tournament...")
    print("-" * 70)

    for game_num in range(num_games):
        # Create game
        game = SushiGoGame(num_players=5, seed=10000 + game_num)

        # Choose opponents for this game via curriculum
        opponents = pick_opponents_for_game(game_num)
        # Ensure we have pairs (name, agent)
        # If we accidentally pass q_agent as "Q-Copy" (self-play), keep it in-place.
        agents = [q_agent] + [agent for _, agent in opponents]
        agent_names = ["Q-Learning"] + [name for name, _ in opponents]

        # Make sure opponent_avg_scores has deques for these names (lazy init)
        for name in agent_names[1:]:
            if name not in opponent_avg_scores:
                opponent_avg_scores[name] = deque(maxlen=window_size)
                if name not in total_stats:
                    total_stats[name] = {"score": 0, "wins": 0}

        # Track rounds for Q-learning updates
        previous_round = -1

        # Play the game
        while not game.is_game_over():
            actions = {}
            for p in range(game.num_players):
                if game.hands[p]:
                    actions[p] = agents[p].select_action(game, p)
            game.play_turn(actions)

            # Update Q-learning agent when round ends
            if game.round_index > previous_round:
                previous_round = game.round_index
                q_agent.update_after_round(game, player_id=0)

        # Final round update
        q_agent.update_after_round(game, player_id=0)
        q_agent.update_after_game()

        # Record statistics
        scores = game.scores
        winner_idx = scores.index(max(scores))
        winner_name = agent_names[winner_idx]

        # Q-Learning stats
        q_score = scores[0]
        q_scores.append(q_score)
        q_wins.append(1 if winner_idx == 0 else 0)
        total_stats["Q-Learning"]["score"] += q_score
        total_stats["Q-Learning"]["wins"] += (1 if winner_idx == 0 else 0)

        # Opponent stats
        for i, (name, _) in enumerate(opponents):
            opp_score = scores[i + 1]
            opponent_avg_scores[name].append(opp_score)
            total_stats[name]["score"] += opp_score
            total_stats[name]["wins"] += (1 if winner_idx == i + 1 else 0)

        # Collect data for plotting (sampled)
        if (game_num + 1) % plot_interval == 0:
            game_numbers.append(game_num + 1)
            avg_scores.append(np.mean(q_scores) if q_scores else 0.0)
            win_rates.append(np.mean(q_wins) * 100 if q_wins else 0.0)
            epsilon_values.append(q_agent.epsilon)

        # Print progress less often
        if (game_num + 1) % print_interval == 0:
            recent_avg = np.mean(q_scores) if q_scores else 0.0
            recent_win_rate = np.mean(q_wins) * 100 if q_wins else 0.0

            print(f"Game {game_num + 1:6d} | "
                  f"Q Avg (last {window_size}): {recent_avg:5.1f} | "
                  f"Win Rate: {recent_win_rate:5.1f}% | "
                  f"ε: {q_agent.epsilon:.4f} | "
                  f"Last Winner: {winner_name}")

        # Periodic checkpoint save
        if (game_num + 1) % checkpoint_interval == 0:
            ckpt_path = f"q_table_checkpoint_after_{game_num+1}.pkl"
            q_agent.save_q_table(ckpt_path)
            print(f"  ✓ Checkpoint saved to {ckpt_path}")

    print("-" * 70)
    print()

    # Print final statistics
    print("=" * 70)
    print("FINAL TOURNAMENT RESULTS")
    print("=" * 70)
    print(f"{'Agent':<25s} {'Wins':<10s} {'Avg Score':<12s} {'Win Rate':<10s}")
    print("-" * 70)

    for agent_name in ["Q-Learning"] + [name for name, _ in opponents]:
        wins = total_stats.get(agent_name, {}).get("wins", 0)
        avg_score = (total_stats.get(agent_name, {}).get("score", 0) / num_games)
        win_rate = wins / num_games * 100
        print(f"{agent_name:<25s} {wins:<10d} {avg_score:<12.2f} {win_rate:<10.1f}%")

    print("-" * 70)

    # Determine winner
    best_agent = max(total_stats.keys(), key=lambda k: total_stats[k]["wins"])
    print(f"Tournament Champion: {best_agent} with {total_stats[best_agent]['wins']} wins")
    print()

    # Show learning progress
    if start_from_checkpoint:
        print(f"Q-Learning Progress:")
        print(f"  Started with: {q_agent.games_played - num_games} games of experience")
        print(f"  Total games now: {q_agent.games_played}")
        print(f"  Q-table size: {len(q_agent.q_table)} state-action pairs")
        print(f"  Final epsilon: {q_agent.epsilon:.4f}")

    # Plot results (only if we collected anything)
    if game_numbers:
        plot_live_training_results(game_numbers, avg_scores, win_rates, epsilon_values,
                                   start_from_checkpoint)
    else:
        print("Not enough sampled points to plot (try increasing plot_interval).")

    # Save updated Q-table
    save_path = "q_table_after_tournament.pkl"
    q_agent.save_q_table(save_path)
    print(f"\nUpdated Q-table saved to {save_path}")

    return q_agent, total_stats



def plot_live_training_results(game_numbers, avg_scores, win_rates, epsilon_values,
                               from_checkpoint=False):
    """Plot the Q-Learning agent's performance over the tournament."""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Average Score
    axes[0].plot(game_numbers, avg_scores, color='green', linewidth=2, marker='o', 
                 markersize=3)
    axes[0].set_xlabel('Game Number')
    axes[0].set_ylabel('Average Score (last 50 games)')
    title = 'Q-Learning Performance During Tournament'
    if from_checkpoint:
        title += ' (Started from Checkpoint)'
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=np.mean(avg_scores), color='red', linestyle='--', 
                    label=f'Overall Avg: {np.mean(avg_scores):.1f}')
    axes[0].legend()
    
    # Plot 2: Win Rate
    axes[1].plot(game_numbers, win_rates, color='blue', linewidth=2, marker='s', 
                 markersize=3)
    axes[1].set_xlabel('Game Number')
    axes[1].set_ylabel('Win Rate % (last 50 games)')
    axes[1].set_title('Win Rate Over Time', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=20, color='gray', linestyle=':', label='Expected (20% for 5 players)')
    axes[1].axhline(y=np.mean(win_rates), color='red', linestyle='--', 
                    label=f'Actual Avg: {np.mean(win_rates):.1f}%')
    axes[1].legend()
    
    # Plot 3: Epsilon (Exploration Rate)
    axes[2].plot(game_numbers, epsilon_values, color='orange', linewidth=2, marker='^',
                 markersize=3)
    axes[2].set_xlabel('Game Number')
    axes[2].set_ylabel('Epsilon (Exploration Rate)')
    axes[2].set_title('Exploration vs Exploitation Trade-off', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = 'live_training_tournament_results.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nPlots saved to '{filename}'")
    plt.close()


def compare_learning_stages(checkpoint_games=1000, tournament_games=500):
    """
    Compare Q-Learning performance at different learning stages.
    Shows improvement from checkpoint to post-tournament.
    """
    print("\n" + "=" * 70)
    print("COMPARING LEARNING STAGES")
    print("=" * 70)
    
    stages = [
        ("Untrained (Random Play)", QLearningAgent(seed=42, epsilon=1.0), None),
        ("After Initial Training", QLearningAgent(seed=42, epsilon=0.0), 
         f"q_table_checkpoint_{checkpoint_games}.pkl"),
        ("After Tournament", QLearningAgent(seed=42, epsilon=0.0), 
         "q_table_after_tournament.pkl")
    ]
    
    results = {}
    
    for stage_name, agent, checkpoint_path in stages:
        print(f"\nEvaluating: {stage_name}")
        
        # Load checkpoint if specified
        if checkpoint_path:
            try:
                agent.load_q_table(checkpoint_path)
            except FileNotFoundError:
                print(f"  ✗ Checkpoint not found: {checkpoint_path}")
                print(f"  Skipping this stage.")
                continue
        
        # Evaluate against random opponents
        num_eval_games = 100
        scores = []
        wins = 0
        
        for game_num in range(num_eval_games):
            game = SushiGoGame(num_players=4, seed=50000 + game_num)
            agents = [agent] + [RandomAgent(seed=game_num + i) for i in range(3)]
            
            while not game.is_game_over():
                actions = {}
                for p in range(game.num_players):
                    if game.hands[p]:
                        actions[p] = agents[p].select_action(game, p)
                game.play_turn(actions)
            
            scores.append(game.scores[0])
            if game.scores.index(max(game.scores)) == 0:
                wins += 1
        
        avg_score = np.mean(scores)
        win_rate = wins / num_eval_games * 100
        
        results[stage_name] = {
            "avg_score": avg_score,
            "win_rate": win_rate,
            "wins": wins
        }
        
        print(f"  Average Score: {avg_score:.2f}")
        print(f"  Win Rate: {win_rate:.1f}%")
    
    # Print comparison
    print("\n" + "=" * 70)
    print("LEARNING PROGRESSION SUMMARY")
    print("=" * 70)
    print(f"{'Stage':<30s} {'Avg Score':<12s} {'Win Rate':<12s}")
    print("-" * 70)
    
    for stage_name, stats in results.items():
        print(f"{stage_name:<30s} {stats['avg_score']:<12.2f} {stats['win_rate']:<12.1f}%")
    
    # Calculate improvement
    if len(results) >= 2:
        stages_list = list(results.keys())
        first_stage = results[stages_list[0]]
        last_stage = results[stages_list[-1]]
        
        score_improvement = last_stage['avg_score'] - first_stage['avg_score']
        win_rate_improvement = last_stage['win_rate'] - first_stage['win_rate']
        
        print("-" * 70)
        print(f"Improvement: +{score_improvement:.2f} points, +{win_rate_improvement:.1f}% win rate")


if __name__ == "__main__":
    # Run live training tournament
    print("This will take a few minutes...")
    print()
    
    # Option 1: Start from checkpoint (requires pre-training)
    trained_agent, stats = run_live_training_tournament(
        num_games=10000,
        start_from_checkpoint=True,
        checkpoint_path="q_table_checkpoint_1000.pkl"
    )
    
    # Option 2: Start from scratch (uncomment to try)
    # trained_agent, stats = run_live_training_tournament(
    #     num_games=500,
    #     start_from_checkpoint=False
    # )
    
    # Compare learning stages
    compare_learning_stages(checkpoint_games=1000, tournament_games=500)