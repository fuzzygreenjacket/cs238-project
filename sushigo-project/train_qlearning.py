"""
Training script for Q-Learning agent.

This script trains a Q-Learning agent by playing many games against
random opponents and itself. The agent learns optimal play through experience.
"""

from sushigo.engine import SushiGoGame
from agents import RandomAgent, QLearningAgent, EVMaximizerAgent
import matplotlib.pyplot as plt
from collections import deque


def train_qlearning(num_training_games=1000, num_players=4, save_every=100):
    """
    Train a Q-Learning agent.
    
    Args:
        num_training_games: Number of games to train on
        num_players: Number of players in each game
        save_every: Save Q-table every N games
    """
    print("=" * 70)
    print("TRAINING Q-LEARNING AGENT")
    print("=" * 70)
    print(f"Training games: {num_training_games}")
    print(f"Players per game: {num_players}")
    print()
    
    # Create Q-learning agent (player 0)
    q_agent = QLearningAgent(
        seed=42,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.3,  # Start with 30% exploration
        epsilon_decay=0.9995,
        min_epsilon=0.05
    )
    
    # Track statistics
    scores_history = deque(maxlen=100)  # Last 100 games
    win_history = deque(maxlen=100)
    epsilon_history = []
    avg_score_history = []
    
    for game_num in range(num_training_games):
        # Create game with random seed
        game = SushiGoGame(num_players=num_players, seed=game_num)
        
        # Q-learning agent plays as player 0, others are random
        agents = [q_agent] + [RandomAgent(seed=game_num + i) for i in range(1, num_players)]
        
        # Track when rounds end for Q-learning updates
        previous_round = -1
        
        # Play the game
        while not game.is_game_over():
            actions = {}
            
            # Get actions from all agents
            for p in range(game.num_players):
                if game.hands[p]:
                    actions[p] = agents[p].select_action(game, p)
            
            # Execute turn
            game.play_turn(actions)
            
            # Check if round ended (for Q-learning update)
            if game.round_index > previous_round:
                previous_round = game.round_index
                q_agent.update_after_round(game, player_id=0)
        
        # Final round update
        q_agent.update_after_round(game, player_id=0)
        
        # Game finished - record statistics
        q_score = game.scores[0]
        scores_history.append(q_score)
        
        winner = game.scores.index(max(game.scores))
        win_history.append(1 if winner == 0 else 0)
        
        q_agent.update_after_game()
        
        # Track for plotting
        if (game_num + 1) % 10 == 0:
            epsilon_history.append(q_agent.epsilon)
            avg_score_history.append(sum(scores_history) / len(scores_history))
        
        # Print progress
        if (game_num + 1) % 100 == 0:
            avg_score = sum(scores_history) / len(scores_history)
            win_rate = sum(win_history) / len(win_history)
            stats = q_agent.get_stats()
            
            print(f"Game {game_num + 1:4d} | "
                  f"Avg Score (last 100): {avg_score:5.1f} | "
                  f"Win Rate: {win_rate:.2%} | "
                  f"Epsilon: {stats['epsilon']:.4f} | "
                  f"Q-table size: {stats['q_table_size']}")
        
        # Save checkpoint
        if (game_num + 1) % save_every == 0 and (game_num + 1) < num_training_games:
            q_agent.save_q_table(f"q_table_checkpoint_{game_num + 1}.pkl")
    
    # Save final Q-table
    q_agent.save_q_table("q_table_final.pkl")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    # Plot training progress
    plot_training_progress(epsilon_history, avg_score_history)
    
    return q_agent


def plot_training_progress(epsilon_history, avg_score_history):
    """Plot training metrics."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot epsilon (exploration rate)
    ax1.plot(epsilon_history, color='blue', linewidth=2)
    ax1.set_xlabel('Training Steps (x10 games)')
    ax1.set_ylabel('Epsilon (Exploration Rate)')
    ax1.set_title('Exploration Rate Over Training')
    ax1.grid(True, alpha=0.3)
    
    # Plot average score
    ax2.plot(avg_score_history, color='green', linewidth=2)
    ax2.set_xlabel('Training Steps (x10 games)')
    ax2.set_ylabel('Average Score (last 100 games)')
    ax2.set_title('Learning Progress: Average Score')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qlearning_training_progress.png', dpi=150)
    print("\nTraining plots saved to 'qlearning_training_progress.png'")
    plt.close()


def evaluate_trained_agent(q_agent, num_eval_games=100):
    """
    Evaluate trained Q-learning agent against various opponents.
    
    Args:
        q_agent: Trained QLearningAgent
        num_eval_games: Number of evaluation games
    """
    print("\n" + "=" * 70)
    print("EVALUATING TRAINED AGENT")
    print("=" * 70)
    
    # Disable exploration for evaluation
    original_epsilon = q_agent.epsilon
    q_agent.epsilon = 0.0  # Pure exploitation
    
    opponent_types = [
        ("Random", RandomAgent),
        ("EV-Maximizer", EVMaximizerAgent)
    ]
    
    for opp_name, OpponentClass in opponent_types:
        print(f"\nVs. {opp_name} opponents:")
        
        scores = []
        wins = 0
        
        for game_num in range(num_eval_games):
            game = SushiGoGame(num_players=4, seed=10000 + game_num)
            
            # Q-learning agent vs 3 opponents
            agents = [q_agent] + [OpponentClass(seed=game_num + i) for i in range(3)]
            
            while not game.is_game_over():
                actions = {}
                for p in range(game.num_players):
                    if game.hands[p]:
                        actions[p] = agents[p].select_action(game, p)
                game.play_turn(actions)
            
            # Record results
            scores.append(game.scores[0])
            if game.scores.index(max(game.scores)) == 0:
                wins += 1
        
        avg_score = sum(scores) / len(scores)
        win_rate = wins / num_eval_games
        
        print(f"  Average Score: {avg_score:.2f}")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Wins: {wins}/{num_eval_games}")
    
    # Restore epsilon
    q_agent.epsilon = original_epsilon


def compare_before_after_training():
    """
    Compare untrained vs trained Q-learning agent.
    """
    print("\n" + "=" * 70)
    print("BEFORE/AFTER COMPARISON")
    print("=" * 70)
    
    num_games = 50
    
    # Untrained agent (high epsilon, empty Q-table)
    print("\n1. Untrained Q-Learning Agent (random play):")
    untrained = QLearningAgent(seed=42, epsilon=1.0)  # Pure exploration
    
    scores = []
    for game_num in range(num_games):
        game = SushiGoGame(num_players=4, seed=20000 + game_num)
        agents = [untrained] + [RandomAgent(seed=game_num + i) for i in range(3)]
        
        while not game.is_game_over():
            actions = {}
            for p in range(game.num_players):
                if game.hands[p]:
                    actions[p] = agents[p].select_action(game, p)
            game.play_turn(actions)
        
        scores.append(game.scores[0])
    
    print(f"  Average Score: {sum(scores) / len(scores):.2f}")
    
    # Trained agent
    print("\n2. Trained Q-Learning Agent (exploiting learned policy):")
    trained = QLearningAgent(seed=42, epsilon=0.0)
    try:
        trained.load_q_table("q_table_final.pkl")
        
        scores = []
        for game_num in range(num_games):
            game = SushiGoGame(num_players=4, seed=20000 + game_num)
            agents = [trained] + [RandomAgent(seed=game_num + i) for i in range(3)]
            
            while not game.is_game_over():
                actions = {}
                for p in range(game.num_players):
                    if game.hands[p]:
                        actions[p] = agents[p].select_action(game, p)
                game.play_turn(actions)
            
            scores.append(game.scores[0])
        
        print(f"  Average Score: {sum(scores) / len(scores):.2f}")
        
    except FileNotFoundError:
        print("  No trained Q-table found. Train first using train_qlearning()")


if __name__ == "__main__":
    # Train the agent
    trained_agent = train_qlearning(num_training_games=1000, num_players=4)
    
    # Evaluate the trained agent
    evaluate_trained_agent(trained_agent, num_eval_games=100)
    
    # Compare before/after
    compare_before_after_training()