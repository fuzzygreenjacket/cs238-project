"""
Comprehensive tournament statistics and visualization.

Generates detailed stats and beautiful plots for agent comparison.
"""

from sushigo.engine import SushiGoGame
from agents import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import pandas as pd


class TournamentStats:
    """Track and analyze tournament statistics."""
    
    def __init__(self):
        self.game_data = []  # List of game results
        self.agent_names = []
        
    def record_game(self, game, agent_names):
        """Record a single game's results."""
        self.game_data.append({
            'scores': game.scores.copy(),
            'puddings': game.puddings.copy(),
            'winner': game.scores.index(max(game.scores)),
            'agent_names': agent_names
        })
        if not self.agent_names:
            self.agent_names = agent_names
    
    def get_summary_stats(self):
        """Calculate summary statistics for all agents."""
        num_agents = len(self.agent_names)
        stats = {}
        
        for i, name in enumerate(self.agent_names):
            scores = [game['scores'][i] for game in self.game_data]
            wins = sum(1 for game in self.game_data if game['winner'] == i)
            puddings = [game['puddings'][i] for game in self.game_data]
            
            stats[name] = {
                'games_played': len(self.game_data),
                'wins': wins,
                'win_rate': wins / len(self.game_data) * 100,
                'avg_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'median_score': np.median(scores),
                'avg_pudding': np.mean(puddings),
                'total_score': sum(scores)
            }
        
        return stats
    
    def get_head_to_head(self):
        """Calculate head-to-head win rates."""
        num_agents = len(self.agent_names)
        h2h = np.zeros((num_agents, num_agents))
        
        for game in self.game_data:
            winner = game['winner']
            for i in range(num_agents):
                if i != winner:
                    h2h[winner][i] += 1
        
        # Convert to win rates
        total_games = len(self.game_data)
        h2h = (h2h / total_games * 100).round(1)
        
        return h2h
    
    def get_score_distribution(self):
        """Get score distribution for each agent."""
        distributions = {}
        for i, name in enumerate(self.agent_names):
            scores = [game['scores'][i] for game in self.game_data]
            distributions[name] = scores
        return distributions
    
    def print_detailed_stats(self):
        """Print comprehensive statistics to console."""
        stats = self.get_summary_stats()
        
        print("\n" + "=" * 80)
        print("DETAILED TOURNAMENT STATISTICS")
        print("=" * 80)
        print(f"Total games played: {len(self.game_data)}")
        print()
        
        # Sort by wins
        sorted_agents = sorted(stats.items(), key=lambda x: x[1]['wins'], reverse=True)
        
        print(f"{'Agent':<20s} {'Wins':<8s} {'Win%':<8s} {'Avg Score':<12s} {'Median':<8s} {'Std':<8s}")
        print("-" * 80)
        
        for name, s in sorted_agents:
            print(f"{name:<20s} {s['wins']:<8d} {s['win_rate']:<8.1f} "
                  f"{s['avg_score']:<12.2f} {s['median_score']:<8.1f} {s['std_score']:<8.2f}")
        
        print()
        print("Score Ranges:")
        print(f"{'Agent':<20s} {'Min':<8s} {'Max':<8s} {'Range':<8s}")
        print("-" * 80)
        
        for name, s in sorted_agents:
            score_range = s['max_score'] - s['min_score']
            print(f"{name:<20s} {s['min_score']:<8.1f} {s['max_score']:<8.1f} {score_range:<8.1f}")


def plot_tournament_results(stats: TournamentStats, save_path='tournament_results.png'):
    """Create comprehensive visualization of tournament results."""
    
    summary = stats.get_summary_stats()
    agent_names = stats.agent_names
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Win Rate Bar Chart
    ax1 = fig.add_subplot(gs[0, 0])
    wins = [summary[name]['wins'] for name in agent_names]
    colors = plt.cm.Set3(np.linspace(0, 1, len(agent_names)))
    ax1.bar(range(len(agent_names)), wins, color=colors)
    ax1.set_xticks(range(len(agent_names)))
    ax1.set_xticklabels(agent_names, rotation=45, ha='right')
    ax1.set_ylabel('Wins')
    ax1.set_title('Total Wins', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Average Score Bar Chart
    ax2 = fig.add_subplot(gs[0, 1])
    avg_scores = [summary[name]['avg_score'] for name in agent_names]
    std_scores = [summary[name]['std_score'] for name in agent_names]
    ax2.bar(range(len(agent_names)), avg_scores, yerr=std_scores, 
            color=colors, capsize=5)
    ax2.set_xticks(range(len(agent_names)))
    ax2.set_xticklabels(agent_names, rotation=45, ha='right')
    ax2.set_ylabel('Average Score')
    ax2.set_title('Average Score (±1 std)', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Win Rate Percentage
    ax3 = fig.add_subplot(gs[0, 2])
    win_rates = [summary[name]['win_rate'] for name in agent_names]
    ax3.bar(range(len(agent_names)), win_rates, color=colors)
    ax3.axhline(y=100/len(agent_names), color='r', linestyle='--', 
                label=f'Random baseline ({100/len(agent_names):.1f}%)')
    ax3.set_xticks(range(len(agent_names)))
    ax3.set_xticklabels(agent_names, rotation=45, ha='right')
    ax3.set_ylabel('Win Rate (%)')
    ax3.set_title('Win Rate', fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Score Distribution (Box Plot)
    ax4 = fig.add_subplot(gs[1, :2])
    distributions = stats.get_score_distribution()
    positions = range(len(agent_names))
    bp = ax4.boxplot([distributions[name] for name in agent_names],
                      positions=positions, patch_artist=True, showmeans=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax4.set_xticks(positions)
    ax4.set_xticklabels(agent_names, rotation=45, ha='right')
    ax4.set_ylabel('Score')
    ax4.set_title('Score Distribution (box = IQR, line = median, diamond = mean)', 
                  fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Head-to-Head Heatmap
    ax5 = fig.add_subplot(gs[1, 2])
    h2h = stats.get_head_to_head()
    im = ax5.imshow(h2h, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax5.set_xticks(range(len(agent_names)))
    ax5.set_yticks(range(len(agent_names)))
    ax5.set_xticklabels(agent_names, rotation=45, ha='right', fontsize=8)
    ax5.set_yticklabels(agent_names, fontsize=8)
    ax5.set_title('Head-to-Head Win %\n(row beats column)', fontweight='bold', fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax5)
    cbar.set_label('Win %', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(len(agent_names)):
        for j in range(len(agent_names)):
            if i != j:
                text = ax5.text(j, i, f'{h2h[i, j]:.0f}',
                              ha="center", va="center", color="black", fontsize=8)
    
    # 6. Score Progression (if many games)
    ax6 = fig.add_subplot(gs[2, :])
    if len(stats.game_data) >= 10:
        window = min(10, len(stats.game_data) // 5)
        for i, name in enumerate(agent_names):
            scores = [game['scores'][i] for game in stats.game_data]
            # Rolling average
            if len(scores) >= window:
                rolling_avg = pd.Series(scores).rolling(window=window).mean()
                ax6.plot(rolling_avg, label=name, linewidth=2, color=colors[i])
        
        ax6.set_xlabel('Game Number')
        ax6.set_ylabel(f'Score (rolling avg, window={window})')
        ax6.set_title('Score Progression Over Tournament', fontweight='bold')
        ax6.legend(loc='best')
        ax6.grid(alpha=0.3)
    else:
        # Just plot raw scores
        for i, name in enumerate(agent_names):
            scores = [game['scores'][i] for game in stats.game_data]
            ax6.plot(scores, 'o-', label=name, color=colors[i])
        
        ax6.set_xlabel('Game Number')
        ax6.set_ylabel('Score')
        ax6.set_title('Score Per Game', fontweight='bold')
        ax6.legend(loc='best')
        ax6.grid(alpha=0.3)
    
    plt.suptitle('Tournament Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plots saved to '{save_path}'")
    plt.close()


def plot_agent_comparison(stats: TournamentStats, save_path='agent_comparison.png'):
    """Create detailed comparison plots."""
    
    summary = stats.get_summary_stats()
    agent_names = stats.agent_names
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.Set3(np.linspace(0, 1, len(agent_names)))
    
    # 1. Radar Chart - Multiple Metrics
    ax = axes[0, 0]
    categories = ['Win Rate', 'Avg Score', 'Consistency', 'Peak Score']
    num_vars = len(categories)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    ax = plt.subplot(2, 2, 1, projection='polar')
    
    for i, name in enumerate(agent_names):
        s = summary[name]
        # Normalize metrics to 0-100 scale
        values = [
            s['win_rate'],
            (s['avg_score'] / 50) * 100,  # Assuming max ~50
            100 - (s['std_score'] / 10) * 100,  # Lower std = higher consistency
            (s['max_score'] / 70) * 100  # Assuming max ~70
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    ax.set_title('Multi-Metric Comparison', fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    # 2. Score Range Visualization
    ax = axes[0, 1]
    for i, name in enumerate(agent_names):
        s = summary[name]
        ax.plot([i, i], [s['min_score'], s['max_score']], 'o-', 
                linewidth=3, markersize=8, color=colors[i])
        ax.plot(i, s['avg_score'], 's', markersize=10, color=colors[i])
        ax.plot(i, s['median_score'], 'd', markersize=8, color='red', alpha=0.6)
    
    ax.set_xticks(range(len(agent_names)))
    ax.set_xticklabels(agent_names, rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Score Range (○ min/max, □ mean, ◇ median)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Performance Consistency
    ax = axes[1, 0]
    avg_scores = [summary[name]['avg_score'] for name in agent_names]
    std_scores = [summary[name]['std_score'] for name in agent_names]
    
    for i, name in enumerate(agent_names):
        ax.scatter(std_scores[i], avg_scores[i], s=200, color=colors[i], 
                  alpha=0.6, edgecolors='black', linewidths=2, label=name)
    
    ax.set_xlabel('Standard Deviation (lower = more consistent)')
    ax.set_ylabel('Average Score')
    ax.set_title('Performance vs Consistency', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add quadrant lines
    ax.axvline(np.mean(std_scores), color='gray', linestyle='--', alpha=0.5)
    ax.axhline(np.mean(avg_scores), color='gray', linestyle='--', alpha=0.5)
    
    # 4. Pudding Strategy
    ax = axes[1, 1]
    avg_puddings = [summary[name]['avg_pudding'] for name in agent_names]
    avg_scores = [summary[name]['avg_score'] for name in agent_names]
    
    ax.scatter(avg_puddings, avg_scores, s=200, c=range(len(agent_names)), 
              cmap='Set3', alpha=0.6, edgecolors='black', linewidths=2)
    
    for i, name in enumerate(agent_names):
        ax.annotate(name, (avg_puddings[i], avg_scores[i]), 
                   fontsize=9, ha='center')
    
    ax.set_xlabel('Average Puddings')
    ax.set_ylabel('Average Score')
    ax.set_title('Pudding Strategy vs Performance', fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Plots saved to '{save_path}'")
    plt.close()


def run_comprehensive_tournament(num_games=50, include_mcts=False):
    """Run comprehensive tournament with full statistics."""
    
    print("=" * 80)
    print("COMPREHENSIVE TOURNAMENT WITH STATISTICS")
    print("=" * 80)
    print(f"Games: {num_games}")
    print()
    
    # Setup agents
    agent_types = [
        # ("Random", RandomAgent),
        ("EV-Maximizer", EVMaximizerAgent),
        ("Set-Completion", SetCompletionAgent),
        ("Meta-Strategy", MetaStrategyAgent)
    ]

    q_agent = QLearningAgent(seed=42, epsilon=0.0)
    q_agent.load_q_table("q_table_final.pkl")
    agent_types.append(("Q-Learning", lambda seed: q_agent))
    
    if include_mcts:
        print("⚠️  Including MCTS will make tournament MUCH slower")
        agent_types.append(("MCTS (Fast)", FastMCTSAgent))
    
    agent_names = [name for name, _ in agent_types]
    num_players = len(agent_types)
    
    # Initialize stats tracker
    stats = TournamentStats()
    
    print(f"Players: {', '.join(agent_names)}")
    print(f"Expected time: ~{num_games * 5 if not include_mcts else num_games * 60} seconds")
    print()
    
    # Run tournament
    for game_num in range(num_games):
        if (game_num + 1) % 10 == 0:
            print(f"Game {game_num + 1}/{num_games}...")
        
        game = SushiGoGame(num_players=num_players, seed=game_num)
        agents = [AgentClass(seed=game_num + i) 
                 for i, (_, AgentClass) in enumerate(agent_types)]
        
        while not game.is_game_over():
            actions = {}
            for p in range(num_players):
                if game.hands[p]:
                    actions[p] = agents[p].select_action(game, p)
            game.play_turn(actions)
        
        stats.record_game(game, agent_names)
    
    # Print detailed statistics
    stats.print_detailed_stats()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_tournament_results(stats, 'tournament_results.png')
    plot_agent_comparison(stats, 'agent_comparison.png')
    
    print("\n" + "=" * 80)
    print("✓ TOURNAMENT COMPLETE")
    print("=" * 80)
    print("Generated files:")
    print("  📊 tournament_results.png - Main dashboard")
    print("  📊 agent_comparison.png - Detailed comparison")
    
    return stats


if __name__ == "__main__":
    # Run tournament
    # stats = run_comprehensive_tournament(num_games=50, include_mcts=False)
    
    # Optional: Include MCTS (slower)
    stats = run_comprehensive_tournament(num_games=20, include_mcts=True)

    