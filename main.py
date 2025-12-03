from sushigo.engine import SushiGoGame
from sushigo.agents import RandomAgent



if __name__ == "__main__":
    game = SushiGoGame(num_players=4, seed=123)

    agents = [RandomAgent(seed=i) for i in range(4)]

    while not game.is_game_over():
        actions = {}
        for p in range(game.num_players):
            if game.hands[p]:
                actions[p] = agents[p].select_action(game, p)

        game.play_turn(actions)
