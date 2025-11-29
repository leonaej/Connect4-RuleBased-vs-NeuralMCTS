# main/play_heuristic_vs_random.py
import time
from pathlib import Path
import sys



def pretty_print(board):
    """Print Connect4 board nicely with circles for players."""
    symbols = {0: ".", 1: "●", -1: "○"}
    rows, cols = board.shape
    print("\nBoard:")
    for r in range(rows):
        print(" ".join(symbols[int(cell)] for cell in board[r]))
    print("-" * (2 * cols - 1))  # line under board



project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from connect4_env.connect4_env import Connect4Env
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent, check_win


NUM_GAMES = 400
SLEEP = 0.5  
HEURISTIC_PLAYER = 1  #


def pretty_print(board):
    # simple visual: 0 -> ., 1 -> X, -1 -> O
    symbols = {0: ".", 1: "X", -1: "O"}
    for r in range(board.shape[0]):
        print(" ".join(symbols[int(x)] for x in board[r]))
    print()

def play_one_game(starting_player, depth=3, verbose=False):
    env = Connect4Env()
    env.current_player = starting_player
    state = env.reset()
    agent_h = HeuristicAgent(depth=depth)
    agent_r = RandomAgent()

    while True:
        if env.current_player == HEURISTIC_PLAYER:
            action = agent_h.select_action(env)
        else:
            action = agent_r.select_action(env.get_valid_actions())

        state, reward, done = env.step(action)



        if verbose:
            pretty_print(state)
            last_player = -env.current_player
            print(f"Player {'●' if last_player==1 else '○'} played column {action}\n")
            time.sleep(0.3)

        last_player = -env.current_player

        if check_win(state, last_player):
            return last_player  # winner (1 or -1)

        if len(env.get_valid_actions()) == 0:
            return 0  # draw
        

def main():
    stats = {"heuristic": 0, "random": 0, "draw": 0}
    #alternating, reducing bias
    starting = 1
    for g in range(1, NUM_GAMES + 1):
        winner = play_one_game(starting_player=starting, depth=3, verbose=False)

        if winner == HEURISTIC_PLAYER:
            stats["heuristic"] += 1
            result = "Heuristic"
        elif winner == 0:
            stats["draw"] += 1
            result = "Draw"
        else:
            stats["random"] += 1
            result = "Random"

        print(f"Game {g}: Starter={starting} Winner={result}")
        
        starting = -starting

    print("\n=== Summary ===")
    print(f"Heuristic wins: {stats['heuristic']}")
    print(f"Random wins:    {stats['random']}")
    print(f"Draws:          {stats['draw']}")

if __name__ == "__main__":
    main()
