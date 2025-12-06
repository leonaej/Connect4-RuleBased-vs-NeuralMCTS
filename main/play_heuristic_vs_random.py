# main/play_heuristic_vs_random.py
import time
from pathlib import Path
import sys


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from connect4_env.connect4_env import Connect4Env
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent


NUM_GAMES = 400
HEURISTIC_PLAYER = 1  # Heuristic plays as player 1


def pretty_print(board):
    """Print Connect4 board nicely."""
    symbols = {0: ".", 1: "●", -1: "○"}
    rows, cols = board.shape
    print("\nBoard:")
    for r in range(rows):
        print(" ".join(symbols[int(cell)] for cell in board[r]))
    print("-" * (2 * cols - 1))


def play_one_game(starting_player, depth=3, verbose=False):
    
    env = Connect4Env()
    env.reset()
    env.current_player = starting_player  # Set who starts
    
    agent_h = HeuristicAgent(depth=depth)
    agent_r = RandomAgent()

    move_count = 0
    max_moves = 42

    while move_count < max_moves:
        # Select action based on current player
        if env.current_player == HEURISTIC_PLAYER:
            action = agent_h.select_action(env)
        else:
            valid_actions = env.get_valid_actions()
            action = agent_r.select_action(valid_actions)

        # Store who moved before the move changes current_player
        player_who_moved = env.current_player
        
        # Make the move
        state, reward, done = env.step(action)
        move_count += 1

        if verbose:
            pretty_print(env.board)
            print(f"Player {'●' if player_who_moved==1 else '○'} played column {action}\n")
            time.sleep(0.3)

        # Check if player who just moved won
        if env.check_win(player_who_moved):
            if verbose:
                winner_symbol = '●' if player_who_moved == 1 else '○'
                print(f"Player {winner_symbol} wins!")
            return player_who_moved  # Return winner (1 or -1)

        # Check for draw
        if env.is_draw():
            if verbose:
                print("Draw!")
            return 0  # Draw

    # If max moves reached (shouldn't happen with is_draw check)
    return 0


def main():
    print("="*70)
    print("HEURISTIC vs RANDOM AGENT")
    print("="*70)
    print(f"Playing {NUM_GAMES} games...")
    print(f"Heuristic is player {HEURISTIC_PLAYER}")
    print("="*70 + "\n")
    
    stats = {"heuristic": 0, "random": 0, "draw": 0}
    
    # Alternate starting player to reduce bias
    starting = 1
    
    for g in range(1, NUM_GAMES + 1):
        winner = play_one_game(starting_player=starting, depth=3, verbose=False)

        # Record result
        if winner == HEURISTIC_PLAYER:
            stats["heuristic"] += 1
            result = "Heuristic"
        elif winner == -HEURISTIC_PLAYER:
            stats["random"] += 1
            result = "Random"
        else:
            stats["draw"] += 1
            result = "Draw"

        # Print progress every 50 games
        if g % 50 == 0 or g == NUM_GAMES:
            print(f"Game {g}/{NUM_GAMES}: Starter={'●' if starting==1 else '○'} "
                  f"Winner={result} | "
                  f"H:{stats['heuristic']} R:{stats['random']} D:{stats['draw']}")
        
        # Alternate starter
        starting = -starting

    # Final summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Heuristic wins: {stats['heuristic']:3d} ({stats['heuristic']/NUM_GAMES*100:.1f}%)")
    print(f"Random wins:    {stats['random']:3d} ({stats['random']/NUM_GAMES*100:.1f}%)")
    print(f"Draws:          {stats['draw']:3d} ({stats['draw']/NUM_GAMES*100:.1f}%)")
    print("="*70)
    
    

if __name__ == "__main__":
    main()