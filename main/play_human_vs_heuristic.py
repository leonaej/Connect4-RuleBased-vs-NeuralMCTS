import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import math
import time

from connect4_env.connect4_env import Connect4Env
from agents.heuristic_agent import HeuristicAgent


def print_board(board):
    symbols = {0: ".", 1: "●", -1: "○"}
    for row in board:
        print(" ".join(symbols[x] for x in row))
    print()

def main():
    env = Connect4Env()
    agent = HeuristicAgent(depth=3)

    print("You are ○ (player -1). Heuristic is ● (player 1)")
    print("Columns go from 0 to 6.\n")
    
    # Ask who starts
    start_choice = input("Do you want to start first? (y/n): ").lower()
    if start_choice == 'y':
        env.current_player = -1  # You start
    else:
        env.current_player = 1   # Heuristic starts
    
    print_board(env.board)

    while True:
        
        if env.current_player == -1:
            valid = env.get_valid_actions()
            move = None

            while move not in valid:
                try:
                    move = int(input(f"Your move (choose column {valid}): "))
                except:
                    move = None

            env.step(move)
            print("\nYou played:", move)
            print_board(env.board)

            if env.check_win(-1):
                print("\n  YOU WIN!")
                break
            if env.is_draw():
                print("\nIt's a draw.")
                break


        else:
            print("Heuristic thinking...")
            move = agent.select_action(env)
            env.step(move)

            print("\nHeuristic played:", move)
            print_board(env.board)

            if env.check_win(1):
                print("\nHeuristic wins.")
                break
            if env.is_draw():
                print("\nIt's a draw.")
                break


if __name__ == "__main__":
    main()