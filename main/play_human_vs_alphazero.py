
import torch
import os
import sys
from connect4_env.connect4_env import Connect4Env
from agents.alphazero.az_network2 import AZNetwork2
from agents.alphazero.alphazero_agent import AlphaZeroAgent


def print_board(board):
    """Pretty print the board"""
    symbols = {0: ".", 1: "●", -1: "○"}
    print("\n  0 1 2 3 4 5 6")
    print(" ┌─────────────┐")
    for row in board:
        print(" │", end="")
        for cell in row:
            print(f" {symbols[cell]}", end="")
        print(" │")
    print(" └─────────────┘")
    print()


def main():
    print("="*70)
    print("HUMAN vs ALPHAZERO")
    print("="*70)
    
    env = Connect4Env()
    
    # Load AlphaZero network
    print("\nLoading AlphaZero network...")
    
    if not os.path.exists("logs2"):
        print(" Error: logs2/ folder not found!")
        print("Please make sure you have trained a model first.")
        sys.exit(1)
    
    # Find latest boosted model
    checkpoints = [f for f in os.listdir("logs2") if f.startswith("model_net2_boosted_iter_") and f.endswith(".pt")]
    
    if not checkpoints:
        print(" Error: No trained models found in logs2/!")
        print("Please train a model first using train_net2_with_heuristic_boost.py")
        sys.exit(1)
    
    # Get latest checkpoint
    latest = sorted(checkpoints, key=lambda x: int(x.split("_")[4].split(".")[0]))[-1]
    checkpoint_path = os.path.join("logs2", latest)
    iteration = latest.split("_")[4].split(".")[0]
    
    print(f"✓ Loading: {latest}")
    print(f"  Iteration: {iteration}")
    
    network = AZNetwork2()
    network.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    network.eval()
    
    alphazero_agent = AlphaZeroAgent(
        env=env,
        network=network,
        num_simulations=200,  # More simulations for stronger play
        c_puct=1.5,
        device="cpu"
    )
    
    print("✓ AlphaZero loaded!\n")
    
    # Choose who plays first
    print("="*70)
    choice = input("Do you want to play first? (y/n): ").strip().lower()
    
    if choice == 'y':
        human_player = 1  # Human is ● (player 1)
        ai_player = -1    # AI is ○ (player -1)
        print("\nYou are ● (player 1)")
        print("AlphaZero is ○ (player -1)")
    else:
        human_player = -1  # Human is ○ (player -1)
        ai_player = 1      # AI is ● (player 1)
        print("\nYou are ○ (player -1)")
        print("AlphaZero is ● (player 1)")
    
    print("="*70)
    
    # Start game
    env.reset()
    print_board(env.board)
    
    move_count = 0
    max_moves = 42
    
    while move_count < max_moves:
        current_player = env.current_player
        
        if current_player == human_player:
            # Human's turn
            valid_moves = env.get_valid_actions()
            move = None
            
            while move not in valid_moves:
                try:
                    move_input = input(f"Your move (choose column {valid_moves}): ").strip()
                    move = int(move_input)
                    if move not in valid_moves:
                        print(f"Invalid move! Choose from {valid_moves}")
                except ValueError:
                    print("Please enter a number!")
                except KeyboardInterrupt:
                    print("\n\nGame interrupted. Goodbye!")
                    return
            
            print(f"\nYou played: Column {move}")
        
        else:
            # AlphaZero's turn
            print("AlphaZero is thinking...")
            move = alphazero_agent.select_action(env, temperature=0.1)
            print(f"\nAlphaZero played: Column {move}")
        
        # Store who moved
        player_who_moved = current_player
        
        # Make the move
        env.step(move)
        move_count += 1
        
        # Show board
        print_board(env.board)
        
        # Check for win
        if env.check_win(player_who_moved):
            print("="*70)
            if player_who_moved == human_player:
                print(" YOU WIN! Congratulations! ")
            else:
                print(" AlphaZero wins!")
            print(f"Game ended in {move_count} moves")
            print("="*70)
            break
        
        # Check for draw
        if env.is_draw():
            print("="*70)
            print(" It's a DRAW!")
            print(f"Game ended in {move_count} moves")
            print("="*70)
            break
    
    # Ask to play again
    print()
    play_again = input("Play again? (y/n): ").strip().lower()
    if play_again == 'y':
        main()
    else:
        print("\nGoodbye")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGame interrupted. Goodbye")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()