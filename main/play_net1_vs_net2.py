# main/play_net1_vs_net2.py
"""
Compare Network1 (original) vs Network2 (improved with skip connections)
"""

import sys
import torch
import os
from connect4_env.connect4_env import Connect4Env
from agents.alphazero.az_network import AZNetwork  # Original Network1
from agents.alphazero.az_network2 import AZNetwork2  # Improved Network2
from agents.alphazero.alphazero_agent import AlphaZeroAgent


def print_board(board):
    """Pretty print the board"""
    print("\n  0 1 2 3 4 5 6")
    print(" ┌─────────────┐")
    for row in board:
        print(" │", end="")
        for cell in row:
            if cell == 1:
                print(" X", end="")
            elif cell == -1:
                print(" O", end="")
            else:
                print(" .", end="")
        print(" │")
    print(" └─────────────┘")


def play_game(env, agent1, agent2, agent1_name="Agent1", agent2_name="Agent2", render=True):
    """
    Play one game between two agents.
    Returns: 1 if agent1 wins, -1 if agent2 wins, 0 for draw
    """
    env.reset()
    move_count = 0
    
    if render:
        print(f"\n{'='*70}")
        print(f"Game: {agent1_name} (X) vs {agent2_name} (O)")
        print(f"{'='*70}")
        print_board(env.board)
    
    while move_count < 42:
        current_player = env.current_player
        
        # Select action based on current player
        if current_player == 1:
            action = agent1.select_action(env, temperature=0.1)
            player_name = agent1_name
        else:
            action = agent2.select_action(env, temperature=0.1)
            player_name = agent2_name
        
        if render:
            print(f"\n{player_name} (Player {current_player}) plays column: {action}")
        
        # Store who moved
        player_who_moved = current_player
        
        # Apply action
        _, reward, done = env.step(action)
        move_count += 1
        
        if render:
            print_board(env.board)
        
        # Check win
        if env.check_win(player_who_moved):
            winner = player_who_moved
            if render:
                winner_name = agent1_name if winner == 1 else agent2_name
                print(f"\n{'='*70}")
                print(f" {winner_name} WINS after {move_count} moves!  ")
                print(f"{'='*70}")
            return winner
        
        # Check draw
        if env.is_draw():
            if render:
                print(f"\n{'='*70}")
                print(f" DRAW after {move_count} moves! ")
                print(f"{'='*70}")
            return 0
    
    return 0


def play_tournament(env, agent1, agent2, agent1_name, agent2_name, num_games=20):
    """
    Play multiple games alternating starting player.
    """
    print(f"\n{'='*70}")
    print(f"TOURNAMENT: {num_games} games")
    print(f"{agent1_name} vs {agent2_name}")
    print(f"{'='*70}\n")
    
    results = {
        agent1_name: 0,
        agent2_name: 0,
        "draws": 0,
        "game_lengths": []
    }
    
    for i in range(num_games):
        print(f"Game {i+1}/{num_games}...", end=" ")
        
        # Alternate who plays first
        if i % 2 == 0:
            result = play_game(env, agent1, agent2, agent1_name, agent2_name, render=False)
            if result == 1:
                results[agent1_name] += 1
                print(f"Winner: {agent1_name}")
            elif result == -1:
                results[agent2_name] += 1
                print(f"Winner: {agent2_name}")
            else:
                results["draws"] += 1
                print("Draw")
        else:
            result = play_game(env, agent2, agent1, agent2_name, agent1_name, render=False)
            if result == 1:
                results[agent2_name] += 1
                print(f"Winner: {agent2_name}")
            elif result == -1:
                results[agent1_name] += 1
                print(f"Winner: {agent1_name}")
            else:
                results["draws"] += 1
                print("Draw")
    
    # Print final results
    print(f"\n{'='*70}")
    print("TOURNAMENT RESULTS")
    print(f"{'='*70}")
    print(f"{agent1_name}: {results[agent1_name]} wins ({results[agent1_name]/num_games*100:.1f}%)")
    print(f"{agent2_name}: {results[agent2_name]} wins ({results[agent2_name]/num_games*100:.1f}%)")
    print(f"Draws: {results['draws']} ({results['draws']/num_games*100:.1f}%)")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    print("="*70)
    print("NETWORK COMPARISON: Net1 (Original) vs Net2 (Improved)")
    print("="*70)
    
    env = Connect4Env()
    
    # ===================================================================
    # Load Network1 (Original from logs/)
    # ===================================================================
    print("\n" + "-"*70)
    print("Loading Network1 (Original)...")
    print("-"*70)
    
    if not os.path.exists("logs"):
        print(" No logs/ folder found!")
        sys.exit(1)
    
    net1_files = [f for f in os.listdir("logs") if f.startswith("model_iter_") and f.endswith(".pt")]
    if not net1_files:
        print(" No Network1 models found in logs/!")
        sys.exit(1)
    
    latest_net1 = sorted(net1_files, key=lambda x: int(x.split("_")[2].split(".")[0]))[-1]
    net1_path = os.path.join("logs", latest_net1)
    net1_iter = latest_net1.split("_")[2].split(".")[0]
    
    print(f"Found: {latest_net1}")
    print(f"Iteration: {net1_iter}")
    
    network1 = AZNetwork()
    network1.load_state_dict(torch.load(net1_path, map_location="cpu"))
    network1.eval()
    print("✓ Network1 loaded!")
    
    agent1 = AlphaZeroAgent(
        env=env,
        network=network1,
        num_simulations=200,
        c_puct=1.5,
        device="cpu"
    )
    
    # ===================================================================
    # Load Network2 (Improved from logs2/)
    # ===================================================================
    print("\n" + "-"*70)
    print("Loading Network2 (Improved)...")
    print("-"*70)
    
    if not os.path.exists("logs2"):
        print(" No logs2/ folder found!")
        sys.exit(1)
    
    net2_files = [f for f in os.listdir("logs2") if f.startswith("model_net2_boosted_iter_") and f.endswith(".pt")]
    if not net2_files:
        print(" No Network2 models found in logs2/!")
        sys.exit(1)
    
    # Fixed: The filename is "model_net2_boosted_iter_X.pt"
    # Split by "_" gives: ["model", "net2", "boosted", "iter", "X.pt"]
    # So index 4 contains "X.pt", and we need to split by "." to get X
    latest_net2 = sorted(net2_files, key=lambda x: int(x.split("_")[4].split(".")[0]))[-1]
    net2_path = os.path.join("logs2", latest_net2)
    net2_iter = latest_net2.split("_")[4].split(".")[0]
    
    print(f"Found: {latest_net2}")
    print(f"Iteration: {net2_iter}")
    
    network2 = AZNetwork2()
    network2.load_state_dict(torch.load(net2_path, map_location="cpu"))
    network2.eval()
    print("✓ Network2 loaded!")
    
    agent2 = AlphaZeroAgent(
        env=env,
        network=network2,
        num_simulations=200,
        c_puct=1.5,
        device="cpu"
    )
    
    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "="*70)
    print("MATCHUP SUMMARY")
    print("="*70)
    print(f"Network1: Iteration {net1_iter} (trained via self-play)")
    print(f"  - Architecture: 3 conv blocks (no skip connections)")
    print(f"  - Training: Self-play against itself")
    print(f"\nNetwork2: Iteration {net2_iter} (trained vs heuristic)")
    print(f"  - Architecture: 4 residual blocks (WITH skip connections)")
    print(f"  - Training: Playing against heuristic agent")
    print("="*70)
    
    # ===================================================================
    # Menu
    # ===================================================================
    while True:
        print("\n" + "="*70)
        print("Network1 vs Network2 Comparison")
        print("="*70)
        print("1. Watch 1 game (Net1 plays first)")
        print("2. Watch 1 game (Net2 plays first)")
        print("3. Tournament (10 games)")
        print("4. Tournament (50 games)")
        print("5. Tournament (100 games)")
        print("6. Exit")
        print("="*70)
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            play_game(env, agent1, agent2, 
                     f"Net1-iter{net1_iter}", f"Net2-iter{net2_iter}", render=True)
        
        elif choice == "2":
            play_game(env, agent2, agent1,
                     f"Net2-iter{net2_iter}", f"Net1-iter{net1_iter}", render=True)
        
        elif choice == "3":
            play_tournament(env, agent1, agent2,
                          f"Net1-iter{net1_iter}", f"Net2-iter{net2_iter}", num_games=10)
        
        elif choice == "4":
            play_tournament(env, agent1, agent2,
                          f"Net1-iter{net1_iter}", f"Net2-iter{net2_iter}", num_games=50)
        
        elif choice == "5":
            play_tournament(env, agent1, agent2,
                          f"Net1-iter{net1_iter}", f"Net2-iter{net2_iter}", num_games=100)
        
        elif choice == "6":
            print("\n Goodbye!")
            break
        
        else:
            print(" Invalid choice. Please enter 1-6.")