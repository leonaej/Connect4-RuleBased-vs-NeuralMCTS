

import sys
import torch
import os
from connect4_env.connect4_env import Connect4Env
from agents.alphazero.az_network2 import AZNetwork2
from agents.alphazero.alphazero_agent import AlphaZeroAgent
from agents.heuristic_agent import HeuristicAgent


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
        print(f"\n{'='*60}")
        print(f"Game: {agent1_name} (Player 1 = X) vs {agent2_name} (Player -1 = O)")
        print(f"{'='*60}")
        print_board(env.board)
    
    while move_count < 42:
        current_player = env.current_player
        
        # Select action based on current player
        if current_player == 1:
            # Agent1's turn
            if hasattr(agent1, 'network'):
                action = agent1.select_action(env, temperature=0.3)  # Low temp for better play
            else:
                action = agent1.select_action(env)
            player_name = agent1_name
        else:
            # Agent2's turn
            if hasattr(agent2, 'network'):
                action = agent2.select_action(env, temperature=0.3)
            else:
                action = agent2.select_action(env)
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
                print(f"\n{'='*60}")
                print(f"  {winner_name} (Player {winner}) WINS after {move_count} moves!  ")
                print(f"{'='*60}")
            return winner
        
        # Check draw
        if env.is_draw():
            if render:
                print(f"\n{'='*60}")
                print(f" DRAW after {move_count} moves! ")
                print(f"{'='*60}")
            return 0
    
    return 0


def play_tournament(env, agent1, agent2, agent1_name, agent2_name, num_games=10):
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
        "draws": 0
    }
    
    for i in range(num_games):
        print(f"\n--- Game {i+1}/{num_games} ---")
        
        # Alternate who plays first
        if i % 2 == 0:
            # agent1 plays as Player 1 (X), agent2 as Player -1 (O)
            result = play_game(env, agent1, agent2, agent1_name, agent2_name, render=False)
            if result == 1:
                results[agent1_name] += 1
                print(f"Winner: {agent1_name}")
            elif result == -1:
                results[agent2_name] += 1
                print(f"Winner: {agent2_name}")
            else:
                results["draws"] += 1
                print("Result: Draw")
        else:
            # agent2 plays as Player 1 (X), agent1 as Player -1 (O)
            result = play_game(env, agent2, agent1, agent2_name, agent1_name, render=False)
            if result == 1:
                results[agent2_name] += 1
                print(f"Winner: {agent2_name}")
            elif result == -1:
                results[agent1_name] += 1
                print(f"Winner: {agent1_name}")
            else:
                results["draws"] += 1
                print("Result: Draw")
    
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
    # Setup environment
    env = Connect4Env()
    
    # Load trained Network2 BOOSTED
    print("="*70)
    print("Loading AlphaZero Network2 BOOSTED...")
    print("="*70)
    
    network = AZNetwork2()
    
    # Look for BOOSTED models in logs2 folder
    log_folder = "logs2"
    
    if not os.path.exists(log_folder):
        print(f" No {log_folder}/ folder found!")
        print("Please train Network2 first")
        sys.exit(1)
    
    # Look for boosted model files
    model_files = [f for f in os.listdir(log_folder) if f.startswith("model_net2_boosted_iter_") and f.endswith(".pt")]
    
    if not model_files:
        print(f" No trained Network2 BOOSTED models found in {log_folder}/ folder!")
        print("Looking for pattern: model_net2_boosted_iter_*.pt")
        sys.exit(1)
    
    # Sort by iteration number
    model_files_sorted = sorted(model_files, key=lambda x: int(x.split("_")[4].split(".")[0]))
    
    print(f"\nAvailable BOOSTED models:")
    for i, model in enumerate(model_files_sorted[-10:], 1):  # Show last 10
        iter_num = model.split("_")[4].split(".")[0]
        print(f"  {i}. Iteration {iter_num}")
    
    print(f"\nLatest BOOSTED model: {model_files_sorted[-1]}")
    choice = input("Use latest model? (y/n) or enter iteration number: ").strip().lower()
    
    if choice == 'y' or choice == '':
        model_path = os.path.join(log_folder, model_files_sorted[-1])
    elif choice == 'n':
        iter_num = input("Enter iteration number: ").strip()
        model_path = f"{log_folder}/model_net2_boosted_iter_{iter_num}.pt"
        if not os.path.exists(model_path):
            print(f" Model not found: {model_path}")
            sys.exit(1)
    else:
        model_path = f"{log_folder}/model_net2_boosted_iter_{choice}.pt"
        if not os.path.exists(model_path):
            print(f" Model not found: {model_path}")
            sys.exit(1)
    
    print(f"\nLoading: {model_path}")
    network.load_state_dict(torch.load(model_path, map_location="cpu"))
    network.eval()
    print(" BOOSTED Model loaded successfully!")
    
    # Create agents
    print("\n" + "="*70)
    print("Agent Configuration")
    print("="*70)
    
    # AlphaZero agent with more simulations for stronger play
    num_sims = input("Number of MCTS simulations for AlphaZero (default=400, higher=stronger): ").strip()
    num_sims = int(num_sims) if num_sims else 400
    
    alphazero_agent = AlphaZeroAgent(
        env=env,
        network=network,
        num_simulations=num_sims,  # Higher for stronger play
        c_puct=1.5,
        device="cpu"  # Use CPU for playing (GPU not needed for single games)
    )
    print(f" AlphaZero agent created with {num_sims} simulations")
    
    # Heuristic agent with configurable depth
    heur_depth = input("Heuristic search depth (default=4, higher=stronger but slower): ").strip()
    heur_depth = int(heur_depth) if heur_depth else 4
    
    heuristic_agent = HeuristicAgent(depth=heur_depth)
    print(f" Heuristic agent created with depth={heur_depth}")
    
    # Menu
    while True:
        print("\n" + "="*70)
        print("Network2 BOOSTED vs Heuristic Agent")
        print("="*70)
        print("1. Watch 1 game (AlphaZero plays first)")
        print("2. Watch 1 game (Heuristic plays first)")
        print("3. Play tournament (10 games)")
        print("4. Play tournament (50 games)")
        print("5. Play tournament (100 games)")
        print("6. Play custom tournament (specify number)")
        print("7. Load different model")
        print("8. Adjust agent settings")
        print("9. Exit")
        print("="*70)
        
        choice = input("\nEnter choice (1-9): ").strip()
        
        if choice == "1":
            play_game(env, alphazero_agent, heuristic_agent, 
                     "AlphaZero-Net2-BOOSTED", "Heuristic", render=True)
        
        elif choice == "2":
            play_game(env, heuristic_agent, alphazero_agent,
                     "Heuristic", "AlphaZero-Net2-BOOSTED", render=True)
        
        elif choice == "3":
            play_tournament(env, alphazero_agent, heuristic_agent,
                          "AlphaZero-Net2-BOOSTED", "Heuristic", num_games=10)
        
        elif choice == "4":
            play_tournament(env, alphazero_agent, heuristic_agent,
                          "AlphaZero-Net2-BOOSTED", "Heuristic", num_games=50)
        
        elif choice == "5":
            play_tournament(env, alphazero_agent, heuristic_agent,
                          "AlphaZero-Net2-BOOSTED", "Heuristic", num_games=100)
        
        elif choice == "6":
            num_games = input("Enter number of games: ").strip()
            try:
                num_games = int(num_games)
                if num_games > 0:
                    play_tournament(env, alphazero_agent, heuristic_agent,
                                  "AlphaZero-Net2-BOOSTED", "Heuristic", num_games=num_games)
                else:
                    print(" Number of games must be positive")
            except ValueError:
                print(" Invalid number")
        
        elif choice == "7":
            # Reload model
            iter_num = input("Enter iteration number: ").strip()
            model_path = f"{log_folder}/model_net2_boosted_iter_{iter_num}.pt"
            if os.path.exists(model_path):
                network.load_state_dict(torch.load(model_path, map_location="cpu"))
                network.eval()
                alphazero_agent.network = network
                print(f" Loaded BOOSTED model from iteration {iter_num}")
            else:
                print(f" Model not found: {model_path}")
        
        elif choice == "8":
            # Adjust settings
            print("\nCurrent settings:")
            print(f"  AlphaZero MCTS simulations: {alphazero_agent.num_simulations}")
            print(f"  Heuristic depth: {heuristic_agent.depth}")
            
            new_sims = input("\nNew MCTS simulations (press Enter to keep current): ").strip()
            if new_sims:
                alphazero_agent.num_simulations = int(new_sims)
                print(f"✓ Updated to {new_sims} simulations")
            
            new_depth = input("New heuristic depth (press Enter to keep current): ").strip()
            if new_depth:
                heuristic_agent.depth = int(new_depth)
                print(f"✓ Updated to depth {new_depth}")
        
        elif choice == "9":
            print("\n Goodbye!")
            break
        
        else:
            print(" Invalid choice. Please enter 1-9.")
