

import sys
import torch
import os
from connect4_env.connect4_env import Connect4Env
from agents.alphazero.az_network2 import AZNetwork2
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
            action = agent1.select_action(env, temperature=0.1)  # Low temp for better play
            player_name = agent1_name
        else:
            # Agent2's turn
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


def load_model(model_path, model_name):
    """Load a model and return the network"""
    if not os.path.exists(model_path):
        print(f" Model not found: {model_path}")
        return None
    
    network = AZNetwork2()
    network.load_state_dict(torch.load(model_path, map_location="cpu"))
    network.eval()
    print(f" Loaded {model_name} from: {model_path}")
    return network


if __name__ == "__main__":
    # Setup environment
    env = Connect4Env()
    
    print("="*70)
    print("Network2 Regular vs Network2 Boosted")
    print("="*70)
    
    # Check if logs2 folder exists
    if not os.path.exists("logs2"):
        print(" No logs2/ folder found!")
        sys.exit(1)
    
    # Get all model files
    all_files = os.listdir("logs2")
    regular_files = [f for f in all_files if f.startswith("model_net2_iter_") and not "boosted" in f]
    boosted_files = [f for f in all_files if f.startswith("model_net2_boosted_iter_")]
    
    if not regular_files:
        print(" No regular Network2 models found!")
        sys.exit(1)
    
    if not boosted_files:
        print(" No boosted Network2 models found!")
        sys.exit(1)
    
    # Sort files
    regular_files_sorted = sorted(regular_files, key=lambda x: int(x.split("_")[3].split(".")[0]))
    boosted_files_sorted = sorted(boosted_files, key=lambda x: int(x.split("_")[4].split(".")[0]))
    
    print("\n" + "="*70)
    print("STEP 1: Select REGULAR Network2 Model")
    print("="*70)
    print(f"\nAvailable regular models (last 10):")
    for i, model in enumerate(regular_files_sorted[-10:], 1):
        iter_num = model.split("_")[3].split(".")[0]
        print(f"  {i}. Iteration {iter_num}")
    
    # Default to iteration 60
    print(f"\nDefault: Iteration 60")
    choice1 = input("Enter iteration number (or press Enter for 60): ").strip()
    
    if choice1 == '':
        model1_path = "logs2/model_net2_iter_60.pt"
        model1_name = "Net2-Regular-Iter60"
    else:
        model1_path = f"logs2/model_net2_iter_{choice1}.pt"
        model1_name = f"Net2-Regular-Iter{choice1}"
    
    network1 = load_model(model1_path, model1_name)
    if network1 is None:
        sys.exit(1)
    
    print("\n" + "="*70)
    print("STEP 2: Select BOOSTED Network2 Model")
    print("="*70)
    print(f"\nAvailable boosted models (last 10):")
    for i, model in enumerate(boosted_files_sorted[-10:], 1):
        iter_num = model.split("_")[4].split(".")[0]
        print(f"  {i}. Iteration {iter_num}")
    
    latest_boosted = boosted_files_sorted[-1]
    latest_iter = latest_boosted.split("_")[4].split(".")[0]
    print(f"\nLatest: Iteration {latest_iter}")
    choice2 = input(f"Enter iteration number (or press Enter for latest={latest_iter}): ").strip()
    
    if choice2 == '':
        model2_path = f"logs2/{latest_boosted}"
        model2_name = f"Net2-Boosted-Iter{latest_iter}"
    else:
        model2_path = f"logs2/model_net2_boosted_iter_{choice2}.pt"
        model2_name = f"Net2-Boosted-Iter{choice2}"
    
    network2 = load_model(model2_path, model2_name)
    if network2 is None:
        sys.exit(1)
    
    # Create agents
    print("\n" + "="*70)
    print("Agent Configuration")
    print("="*70)
    
    num_sims = input("Number of MCTS simulations for both agents (default=200): ").strip()
    num_sims = int(num_sims) if num_sims else 200
    
    agent1 = AlphaZeroAgent(
        env=env,
        network=network1,
        num_simulations=num_sims,
        c_puct=1.5,
        device="cpu"
    )
    
    agent2 = AlphaZeroAgent(
        env=env,
        network=network2,
        num_simulations=num_sims,
        c_puct=1.5,
        device="cpu"
    )
    
    print(f"✓ Both agents created with {num_sims} simulations")
    
    # Menu
    while True:
        print("\n" + "="*70)
        print(f"{model1_name} vs {model2_name}")
        print("="*70)
        print(f"1. Watch 1 game ({model1_name} plays first)")
        print(f"2. Watch 1 game ({model2_name} plays first)")
        print("3. Play tournament (10 games, alternating)")
        print("4. Play tournament (50 games, alternating)")
        print("5. Play tournament (100 games, alternating)")
        print("6. Play custom tournament")
        print("7. Load different models")
        print("8. Adjust MCTS simulations")
        print("9. Exit")
        print("="*70)
        
        choice = input("\nEnter choice (1-9): ").strip()
        
        if choice == "1":
            play_game(env, agent1, agent2, model1_name, model2_name, render=True)
        
        elif choice == "2":
            play_game(env, agent2, agent1, model2_name, model1_name, render=True)
        
        elif choice == "3":
            play_tournament(env, agent1, agent2, model1_name, model2_name, num_games=10)
        
        elif choice == "4":
            play_tournament(env, agent1, agent2, model1_name, model2_name, num_games=50)
        
        elif choice == "5":
            play_tournament(env, agent1, agent2, model1_name, model2_name, num_games=100)
        
        elif choice == "6":
            num_games = input("Enter number of games: ").strip()
            try:
                num_games = int(num_games)
                if num_games > 0:
                    play_tournament(env, agent1, agent2, model1_name, model2_name, num_games=num_games)
                else:
                    print(" Number of games must be positive")
            except ValueError:
                print(" Invalid number")
        
        elif choice == "7":
            print("\n--- Reload Models ---")
            
            # Reload model 1
            print("\nModel 1 (Regular):")
            iter1 = input(f"Enter iteration (current={model1_name}): ").strip()
            if iter1:
                new_path1 = f"logs2/model_net2_iter_{iter1}.pt"
                new_net1 = load_model(new_path1, f"Net2-Regular-Iter{iter1}")
                if new_net1:
                    network1 = new_net1
                    agent1.network = network1
                    model1_name = f"Net2-Regular-Iter{iter1}"
            
            # Reload model 2
            print("\nModel 2 (Boosted):")
            iter2 = input(f"Enter iteration (current={model2_name}): ").strip()
            if iter2:
                new_path2 = f"logs2/model_net2_boosted_iter_{iter2}.pt"
                new_net2 = load_model(new_path2, f"Net2-Boosted-Iter{iter2}")
                if new_net2:
                    network2 = new_net2
                    agent2.network = network2
                    model2_name = f"Net2-Boosted-Iter{iter2}"
        
        elif choice == "8":
            new_sims = input(f"New MCTS simulations (current={num_sims}): ").strip()
            if new_sims:
                num_sims = int(new_sims)
                agent1.num_simulations = num_sims
                agent2.num_simulations = num_sims
                print(f"✓ Updated both agents to {num_sims} simulations")
        
        elif choice == "9":
            print("\n Goodbye!")
            break
        
        else:
            print(" Invalid choice. Please enter 1-9.")