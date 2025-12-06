
"""
Watch two AlphaZero agents (same model) play against each other.
this checks that they create varied board positions.
"""

import sys
import torch
import time
from connect4_env.connect4_env import Connect4Env
from agents.alphazero.az_network import AZNetwork
from agents.alphazero.alphazero_agent import AlphaZeroAgent


def print_board(board):
    
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


def play_game(env, agent1, agent2, temperature1=0.3, temperature2=0.3, 
              render=True, delay=1.0):
   
    env.reset()
    move_count = 0
    
    if render:
        print(f"\n{'='*60}")
        print(f"Game: AlphaZero-1 (X) vs AlphaZero-2 (O)")
        print(f"Temperature: {temperature1} vs {temperature2}")
        print(f"{'='*60}")
        print_board(env.board)
        if delay > 0:
            time.sleep(delay)
    
    while True:
        current_player = env.current_player
        
        # Select agent and temperature
        if current_player == 1:
            action = agent1.select_action(env, temperature=temperature1)
            player_name = "AlphaZero-1 (X)"
        else:
            action = agent2.select_action(env, temperature=temperature2)
            player_name = "AlphaZero-2 (O)"
        
        if render:
            print(f"\nMove {move_count + 1}: {player_name} plays column {action}")
        
        # Store who moved before step changes current_player
        player_who_moved = current_player
        
        # Apply action
        _, reward, done = env.step(action)
        move_count += 1
        
        if render:
            print_board(env.board)
            if delay > 0:
                time.sleep(delay)
        
        # Check win
        if env.check_win(player_who_moved):
            winner = player_who_moved
            if render:
                winner_name = "AlphaZero-1 (X)" if winner == 1 else "AlphaZero-2 (O)"
                print(f"\n{'='*60}")
                print(f"  {winner_name} WINS after {move_count} moves!  ")
                print(f"{'='*60}")
            return winner
        
        # Check draw
        if env.is_draw():
            if render:
                print(f"\n{'='*60}")
                print(f" DRAW after {move_count} moves! ")
                print(f"{'='*60}")
            return 0
        
        if move_count > 42:
            if render:
                print(f"\n{'='*60}")
                print(" Max moves reached - DRAW ")
                print(f"{'='*60}")
            return 0
    
    return 0


def play_multiple_games(env, agent1, agent2, num_games=10, 
                       temperature1=0.3, temperature2=0.3):
    """
    Play multiple games and show statistics.
    """
    print(f"\n{'='*60}")
    print(f"Playing {num_games} games: AlphaZero vs AlphaZero")
    print(f"Temperature: {temperature1} vs {temperature2}")
    print(f"{'='*60}\n")
    
    results = {
        "az1_wins": 0,
        "az2_wins": 0,
        "draws": 0,
        "move_counts": []
    }
    
    for i in range(num_games):
        print(f"\n--- Game {i+1}/{num_games} ---")
        
        # Count moves
        move_count = 0
        env.reset()
        
        while True:
            current_player = env.current_player
            
            if current_player == 1:
                action = agent1.select_action(env, temperature=temperature1)
            else:
                action = agent2.select_action(env, temperature=temperature2)
            
            player_who_moved = current_player
            _, reward, done = env.step(action)
            move_count += 1
            
            if env.check_win(player_who_moved):
                winner = player_who_moved
                break
            
            if env.is_draw() or move_count > 42:
                winner = 0
                break
        
        # Record result
        results["move_counts"].append(move_count)
        if winner == 1:
            results["az1_wins"] += 1
            print(f"Winner: AlphaZero-1 (X) in {move_count} moves")
        elif winner == -1:
            results["az2_wins"] += 1
            print(f"Winner: AlphaZero-2 (O) in {move_count} moves")
        else:
            results["draws"] += 1
            print(f"Result: Draw in {move_count} moves")
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"AlphaZero-1 (X) wins: {results['az1_wins']} ({results['az1_wins']/num_games*100:.1f}%)")
    print(f"AlphaZero-2 (O) wins: {results['az2_wins']} ({results['az2_wins']/num_games*100:.1f}%)")
    print(f"Draws: {results['draws']} ({results['draws']/num_games*100:.1f}%)")
    print(f"Average game length: {sum(results['move_counts'])/len(results['move_counts']):.1f} moves")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    # Load trained AlphaZero model
    print("Loading AlphaZero agent...")
    env = Connect4Env()
    network = AZNetwork()
    
    import os
    model_files = [f for f in os.listdir("logs") if f.startswith("model_iter_") and f.endswith(".pt")]
    if not model_files:
        print(" No trained models found in logs/ folder!")
        print("Please train the model first using: python -m main.train_alphazero")
        sys.exit(1)
    
    # Get latest model
    latest_model = sorted(model_files, key=lambda x: int(x.split("_")[2].split(".")[0]))[-1]
    model_path = os.path.join("logs", latest_model)
    
    print(f"Loading model: {model_path}")
    network.load_state_dict(torch.load(model_path, map_location="cpu"))
    network.eval()
    
    # Create two agents with same network (they share the same brain!)
    agent1 = AlphaZeroAgent(
        env=env,
        network=network,
        num_simulations=100,  # Reduced for speed
        c_puct=1.5,
        device="cpu"
    )
    
    agent2 = AlphaZeroAgent(
        env=env,
        network=network,  # Same network!
        num_simulations=100,
        c_puct=1.5,
        device="cpu"
    )
    
    # Menu
    while True:
        print("\n" + "="*60)
        print("AlphaZero vs AlphaZero (Self-Play Visualization)")
        print("="*60)
        print("1. Watch 1 game (with delay between moves)")
        print("2. Watch 1 game (fast, no delay)")
        print("3. Watch 1 game (high temperature = more random/creative)")
        print("4. Play 10 games (statistics only, no visualization)")
        print("5. Play 50 games (statistics only)")
        print("6. Exit")
        print("="*60)
        
        import sys
        sys.stdout.flush()
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            play_game(env, agent1, agent2, 
                     temperature1=0.3, temperature2=0.3,
                     render=True, delay=1.5)
        
        elif choice == "2":
            play_game(env, agent1, agent2,
                     temperature1=0.3, temperature2=0.3,
                     render=True, delay=0)
        
        elif choice == "3":
            print("\n High temperature mode - more random/creative play!")
            play_game(env, agent1, agent2,
                     temperature1=1.0, temperature2=1.0,
                     render=True, delay=1.5)
        
        elif choice == "4":
            play_multiple_games(env, agent1, agent2, num_games=10,
                              temperature1=0.3, temperature2=0.3)
        
        elif choice == "5":
            play_multiple_games(env, agent1, agent2, num_games=50,
                              temperature1=0.3, temperature2=0.3)
        
        elif choice == "6":
            print("\n Goodbye!")
            break
        
        else:
            print(" Invalid choice. Please enter 1-6.")