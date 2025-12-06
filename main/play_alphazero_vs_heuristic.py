import sys
import torch
from connect4_env.connect4_env import Connect4Env
from agents.alphazero.az_network import AZNetwork
from agents.alphazero.alphazero_agent import AlphaZeroAgent
from agents.heuristic_agent import HeuristicAgent


def play_game(env, agent1, agent2, agent1_name="Agent1", agent2_name="Agent2", render=True, temperature=0.0):
    """
    Play one game between two agents.
    Returns: 1 if agent1 wins, -1 if agent2 wins, 0 for draw
    """
    env.reset()
    done = False
    move_count = 0
    
    if render:
        print(f"\n{'='*50}")
        print(f"Game: {agent1_name} (Player 1) vs {agent2_name} (Player -1)")
        print(f"{'='*50}\n")
        print_board(env.board)
    
    while not done:
        current_player = env.current_player
        
        # Select action based on current player
        if current_player == 1:
            # Check if agent is AlphaZero (has network attribute)
            if hasattr(agent1, 'network'):
                action = agent1.select_action(env, temperature=temperature)
            else:
                action = agent1.select_action(env)
            player_name = agent1_name
        else:
            # Check if agent is AlphaZero (has network attribute)
            if hasattr(agent2, 'network'):
                action = agent2.select_action(env, temperature=temperature)
            else:
                action = agent2.select_action(env)
            player_name = agent2_name
        
        if render:
            print(f"\n{player_name} (Player {current_player}) plays column: {action}")
        
        # Store who just moved before step changes current_player
        player_who_moved = current_player
        
        # Apply action
        _, reward, done = env.step(action)
        move_count += 1
        
        if render:
            print_board(env.board)
        
        # Check if the player who just moved won
        if env.check_win(player_who_moved):
            winner = player_who_moved
            if render:
                winner_name = agent1_name if winner == 1 else agent2_name
                print(f"\n  {winner_name} (Player {winner}) WINS!  ")
            return winner
        
        # Check draw
        if env.is_draw():
            if render:
                print("\n DRAW! ")
            return 0
        
        if move_count > 42:  # Safety check
            if render:
                print("\n Max moves reached - DRAW ⚠️")
            return 0
    
    return 0


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


def play_tournament(env, agent1, agent2, agent1_name, agent2_name, num_games=10, temperature=0.1):
    """
    Play multiple games alternating starting player.
    """
    print(f"\n{'='*60}")
    print(f"TOURNAMENT: {num_games} games (temperature={temperature})")
    print(f"{agent1_name} vs {agent2_name}")
    print(f"{'='*60}\n")
    
    results = {
        agent1_name: 0,
        agent2_name: 0,
        "draws": 0
    }
    
    for i in range(num_games):
        print(f"\n--- Game {i+1}/{num_games} ---")
        
        # IMPORTANT: Reset the environment before each game
        env.reset()
        
        # Alternate who plays first
        if i % 2 == 0:
            # agent1 plays as Player 1, agent2 as Player -1
            print(f"{agent1_name} is Player 1, {agent2_name} is Player -1")
            result = play_game(env, agent1, agent2, agent1_name, agent2_name, render=False, temperature=temperature)
            print(f"Raw result: {result}")
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
            # agent2 plays as Player 1, agent1 as Player -1
            print(f"{agent2_name} is Player 1, {agent1_name} is Player -1")
            result = play_game(env, agent2, agent1, agent2_name, agent1_name, render=False, temperature=temperature)
            print(f"Raw result: {result}")
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
    print(f"\n{'='*60}")
    print("TOURNAMENT RESULTS")
    print(f"{'='*60}")
    print(f"{agent1_name}: {results[agent1_name]} wins ({results[agent1_name]/num_games*100:.1f}%)")
    print(f"{agent2_name}: {results[agent2_name]} wins ({results[agent2_name]/num_games*100:.1f}%)")
    print(f"Draws: {results['draws']} ({results['draws']/num_games*100:.1f}%)")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    # Setup environment
    env = Connect4Env()
    
    # Load trained AlphaZero agent
    print("Loading AlphaZero agent...")
    network = AZNetwork()
    
    # Try to load the latest model
    import os
    model_files = [f for f in os.listdir("logs") if f.startswith("model_iter_") and f.endswith(".pt")]
    if not model_files:
        print(" No trained models found in logs/ folder!")
        print("Please train the model first using: python -m main.train_alphazero")
        sys.exit(1)
    
    # Get the latest iteration model
    latest_model = sorted(model_files, key=lambda x: int(x.split("_")[2].split(".")[0]))[-1]
    model_path = os.path.join("logs", latest_model)
    print(f"Found model: {latest_model}")
    
    print(f"Loading model: {model_path}")
    network.load_state_dict(torch.load(model_path, map_location="cpu"))
    network.eval()
    
    alphazero_agent = AlphaZeroAgent(
        env=env,
        network=network,
        num_simulations=200,  # Use full strength
        c_puct=1.5,
        device="cpu"
    )
    
    # Create heuristic agent
    print("Creating Heuristic agent...")
    heuristic_agent = HeuristicAgent()
    
    # Menu
    while True:
        print("\n" + "="*60)
        print("AlphaZero vs Heuristic Agent")
        print("="*60)
        print("1. Play single game (with visualization)")
        print("2. Play tournament (10 games)")
        print("3. Play tournament (50 games)")
        print("4. Exit")
        print("="*60)
        
        import sys
        sys.stdout.flush()
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            env.reset()  # Ensure clean state
            # Use small temperature for variety in demo games
            play_game(env, alphazero_agent, heuristic_agent, 
                     "AlphaZero", "Heuristic", render=True, temperature=0.5)
        
        elif choice == "2":
            env.reset()  # Ensure clean state
            play_tournament(env, alphazero_agent, heuristic_agent,
                          "AlphaZero", "Heuristic", num_games=10)
        
        elif choice == "3":
            env.reset()  # Ensure clean state
            play_tournament(env, alphazero_agent, heuristic_agent,
                          "AlphaZero", "Heuristic", num_games=50)
        
        elif choice == "4":
            print("\n Goodbye!")
            break
        
        else:
            print(" Invalid choice. Please enter 1-4.")
