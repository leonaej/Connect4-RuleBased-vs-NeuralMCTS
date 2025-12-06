

import numpy as np
import torch
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from connect4_env.connect4_env import Connect4Env
from agents.alphazero.az_network import AZNetwork
from agents.alphazero.alphazero_agent import AlphaZeroAgent
from agents.heuristic_agent import HeuristicAgent
from training.replay_buffer import ReplayBuffer
from training.train_one_iter import train_one_iteration


def encode_board(board, current_player):
    
    p1 = (board == current_player).astype(np.float32)
    p2 = (board == -current_player).astype(np.float32)
    stacked = np.stack([p1, p2], axis=0)
    return torch.tensor(stacked, dtype=torch.float32)


def play_game_vs_heuristic(env, alphazero_agent, heuristic_agent, 
                           buffer, alphazero_player, temperature=1.0):
    
    env.reset()
    game_states = []  # (state, policy, player)
    
    # If AlphaZero should go second, let heuristic make first move
    if alphazero_player == -1:
        # Heuristic plays first
        action = heuristic_agent.select_action(env)
        if action is None or action not in env.get_valid_actions():
            # Fallback to first valid move
            action = env.get_valid_actions()[0]
        env.step(int(action))
    
    # Main game loop
    move_count = 0
    max_moves = 42
    
    while move_count < max_moves:
        board = env.board.copy()
        current_player = env.current_player
        
        # Determine which agent moves
        if current_player == alphazero_player:
            # AlphaZero's turn
            action, pi = alphazero_agent.policy_from_root(
                board, current_player, temperature=temperature
            )
            # Ensure action is a Python int (not tensor)
            if hasattr(action, 'item'):
                action = action.item()
            action = int(action)
            
            # Store this state for training
            encoded_state = encode_board(board, current_player)
            game_states.append((encoded_state, pi, current_player))
        else:
            # Heuristic's turn
            action = heuristic_agent.select_action(env)
            if action is None or action not in env.get_valid_actions():
                # Fallback: pick first valid move
                valid_moves = env.get_valid_actions()
                if not valid_moves:
                    winner = 0  # Draw
                    break
                action = valid_moves[0]
            action = int(action)
        
        # Store who moved before step changes player
        player_who_moved = current_player
        
        # Apply action
        _, reward, done = env.step(action)
        move_count += 1
        
        # Check win (player who just moved)
        if env.check_win(player_who_moved):
            winner = player_who_moved
            break
        
        # Check draw
        if env.is_draw():
            winner = 0
            break
    else:
        # Max moves reached
        winner = 0
    
    # Label AlphaZero's moves with outcome
    for state, pi, player in game_states:
        if winner == 0:
            value = 0.0
        else:
            # From AlphaZero's perspective
            value = 1.0 if winner == alphazero_player else -1.0
        buffer.push(state, pi, value)
    
    return winner


def evaluate_vs_heuristic(env, alphazero_agent, heuristic_agent, num_games=20):
    """
    Evaluate AlphaZero against Heuristic without training.
    Returns win/loss/draw counts.
    """
    results = {"az_wins": 0, "heuristic_wins": 0, "draws": 0}
    
    for i in range(num_games):
        # Alternate starting player
        alphazero_player = 1 if i % 2 == 0 else -1
        winner = play_game_vs_heuristic(
            env, alphazero_agent, heuristic_agent,
            buffer=ReplayBuffer(capacity=1),  # Dummy buffer
            alphazero_player=alphazero_player,
            temperature=0.1  # Low temp for evaluation
        )
        
        if winner == alphazero_player:
            results["az_wins"] += 1
        elif winner == -alphazero_player:
            results["heuristic_wins"] += 1
        else:
            results["draws"] += 1
    
    return results


if __name__ == "__main__":
    print("="*70)
    print("Training AlphaZero Against Heuristic Agent")
    print("="*70)
    
    env = Connect4Env()
    network = AZNetwork()
    
    import os
    
    # Look for existing self-play trained models
    model_files = [f for f in os.listdir("logs") if f.startswith("model_iter_") and f.endswith(".pt")]
    
    if model_files:
        # Get the latest self-play model
        latest_model = sorted(model_files, key=lambda x: int(x.split("_")[2].split(".")[0]))[-1]
        model_path = os.path.join("logs", latest_model)
        
        print(f"\n✓ Found existing model: {latest_model}")
        choice = input("Load this model and continue training vs Heuristic? (y/n): ").strip().lower()
        
        if choice == 'y':
            print(f"Loading weights from {model_path}...")
            network.load_state_dict(torch.load(model_path, map_location="cpu"))
            print("✓ Loaded! Will now fine-tune against Heuristic.\n")
        else:
            print("Starting with random weights.\n")
    else:
        print("\nNo existing models found. Starting with random weights.\n")
    
    alphazero_agent = AlphaZeroAgent(
        env=env, 
        network=network, 
        num_simulations=100,  # Reduced for speed
        c_puct=1.5, 
        device="cuda"  # Using GPU for 3-10x speedup!
    )
    heuristic_agent = HeuristicAgent(depth=3)
    buffer = ReplayBuffer(capacity=20000)
    
    os.makedirs("logs", exist_ok=True)
    
    # Training settings
    num_iterations = 50
    games_per_iteration = 50
    
    # History tracking
    reward_history = []
    loss_history = []
    eval_history = []
    
    
    for iteration in range(num_iterations):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration+1}/{num_iterations}")
        print(f"{'='*70}")
        
        print(f"\nPlaying {games_per_iteration} games vs Heuristic...")
        
        az_wins = 0
        heuristic_wins = 0
        draws = 0
        
        # Temperature schedule: high early, low later
        temp = 1.0 if iteration < 15 else 0.3
        
        for g in tqdm(range(games_per_iteration), desc="Games", leave=False):
            # Alternate who plays first
            alphazero_player = 1 if g % 2 == 0 else -1
            
            winner = play_game_vs_heuristic(
                env, alphazero_agent, heuristic_agent,
                buffer, alphazero_player, temperature=temp
            )
            
            if winner == alphazero_player:
                az_wins += 1
            elif winner == -alphazero_player:
                heuristic_wins += 1
            else:
                draws += 1
        
        az_winrate = az_wins / games_per_iteration
        print(f"\nGame Results:")
        print(f"  AlphaZero wins: {az_wins} ({az_winrate*100:.1f}%)")
        print(f"  Heuristic wins: {heuristic_wins} ({heuristic_wins/games_per_iteration*100:.1f}%)")
        print(f"  Draws: {draws} ({draws/games_per_iteration*100:.1f}%)")
        print(f"  Buffer size: {len(buffer)}")
        
        reward_history.append(az_winrate)
        
        if len(buffer) >= 64:
            print(f"\nTraining network...")
            loss = train_one_iteration(alphazero_agent, buffer, batch_size=64, steps=100)
            loss_history.append(loss)
            print(f"  Training loss: {loss:.4f}")
        else:
            print(f"\nSkipping training (buffer size: {len(buffer)}/64)")
            loss_history.append(0.0)
        
        if (iteration + 1) % 5 == 0:
            print(f"\nEvaluating against Heuristic (20 games)...")
            eval_results = evaluate_vs_heuristic(env, alphazero_agent, heuristic_agent, num_games=20)
            eval_winrate = eval_results["az_wins"] / 20
            eval_history.append({
                "iteration": iteration + 1,
                "az_wins": eval_results["az_wins"],
                "heuristic_wins": eval_results["heuristic_wins"],
                "draws": eval_results["draws"],
                "winrate": eval_winrate
            })
            print(f"  Evaluation: AZ {eval_results['az_wins']}/20 wins ({eval_winrate*100:.1f}%)")
        
        if (iteration + 1) % 10 == 0 or iteration == num_iterations - 1:
            model_path = f"logs/model_vs_heuristic_iter_{iteration+1}.pt"
            torch.save(alphazero_agent.network.state_dict(), model_path)
            print(f"\n✓ Saved model: {model_path}")
        
        log_data = {
            "reward_history": reward_history,
            "loss_history": loss_history,
            "eval_history": eval_history
        }
        with open("logs/training_vs_heuristic_logs.json", "w") as f:
            json.dump(log_data, f, indent=4)
        
        if len(reward_history) > 1:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Win rate over time
            axes[0].plot(reward_history)
            axes[0].set_title("AlphaZero Win Rate vs Heuristic")
            axes[0].set_xlabel("Training Iteration (vs Heuristic)")
            axes[0].set_ylabel("Win Rate")
            axes[0].grid(True)
            axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='50% (even)')
            axes[0].legend()
            
            # Training loss
            axes[1].plot(loss_history)
            axes[1].set_title("Training Loss (vs Heuristic)")
            axes[1].set_xlabel("Training Iteration")
            axes[1].set_ylabel("Loss")
            axes[1].grid(True)
            
            # Evaluation results
            if eval_history:
                eval_iters = [e["iteration"] for e in eval_history]
                eval_winrates = [e["winrate"] for e in eval_history]
                axes[2].plot(eval_iters, eval_winrates, marker='o')
                axes[2].set_title("Evaluation Win Rate")
                axes[2].set_xlabel("Iteration")
                axes[2].set_ylabel("Win Rate")
                axes[2].grid(True)
                axes[2].axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("logs/training_vs_heuristic_curves.png", dpi=100)
            plt.close()
            
            old_logs_file = "logs/training_logs.json"
            if os.path.exists(old_logs_file):
                try:
                    with open(old_logs_file, "r") as f:
                        old_log_data = json.load(f)
                    
                    old_reward = old_log_data.get("reward_history", [])
                    old_loss = old_log_data.get("loss_history", [])
                    old_winrate = old_log_data.get("winrate_history", [])
                    
                    # Create combined plot
                    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                    
                    # Combined Loss
                    all_loss = old_loss + loss_history
                    axes[0].plot(all_loss)
                    axes[0].axvline(x=len(old_loss), color='red', linestyle='--', 
                                   linewidth=2, label=f'Switched to vs-Heuristic (iter {len(old_loss)})')
                    axes[0].set_title("Training Loss (Complete History)")
                    axes[0].set_xlabel("Iteration")
                    axes[0].set_ylabel("Loss")
                    axes[0].grid(True)
                    axes[0].legend()
                    
                    # Combined Reward (self-play vs vs-heuristic)
                    # Note: These are different metrics so we plot separately
                    x_old = list(range(len(old_reward)))
                    x_new = list(range(len(old_reward), len(old_reward) + len(reward_history)))
                    
                    axes[1].plot(x_old, old_reward, label='Self-play (avg reward)', color='blue')
                    axes[1].plot(x_new, reward_history, label='Vs Heuristic (win rate)', color='green')
                    axes[1].axvline(x=len(old_reward), color='red', linestyle='--', 
                                   linewidth=2, alpha=0.7)
                    axes[1].set_title("Performance Metric")
                    axes[1].set_xlabel("Iteration")
                    axes[1].set_ylabel("Value")
                    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.2)
                    axes[1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.3, label='50% win rate')
                    axes[1].grid(True)
                    axes[1].legend()
                    
                    # Win rates (self-play P1/P2, then vs heuristic)
                    if old_winrate:
                        old_p1 = [w["p1"] for w in old_winrate]
                        x_old_wr = list(range(len(old_p1)))
                        axes[2].plot(x_old_wr, old_p1, label='Self-play P1 win rate', 
                                    color='blue', alpha=0.6)
                    
                    # vs Heuristic win rate
                    x_new_wr = list(range(len(old_reward), len(old_reward) + len(reward_history)))
                    axes[2].plot(x_new_wr, reward_history, label='Vs Heuristic win rate', 
                                color='green', linewidth=2)
                    axes[2].axvline(x=len(old_reward), color='red', linestyle='--', 
                                   linewidth=2, alpha=0.7)
                    axes[2].axhline(y=0.5, color='orange', linestyle='--', alpha=0.3)
                    axes[2].set_title("Win Rates (Combined)")
                    axes[2].set_xlabel("Iteration")
                    axes[2].set_ylabel("Win Rate")
                    axes[2].grid(True)
                    axes[2].legend()
                    
                    plt.tight_layout()
                    plt.savefig("logs/training_combined_history.png", dpi=100)
                    plt.close()
                    
                    print(f"✓ Updated plots:")
                    print(f"  - logs/training_vs_heuristic_curves.png (vs-heuristic only)")
                    print(f"  - logs/training_combined_history.png (complete history)")
                    
                except Exception as e:
                    print(f"⚠️ Could not create combined plot: {e}")
                    print(f"✓ Updated plot: logs/training_vs_heuristic_curves.png")
            else:
                print(f"✓ Updated plot: logs/training_vs_heuristic_curves.png")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nFinal win rate: {reward_history[-1]*100:.1f}%")
    print(f"Models saved in logs/")
    print(f"Logs saved to: logs/training_vs_heuristic_logs.json")
    print(f"Plots saved to: logs/training_vs_heuristic_curves.png")