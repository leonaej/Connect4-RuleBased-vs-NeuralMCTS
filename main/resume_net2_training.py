

import numpy as np
import torch
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from connect4_env.connect4_env import Connect4Env
from agents.alphazero.az_network2 import AZNetwork2
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
    """
    Play one game: AlphaZero vs Heuristic.
    """
    env.reset()
    game_states = []
    
    # If AlphaZero should go second, let heuristic make first move
    if alphazero_player == -1:
        action = heuristic_agent.select_action(env)
        if action is None or action not in env.get_valid_actions():
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
            if hasattr(action, 'item'):
                action = action.item()
            action = int(action)
            
            encoded_state = encode_board(board, current_player)
            game_states.append((encoded_state, pi, current_player))
        else:
            # Heuristic's turn
            action = heuristic_agent.select_action(env)
            if action is None or action not in env.get_valid_actions():
                valid_moves = env.get_valid_actions()
                if not valid_moves:
                    winner = 0
                    break
                action = valid_moves[0]
            action = int(action)
        
        player_who_moved = current_player
        _, reward, done = env.step(action)
        move_count += 1
        
        if env.check_win(player_who_moved):
            winner = player_who_moved
            break
        
        if env.is_draw():
            winner = 0
            break
    else:
        winner = 0
    
    # Label AlphaZero's moves
    for state, pi, player in game_states:
        if winner == 0:
            value = 0.0
        else:
            value = 1.0 if winner == alphazero_player else -1.0
        buffer.push(state, pi, value)
    
    return winner


def evaluate_vs_heuristic(env, alphazero_agent, heuristic_agent, num_games=20):
    """Evaluate AlphaZero against Heuristic"""
    results = {"az_wins": 0, "heuristic_wins": 0, "draws": 0}
    
    for i in range(num_games):
        alphazero_player = 1 if i % 2 == 0 else -1
        winner = play_game_vs_heuristic(
            env, alphazero_agent, heuristic_agent,
            buffer=ReplayBuffer(capacity=1),
            alphazero_player=alphazero_player,
            temperature=0.1
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
    print("RESUMING Network2 Training from Iteration 50")
    print("="*70)
    print("New settings:")
    print("  - 50 more iterations (total will be 100)")
    print("  - 100 games per iteration (increased from 50)")
    print("  - Continue training against heuristic")
    print("="*70)
    
    # Setup
    env = Connect4Env()
    network = AZNetwork2()
    
    # Load existing model
    checkpoint_path = "logs2/model_net2_iter_50.pt"
    if not os.path.exists(checkpoint_path):
        print(f"\n Checkpoint not found: {checkpoint_path}")
        print("Please ensure you have trained Network2 to iteration 50 first.")
        import sys
        sys.exit(1)
    
    print(f"\n  Loading checkpoint: {checkpoint_path}")
    network.load_state_dict(torch.load(checkpoint_path, map_location="cuda"))
    network.eval()
    print("  Model loaded successfully!")
    
    # Load training history
    logs_path = "logs2/training_net2_logs.json"
    if os.path.exists(logs_path):
        with open(logs_path, "r") as f:
            old_logs = json.load(f)
        reward_history = old_logs.get("reward_history", [])
        loss_history = old_logs.get("loss_history", [])
        eval_history = old_logs.get("eval_history", [])
        print(f"  Loaded training history (50 iterations)")
        print(f"  - Last win rate: {reward_history[-1]*100:.1f}%")
        print(f"  - Last loss: {loss_history[-1]:.4f}")
    else:
        reward_history = []
        loss_history = []
        eval_history = []
        print("  No previous logs found, starting fresh history")
    
    start_iteration = len(reward_history)
    
    alphazero_agent = AlphaZeroAgent(
        env=env, 
        network=network, 
        num_simulations=100,
        c_puct=1.5, 
        device="cuda"
    )
    heuristic_agent = HeuristicAgent(depth=3)
    buffer = ReplayBuffer(capacity=30000)  # Increased capacity
    
    # Load existing buffer if available
    buffer_path = "logs2/replay_buffer_iter50.pkl"
    if os.path.exists(buffer_path):
        buffer.load(buffer_path)
        print(f"  Loaded replay buffer ({len(buffer)} samples)")
    
    # Training settings
    num_new_iterations = 50
    games_per_iteration = 100  # Increased from 50
    total_iterations = start_iteration + num_new_iterations
    
    print(f"\nStarting training from iteration {start_iteration + 1} to {total_iterations}")
    print("="*70)
    
    # Training Loop
    for iteration in range(start_iteration, total_iterations):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration+1}/{total_iterations}")
        print(f"{'='*70}")
        
        # 1. PLAY GAMES vs Heuristic
        print(f"\nPlaying {games_per_iteration} games vs Heuristic...")
        
        az_wins = 0
        heuristic_wins = 0
        draws = 0
        
        # Temperature schedule
        temp = 1.0 if iteration < start_iteration + 15 else 0.3
        
        for g in tqdm(range(games_per_iteration), desc="Games", leave=False):
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
        
        # 2. TRAIN NETWORK
        if len(buffer) >= 64:
            print(f"\nTraining network...")
            loss = train_one_iteration(alphazero_agent, buffer, batch_size=64, steps=150)  # More training steps
            loss_history.append(loss)
            print(f"  Training loss: {loss:.4f}")
        else:
            print(f"\nSkipping training (buffer size: {len(buffer)}/64)")
            loss_history.append(0.0)
        
        # 3. EVALUATE (every 5 iterations)
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
        
        # 4. SAVE CHECKPOINT
        model_path = f"logs2/model_net2_iter_{iteration+1}.pt"
        torch.save(alphazero_agent.network.state_dict(), model_path)
        print(f"\n  Saved model: {model_path}")
        
        # 5. SAVE BUFFER (every 10 iterations)
        if (iteration + 1) % 10 == 0:
            buffer_save_path = f"logs2/replay_buffer_iter{iteration+1}.pkl"
            buffer.save(buffer_save_path)
            print(f"  Saved buffer: {buffer_save_path}")
        
        # 6. SAVE LOGS
        log_data = {
            "reward_history": reward_history,
            "loss_history": loss_history,
            "eval_history": eval_history,
            "network": "AZNetwork2",
            "num_iterations": iteration + 1
        }
        with open("logs2/training_net2_logs.json", "w") as f:
            json.dump(log_data, f, indent=4)
        
        # 7. PLOT PROGRESS
        if len(reward_history) > 1:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Win rate
            axes[0].plot(reward_history, 'g-', linewidth=2)
            axes[0].axvline(x=start_iteration, color='red', linestyle='--', 
                          linewidth=2, label=f'Resumed (100 games/iter)')
            axes[0].set_title("AlphaZero Win Rate vs Heuristic")
            axes[0].set_xlabel("Iteration")
            axes[0].set_ylabel("Win Rate")
            axes[0].grid(True)
            axes[0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.3)
            axes[0].legend()
            
            # Loss
            axes[1].plot(loss_history, 'b-', linewidth=2)
            axes[1].axvline(x=start_iteration, color='red', linestyle='--', linewidth=2)
            axes[1].set_title("Training Loss")
            axes[1].set_xlabel("Iteration")
            axes[1].set_ylabel("Loss")
            axes[1].grid(True)
            
            # Evaluation
            if eval_history:
                eval_iters = [e["iteration"] for e in eval_history]
                eval_winrates = [e["winrate"] for e in eval_history]
                axes[2].plot(eval_iters, eval_winrates, 'ro-', linewidth=2, markersize=8)
                axes[2].axvline(x=start_iteration, color='red', linestyle='--', linewidth=2)
                axes[2].set_title("Evaluation Win Rate")
                axes[2].set_xlabel("Iteration")
                axes[2].set_ylabel("Win Rate")
                axes[2].grid(True)
                axes[2].axhline(y=0.5, color='orange', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("logs2/training_net2_curves.png", dpi=100)
            plt.close()
            print(f"  Updated plot: logs2/training_net2_curves.png")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nTotal iterations: {total_iterations}")
    print(f"Final win rate: {reward_history[-1]*100:.1f}%")
    print(f"Final loss: {loss_history[-1]:.4f}")
    print(f"Buffer size: {len(buffer)}")
    print(f"\n  All models saved: logs2/model_net2_iter_*.pt")
    print(f"  Logs: logs2/training_net2_logs.json")
    print(f"  Plots: logs2/training_net2_curves.png")