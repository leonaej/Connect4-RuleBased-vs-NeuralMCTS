# main/train_alphazero_vs_heuristic_net2.py
"""
Train AlphaZero (Network2 - improved) against Heuristic agent
This uses the improved network with proper residual connections
"""

import numpy as np
import torch
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from connect4_env.connect4_env import Connect4Env
from agents.alphazero.az_network2 import AZNetwork2  # Using Network2!
from agents.alphazero.alphazero_agent import AlphaZeroAgent
from agents.heuristic_agent import HeuristicAgent
from training.replay_buffer import ReplayBuffer
from training.train_one_iter import train_one_iteration


def encode_board(board, current_player):
    """
    Returns torch.FloatTensor shape (2,6,7):
      channel 0 = current_player pieces
      channel 1 = opponent pieces
    """
    p1 = (board == current_player).astype(np.float32)
    p2 = (board == -current_player).astype(np.float32)
    stacked = np.stack([p1, p2], axis=0)
    return torch.tensor(stacked, dtype=torch.float32)


def play_game_vs_heuristic(env, alphazero_agent, heuristic_agent, 
                           buffer, alphazero_player, temperature=1.0):
    """
    Play one game: AlphaZero vs Heuristic.
    
    Args:
        alphazero_player: 1 if AlphaZero plays first, -1 if second
        
    Returns:
        winner: 1, -1, or 0
    """
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


# MAIN TRAINING LOOP

if __name__ == "__main__":
    print("="*70)
    print("Training AlphaZero (Network2) Against Heuristic Agent")
    print("Using improved network with proper residual connections")
    print("="*70)
    
    # Setup
    env = Connect4Env()
    network = AZNetwork2()  # Using Network2!
    print("\n✓ Using AZNetwork2 (improved architecture)")
    print("  - 4 residual blocks with skip connections")
    print("  - Better gradient flow for faster learning\n")
    
    alphazero_agent = AlphaZeroAgent(
        env=env, 
        network=network, 
        num_simulations=100,
        c_puct=1.5, 
        device="cuda"  # Using GPU
    )
    heuristic_agent = HeuristicAgent(depth=3)  # Keep depth=3
    buffer = ReplayBuffer(capacity=20000)
    
    # Create separate logs directory for Network2
    os.makedirs("logs2", exist_ok=True)
    print("✓ Saving all Network2 training data to logs2/\n")
    
    # Training settings
    num_iterations = 50
    games_per_iteration = 50
    
    # History tracking
    reward_history = []
    loss_history = []
    eval_history = []
    
    # Training Loop
    for iteration in range(num_iterations):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration+1}/{num_iterations}")
        print(f"{'='*70}")
        
        # 1. PLAY GAMES vs Heuristic
        print(f"\nPlaying {games_per_iteration} games vs Heuristic...")
        
        az_wins = 0
        heuristic_wins = 0
        draws = 0
        
        # Temperature schedule
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
        
        # 2. TRAIN NETWORK
        if len(buffer) >= 64:
            print(f"\nTraining network...")
            loss = train_one_iteration(alphazero_agent, buffer, batch_size=64, steps=100)
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
        
        # 4. SAVE CHECKPOINT (every iteration)
        model_path = f"logs2/model_net2_iter_{iteration+1}.pt"
        torch.save(alphazero_agent.network.state_dict(), model_path)
        print(f"\n✓ Saved model: {model_path}")
        
        # 5. SAVE LOGS
        log_data = {
            "reward_history": reward_history,
            "loss_history": loss_history,
            "eval_history": eval_history,
            "network": "AZNetwork2",
            "num_iterations": iteration + 1
        }
        with open("logs2/training_net2_logs.json", "w") as f:
            json.dump(log_data, f, indent=4)
        
        # 6. PLOT PROGRESS (every 5 iterations)
        if (iteration + 1) % 5 == 0 and len(reward_history) > 1:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # Win rate
            axes[0].plot(reward_history, 'g-', linewidth=2)
            axes[0].set_title("AlphaZero Win Rate vs Heuristic (Network2)")
            axes[0].set_xlabel("Iteration")
            axes[0].set_ylabel("Win Rate")
            axes[0].grid(True)
            axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='50%')
            axes[0].legend()
            
            # Loss
            axes[1].plot(loss_history, 'b-', linewidth=2)
            axes[1].set_title("Training Loss")
            axes[1].set_xlabel("Iteration")
            axes[1].set_ylabel("Loss")
            axes[1].grid(True)
            
            # Evaluation
            if eval_history:
                eval_iters = [e["iteration"] for e in eval_history]
                eval_winrates = [e["winrate"] for e in eval_history]
                axes[2].plot(eval_iters, eval_winrates, 'ro-', linewidth=2, markersize=8)
                axes[2].set_title("Evaluation Win Rate")
                axes[2].set_xlabel("Iteration")
                axes[2].set_ylabel("Win Rate")
                axes[2].grid(True)
                axes[2].axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig("logs2/training_net2_curves.png", dpi=100)
            plt.close()
            print(f"✓ Updated plot: logs2/training_net2_curves.png")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nFinal win rate: {reward_history[-1]*100:.1f}%")
    print(f"Models saved: logs2/model_net2_iter_*.pt")
    print(f"Logs: logs2/training_net2_logs.json")
    print(f"Plots: logs2/training_net2_curves.png")