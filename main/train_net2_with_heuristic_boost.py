

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
    """Returns torch.FloatTensor shape (2,6,7)"""
    p1 = (board == current_player).astype(np.float32)
    p2 = (board == -current_player).astype(np.float32)
    stacked = np.stack([p1, p2], axis=0)
    return torch.tensor(stacked, dtype=torch.float32)


def blend_mcts_with_heuristic(mcts_policy, heuristic_move, boost_factor=0.3):
    
    policy = np.array(mcts_policy, dtype=float)
    
    # Take boost_factor probability mass from all moves
    boost_amount = boost_factor * policy[heuristic_move]
    
    # Reduce all moves proportionally
    policy = policy * (1 - boost_factor)
    
    # Give all that mass to heuristic move
    policy[heuristic_move] += boost_amount
    
    # Normalize to ensure sum = 1.0
    policy = policy / (np.sum(policy) + 1e-10)
    
    return policy


def play_game_with_heuristic_guidance(env, alphazero_agent, heuristic_agent, 
                                      buffer, alphazero_player, 
                                      temperature=1.0, boost_factor=0.3):
    """
    Play game with heuristic-augmented MCTS policy
    """
    env.reset()
    game_states = []
    
    # If AlphaZero goes second, heuristic plays first
    if alphazero_player == -1:
        action = heuristic_agent.select_action(env)
        if action is None or action not in env.get_valid_actions():
            action = env.get_valid_actions()[0]
        env.step(int(action))
    
    move_count = 0
    max_moves = 42
    
    while move_count < max_moves:
        board = env.board.copy()
        current_player = env.current_player
        
        if current_player == alphazero_player:
            # AlphaZero's turn - get MCTS policy
            _, mcts_policy = alphazero_agent.policy_from_root(
                board, current_player, temperature=temperature
            )
            
            # Get heuristic's suggestion
            temp_env = Connect4Env()
            temp_env.board = board.copy()
            temp_env.current_player = current_player
            heuristic_move = heuristic_agent.select_action(temp_env)
            
            # Blend MCTS with heuristic
            if heuristic_move is not None and heuristic_move in env.get_valid_actions():
                blended_policy = blend_mcts_with_heuristic(
                    mcts_policy, heuristic_move, boost_factor
                )
            else:
                blended_policy = mcts_policy
            
            # Sample action from blended policy
            if temperature == 0:
                action = int(np.argmax(blended_policy))
            else:
                action = int(np.random.choice(len(blended_policy), p=blended_policy))
            
            # Store with BLENDED policy (this is what network learns)
            encoded_state = encode_board(board, current_player)
            game_states.append((encoded_state, blended_policy, current_player))
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
    
    # Label moves with outcome
    for state, policy, player in game_states:
        if winner == 0:
            value = 0.0
        else:
            value = 1.0 if winner == alphazero_player else -1.0
        buffer.push(state, policy, value)
    
    return winner


def evaluate_vs_heuristic(env, alphazero_agent, heuristic_agent, num_games=20):
    """Evaluate without heuristic guidance"""
    results = {"az_wins": 0, "heuristic_wins": 0, "draws": 0}
    
    for i in range(num_games):
        alphazero_player = 1 if i % 2 == 0 else -1
        winner = play_game_with_heuristic_guidance(
            env, alphazero_agent, heuristic_agent,
            buffer=ReplayBuffer(capacity=1),
            alphazero_player=alphazero_player,
            temperature=0.1,
            boost_factor=0.0  # No guidance during evaluation
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
    print("Training Network2 with Heuristic-Augmented MCTS")
    print("="*70)
    print("Strategy: Blend MCTS policy with heuristic guidance")
    print("  - MCTS provides exploration")
    print("  - Heuristic provides expert knowledge")
    print("  - Network learns from combined policy")
    print("="*70)
    
    # Setup
    env = Connect4Env()
    network = AZNetwork2()
    
    # Find latest checkpoint
    if os.path.exists("logs2"):
        checkpoints = [f for f in os.listdir("logs2") if f.startswith("model_net2_iter_") and f.endswith(".pt")]
        if checkpoints:
            # Sort by iteration number
            latest = sorted(checkpoints, key=lambda x: int(x.split("_")[3].split(".")[0]))[-1]
            checkpoint_path = os.path.join("logs2", latest)
            iter_num = int(latest.split("_")[3].split(".")[0])
            
            print(f"\n✓ Found latest checkpoint: {latest}")
            print(f"   Iteration: {iter_num}")
            choice = input("Load this checkpoint? (y/n): ").strip().lower()
            
            if choice == 'y':
                network.load_state_dict(torch.load(checkpoint_path, map_location="cuda"))
                print("✓ Loaded existing model!")
                start_iteration = iter_num
            else:
                print("Starting with random weights")
                start_iteration = 0
        else:
            print("\nNo checkpoints found, starting fresh")
            start_iteration = 0
    else:
        print("\nNo logs2/ folder found, starting fresh")
        start_iteration = 0
    
    alphazero_agent = AlphaZeroAgent(
        env=env, 
        network=network, 
        num_simulations=100,
        c_puct=1.5, 
        device="cuda"
    )
    heuristic_agent = HeuristicAgent(depth=3)
    buffer = ReplayBuffer(capacity=30000)
    
    # Training settings
    num_iterations = 50
    games_per_iteration = 100
    boost_factor = 0.3  # 30% guidance from heuristic
    
    print(f"\nTraining settings:")
    print(f"  - Iterations: {num_iterations}")
    print(f"  - Games per iteration: {games_per_iteration}")
    print(f"  - Heuristic boost factor: {boost_factor}")
    print(f"  - Starting from iteration: {start_iteration}")
    
    # History
    reward_history = []
    loss_history = []
    eval_history = []
    
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    # Training Loop
    for iteration in range(num_iterations):
        current_iter = start_iteration + iteration + 1
        print(f"\n{'='*70}")
        print(f"ITERATION {current_iter}")
        print(f"{'='*70}")
        
        # Play games with heuristic guidance
        print(f"\nPlaying {games_per_iteration} games (with heuristic guidance)...")
        
        az_wins = 0
        heuristic_wins = 0
        draws = 0
        
        # Temperature schedule
        temp = 1.0 if iteration < 15 else 0.3
        
        # Reduce boost factor over time (fade out guidance)
        current_boost = boost_factor * max(0, 1 - iteration / (num_iterations * 0.7))
        print(f"Current boost factor: {current_boost:.2f}")
        
        for g in tqdm(range(games_per_iteration), desc="Games", leave=False):
            alphazero_player = 1 if g % 2 == 0 else -1
            
            winner = play_game_with_heuristic_guidance(
                env, alphazero_agent, heuristic_agent,
                buffer, alphazero_player, 
                temperature=temp,
                boost_factor=current_boost
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
        print(f"  Heuristic wins: {heuristic_wins}")
        print(f"  Draws: {draws}")
        print(f"  Buffer size: {len(buffer)}")
        
        reward_history.append(az_winrate)
        
        # Train network
        if len(buffer) >= 64:
            print(f"\nTraining network...")
            loss = train_one_iteration(alphazero_agent, buffer, batch_size=64, steps=150)
            loss_history.append(loss)
            print(f"  Training loss: {loss:.4f}")
        else:
            loss_history.append(0.0)
        
        # Evaluate (without guidance)
        if (current_iter) % 5 == 0:
            print(f"\nEvaluating (pure MCTS, no guidance)...")
            eval_results = evaluate_vs_heuristic(env, alphazero_agent, heuristic_agent, 20)
            eval_winrate = eval_results["az_wins"] / 20
            eval_history.append({
                "iteration": current_iter,
                "az_wins": eval_results["az_wins"],
                "winrate": eval_winrate
            })
            print(f"  Pure MCTS: {eval_results['az_wins']}/20 wins ({eval_winrate*100:.1f}%)")
        
        # Save model
        model_path = f"logs2/model_net2_boosted_iter_{current_iter}.pt"
        torch.save(alphazero_agent.network.state_dict(), model_path)
        print(f"\n✓ Saved: {model_path}")
        
        # Save logs
        log_data = {
            "reward_history": reward_history,
            "loss_history": loss_history,
            "eval_history": eval_history,
            "boost_factor": boost_factor,
            "method": "heuristic_augmented_mcts"
        }
        with open("logs2/training_boosted_logs.json", "w") as f:
            json.dump(log_data, f, indent=4)
        
        # Plot
        if len(reward_history) > 1:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            axes[0].plot(reward_history, 'g-', linewidth=2, label='Training (with guidance)')
            if eval_history:
                eval_x = [e["iteration"] - start_iteration - 1 for e in eval_history]
                eval_y = [e["winrate"] for e in eval_history]
                axes[0].plot(eval_x, eval_y, 'ro-', linewidth=2, label='Eval (pure MCTS)')
            axes[0].set_title("Win Rate vs Heuristic")
            axes[0].set_xlabel("Iteration")
            axes[0].set_ylabel("Win Rate")
            axes[0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.3)
            axes[0].legend()
            axes[0].grid(True)
            
            axes[1].plot(loss_history, 'b-', linewidth=2)
            axes[1].set_title("Training Loss")
            axes[1].set_xlabel("Iteration")
            axes[1].set_ylabel("Loss")
            axes[1].grid(True)
            
            plt.tight_layout()
            plt.savefig("logs2/training_boosted_curves.png", dpi=100)
            plt.close()
            print(f"✓ Updated: logs2/training_boosted_curves.png")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Final training win rate: {reward_history[-1]*100:.1f}%")
    if eval_history:
        print(f"Final eval win rate: {eval_history[-1]['winrate']*100:.1f}%")