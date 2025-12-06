
# Resume Training from Checkpoint

from connect4_env.connect4_env import Connect4Env
from agents.alphazero.az_network import AZNetwork
from agents.alphazero.alphazero_agent import AlphaZeroAgent
from training.replay_buffer import ReplayBuffer
from training.train_one_iter import train_one_iteration
from training.play_one_game import play_one_game_with_mcts

from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt
import torch

# -------------------------
# Configuration
# -------------------------
RESUME_FROM_CHECKPOINT = True  # Set to False to start fresh
CHECKPOINT_NAME = "model_iter_20.pt"  # Which checkpoint to load
LOGS_FILE = "logs/training_logs.json"

#change the following to 30 and 50 
#NUM_NEW_ITERATIONS = 30  # How many MORE iterations to train
#GAMES_PER_ITERATION = 50


# Training settings
NUM_NEW_ITERATIONS = 30  # How many MORE iterations to train
GAMES_PER_ITERATION = 60
NUM_SIMULATIONS = 200
BATCH_SIZE = 64
TRAINING_STEPS = 100

# -------------------------
# Setup
# -------------------------
env = Connect4Env()
net = AZNetwork()

# Load existing model weights if resuming
start_iteration = 0
reward_history = []
loss_history = []
winrate_history = []

if RESUME_FROM_CHECKPOINT and os.path.exists(f"logs/{CHECKPOINT_NAME}"):
    print(f"{'='*70}")
    print(f"RESUMING TRAINING FROM: {CHECKPOINT_NAME}")
    print(f"{'='*70}\n")
    
    # Load model weights
    checkpoint_path = f"logs/{CHECKPOINT_NAME}"
    net.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    print(f"✓ Loaded model weights from {checkpoint_path}")
    
    # Load training history
    if os.path.exists(LOGS_FILE):
        with open(LOGS_FILE, "r") as f:
            log_data = json.load(f)
        reward_history = log_data.get("reward_history", [])
        loss_history = log_data.get("loss_history", [])
        winrate_history = log_data.get("winrate_history", [])
        start_iteration = len(reward_history)
        print(f"✓ Loaded training history ({start_iteration} iterations completed)")
        print(f"  - Last avg reward: {reward_history[-1]:.3f}")
        print(f"  - Last loss: {loss_history[-1]:.4f}")
    else:
        print(" No training logs found, starting fresh history")
    
    print()
else:
    print(f"{'='*70}")
    print("STARTING FRESH TRAINING")
    print(f"{'='*70}\n")

agent = AlphaZeroAgent(
    env=env, 
    network=net, 
    num_simulations=NUM_SIMULATIONS, 
    c_puct=1.5, 
    device="cpu"
)
buffer = ReplayBuffer(capacity=20000)

# Optionally load existing replay buffer
BUFFER_FILE = "logs/replay_buffer.pkl"
if RESUME_FROM_CHECKPOINT and os.path.exists(BUFFER_FILE):
    buffer.load(BUFFER_FILE)
    print(f"✓ Loaded replay buffer ({len(buffer)} samples)\n")

os.makedirs("logs", exist_ok=True)

# -------------------------
# AlphaZero Training Loop
# -------------------------
total_iterations = start_iteration + NUM_NEW_ITERATIONS

for it in range(start_iteration, total_iterations):
    iteration_num = it + 1
    print(f"\n=== ITERATION {iteration_num}/{total_iterations} ===")

    # 1. SELF-PLAY
    rewards = []
    p1_wins = p2_wins = draws = 0

    # Temperature schedule: high exploration early, then lower
    # Adjust based on total iterations including previous training
    temperature = 1.0 if iteration_num < 15 else 0.3
    
    print(f"Playing {GAMES_PER_ITERATION} self-play games...")
    for g in range(GAMES_PER_ITERATION):
        result = play_one_game_with_mcts(env, agent, buffer, temperature=temperature)
        
        # Print progress every 10 games
        if (g + 1) % 10 == 0:
            print(f"  Completed {g+1}/{GAMES_PER_ITERATION} games")
            
        rewards.append(result)

        if result == 1:
            p1_wins += 1
        elif result == -1:
            p2_wins += 1
        else:
            draws += 1

    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    reward_history.append(avg_reward)

    winrate_history.append({
        "p1": p1_wins / GAMES_PER_ITERATION,
        "p2": p2_wins / GAMES_PER_ITERATION,
        "draw": draws / GAMES_PER_ITERATION
    })

    print(f"Self-play → Avg Reward: {avg_reward:.3f} | P1: {p1_wins} | P2: {p2_wins} | Draws: {draws}")
    print(f"Buffer size: {len(buffer)}")

    # 2. TRAINING (only if we have enough samples)
    if len(buffer) >= BATCH_SIZE:
        loss = train_one_iteration(agent, buffer, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)
        loss_history.append(loss)
        print(f"Training loss: {loss:.4f}")
    else:
        print(f"Skipping training - not enough samples ({len(buffer)}/{BATCH_SIZE})")
        loss_history.append(0.0)

    # 3. SAVE MODEL CHECKPOINT (every 5 iterations + at end)
    if iteration_num % 5 == 0 or iteration_num == total_iterations:
        save_path = f"logs/model_iter_{iteration_num}.pt"
        torch.save(agent.network.state_dict(), save_path)
        print(f"Saved model → {save_path}")

    # 4. SAVE REPLAY BUFFER (every 10 iterations)
    if iteration_num % 10 == 0:
        buffer.save(BUFFER_FILE)
        print(f"Saved replay buffer → {BUFFER_FILE}")

    # 5. SAVE LOGS (every iteration)
    log_data = {
        "reward_history": reward_history,
        "loss_history": loss_history,
        "winrate_history": winrate_history
    }
    with open(LOGS_FILE, "w") as f:
        json.dump(log_data, f, indent=4)

    # 6. PLOT CURVES (every 5 iterations)
    if iteration_num % 5 == 0 and len(loss_history) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss
        axes[0].plot(loss_history)
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True)
        if RESUME_FROM_CHECKPOINT and start_iteration > 0:
            axes[0].axvline(x=start_iteration, color='r', linestyle='--', 
                          label=f'Resumed at iter {start_iteration}')
            axes[0].legend()

        # Average Reward
        axes[1].plot(reward_history)
        axes[1].set_title("Average Reward")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Reward")
        axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1].grid(True)
        if RESUME_FROM_CHECKPOINT and start_iteration > 0:
            axes[1].axvline(x=start_iteration, color='r', linestyle='--',
                          label=f'Resumed at iter {start_iteration}')
            axes[1].legend()

        # Win rates
        if winrate_history:
            p1_rates = [w["p1"] for w in winrate_history]
            p2_rates = [w["p2"] for w in winrate_history]
            draw_rates = [w["draw"] for w in winrate_history]
            x = list(range(len(p1_rates)))
            
            axes[2].plot(x, p1_rates, label="P1", marker='o', markersize=3)
            axes[2].plot(x, p2_rates, label="P2", marker='s', markersize=3)
            axes[2].plot(x, draw_rates, label="Draw", marker='^', markersize=3)
            axes[2].set_title("Win Rates")
            axes[2].set_xlabel("Iteration")
            axes[2].set_ylabel("Rate")
            axes[2].legend()
            axes[2].grid(True)
            if RESUME_FROM_CHECKPOINT and start_iteration > 0:
                axes[2].axvline(x=start_iteration, color='r', linestyle='--')

        plt.tight_layout()
        plt.savefig("logs/learning_curves.png", dpi=100)
        plt.close()
        print(f"Updated plots → logs/learning_curves.png")

print("\nTraining complete!")