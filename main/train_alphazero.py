
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

env = Connect4Env()
net = AZNetwork()
agent = AlphaZeroAgent(env=env, network=net, num_simulations=200, c_puct=1.5, device="cpu")
buffer = ReplayBuffer(capacity=20000)

os.makedirs("logs", exist_ok=True)

reward_history = []
loss_history = []
winrate_history = []

num_iterations = 20
games_per_iteration = 50

for it in range(num_iterations):
    print(f"\n=== ITERATION {it+1}/{num_iterations} ===")

    # 1. SELF-PLAY
    rewards = []
    p1_wins = p2_wins = draws = 0

    # Temperature schedule: high exploration early, low later
    temperature = 1.0 if it < 10 else 0.1
    
    # Use this:
    print(f"Playing {games_per_iteration} self-play games...")
    for g in range(games_per_iteration):
        result = play_one_game_with_mcts(env, agent, buffer, temperature=temperature)
        
        # Print progress every 10 games
        if (g + 1) % 10 == 0:
            print(f"  Completed {g+1}/{games_per_iteration} games")
            
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
        "p1": p1_wins / games_per_iteration,
        "p2": p2_wins / games_per_iteration,
        "draw": draws / games_per_iteration
    })

    print(f"Self-play → Avg Reward: {avg_reward:.3f} | P1: {p1_wins} | P2: {p2_wins} | Draws: {draws}")
    print(f"Buffer size: {len(buffer)}")

    # 2. TRAINING (only if we have enough samples)
    if len(buffer) >= 64:
        loss = train_one_iteration(agent, buffer, batch_size=64, steps=100)
        loss_history.append(loss)
        print(f"Training loss: {loss:.4f}")
    else:
        print(f"Skipping training - not enough samples ({len(buffer)}/64)")
        loss_history.append(0.0)

    # 3. SAVE MODEL
    save_path = f"logs/model_iter_{it+1}.pt"
    torch.save(agent.network.state_dict(), save_path)
    print(f"Saved model → {save_path}")

    # 4. SAVE LOGS
    log_data = {
        "reward_history": reward_history,
        "loss_history": loss_history,
        "winrate_history": winrate_history
    }
    with open("logs/training_logs.json", "w") as f:
        json.dump(log_data, f, indent=4)

    # 5. PLOT CURVES
    if len(loss_history) > 1:
        plt.figure(figsize=(12,4))

        plt.subplot(1,2,1)
        plt.plot(loss_history)
        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")

        plt.subplot(1,2,2)
        plt.plot(reward_history)
        plt.title("Average Reward")
        plt.xlabel("Iteration")
        plt.ylabel("Reward")

        plt.tight_layout()
        plt.savefig("logs/learning_curves.png")
        plt.close()

print("\nTraining complete!")