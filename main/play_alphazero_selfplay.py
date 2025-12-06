
from pathlib import Path
import sys


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


import torch
from connect4_env.connect4_env import Connect4Env
from agents.alphazero_agent import AlphaZeroAgent
from az_network import AZNetwork
from mcts import MCTS  
import random

# Hyperparameters
NUM_SELFPLAY_GAMES = 100
NUM_SIMULATIONS = 200
C_PUCT = 1.4
DEPTH = 3  # optional for heuristics
BATCH_SIZE = 64
EPOCHS = 5
DEVICE = "cpu"

# Initialize environment and network
env = Connect4Env()
network = AZNetwork()
agent = AlphaZeroAgent(env, network, num_simulations=NUM_SIMULATIONS, c_puct=C_PUCT, device=DEVICE)

# Replay buffer for storing (board, policy, outcome)
replay_buffer = []

def self_play_game():
    board = env.reset()
    current_player = env.current_player
    done = False
    game_data = []

    while not done:
        # Run MCTS + network to get policy
        mcts = MCTS(env, network, NUM_SIMULATIONS, C_PUCT, DEVICE)
        policy = mcts.run(board.copy(), current_player)

        # Choose action
        action = int(policy.argmax())

        # Record board and policy
        game_data.append((board.copy(), policy, current_player))

        # Apply action
        board, _, done = env.step(action)
        current_player = env.current_player

    # Assign outcome to each move
    winner = 1 if env.check_win(1) else -1 if env.check_win(-1) else 0
    labeled_game_data = [(b, p, winner if player == 1 else -winner) for (b, p, player) in game_data]
    return labeled_game_data

# Generate self-play games
for i in range(NUM_SELFPLAY_GAMES):
    game_samples = self_play_game()
    replay_buffer.extend(game_samples)

# Training loop (simplified)
optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
loss_fn_value = torch.nn.MSELoss()
loss_fn_policy = torch.nn.CrossEntropyLoss()

# Example of one epoch
for epoch in range(EPOCHS):
    random.shuffle(replay_buffer)
    for i in range(0, len(replay_buffer), BATCH_SIZE):
        batch = replay_buffer[i:i+BATCH_SIZE]
        boards, policies, values = zip(*batch)
        # convert boards/policies/values to tensors
        boards_tensor = torch.tensor(boards, dtype=torch.float32)
        policies_tensor = torch.tensor(policies, dtype=torch.float32)
        values_tensor = torch.tensor(values, dtype=torch.float32)

        # Forward pass
        pred_policy, pred_value = network(boards_tensor)

        # Compute losses
        loss_policy = loss_fn_policy(pred_policy, policies_tensor)
        loss_value = loss_fn_value(pred_value.squeeze(), values_tensor)

        loss = loss_policy + loss_value

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
   