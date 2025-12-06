import numpy as np
import torch
from connect4_env.connect4_env import Connect4Env
from agents.alphazero.alphazero_agent import AlphaZeroAgent
from training.replay_buffer import ReplayBuffer
from agents.alphazero.az_network import AZNetwork

def encode_board(board, current_player):
    
    b = board
    p1 = (b == current_player).astype(np.float32)
    p2 = (b == -current_player).astype(np.float32)
    stacked = np.stack([p1, p2], axis=0)
    return torch.tensor(stacked, dtype=torch.float32)




def self_play_games(env, agent, replay_buffer, num_games=50, temperature_sched=None, verbose=False):
    
    for g in range(num_games):
        env.reset()
        states = []
        pis = []
        players = []
        move_count = 0

        # Optionally alternate starting player
        # env.current_player = 1 if (g % 2 == 0) else -1

        done = False
        while True:
            board = env.board.copy()
            current_player = env.current_player

            # pick temperature for exploration
            temp = 1.0
            if temperature_sched is not None:
                temp = temperature_sched(move_count)

            # run MCTS via agent and get policy
            action, pi = agent.policy_from_root(board, current_player, temperature=temp)
            # record pre-move state and root policy
            states.append(encode_board(board, current_player))
            pis.append(pi)
            players.append(current_player)

            # apply action on real env
            _, _, done = env.step(action)
            move_count += 1

            # check for terminal
            if env.check_win(current_player):
                winner = current_player
                break
            if env.is_draw():
                winner = 0
                break

        # label each stored (s, pi) with z in perspective of stored player
        for (s, pi_vec, p) in zip(states, pis, players):
            if winner == 0:
                z = 0.0
            else:
                z = 1.0 if winner == p else -1.0
            replay_buffer.push(s, pi_vec, float(z))

        if verbose:
            print(f"Self-play game {g+1}/{num_games} done. Winner: {winner}. Buffer size: {len(replay_buffer)}")

    return replay_buffer

# small temperature schedule example:
def default_temp(move):
    # first 10 moves high exploration, then low
    return 1.0 if move < 10 else 0.1

def train_step(self, batch):
    
    # States are already tensors from encode_board, just move to device
    states = batch["states"].to(self.device)
    target_policies = batch["policies"].to(self.device)
    target_values = batch["values"].to(self.device)
    