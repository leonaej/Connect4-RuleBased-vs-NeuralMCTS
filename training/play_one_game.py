
import numpy as np
import torch

def encode_board(board, current_player):
    
    b = board
    p1 = (b == current_player).astype(np.float32)
    p2 = (b == -current_player).astype(np.float32)
    stacked = np.stack([p1, p2], axis=0)
    return torch.tensor(stacked, dtype=torch.float32)


def play_one_game_with_mcts(env, agent, replay_buffer, temperature=1.0):
    
    obs = env.reset()
    done = False
    game_states = []
    winner = 0

    while not done:
        current_player = env.current_player  # Get current player BEFORE move
        
        # MCTS move
        action, pi = agent.policy_from_root(obs, current_player, temperature)
        
        # Store encoded state (2, 6, 7)
        encoded_state = encode_board(obs, current_player)
        game_states.append((encoded_state, pi, current_player))
        
        # Step (this switches players internally and checks for win)
        obs, reward, done = env.step(action)
        
        if done:
            # The reward is from perspective of player who just moved
            if reward == 1:
                winner = current_player  # current_player (who just moved) won
            elif reward == 0:
                winner = 0  # draw
            break

    # Assign values from perspective of each stored state
    for state, pi, player in game_states:
        if winner == 0:
            value = 0.0
        else:
            value = 1.0 if winner == player else -1.0
        replay_buffer.push(state, pi, value)

    return winner