import numpy as np
import torch
from agents.alphazero.mcts import MCTS
from agents.alphazero.az_network import AZNetwork
from training.train_network import NetworkTrainer 

class AlphaZeroAgent:
    def __init__(self, env, network=None, num_simulations=300, c_puct=1.5, device="cpu"):
        
        self.env = env
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct

        if network is None:
            self.network = AZNetwork().to(self.device)
        else:
            self.network = network.to(self.device)
        
        self.trainer = NetworkTrainer(self.network, lr=1e-3, device=self.device)
        
      

    def policy_from_root(self, board, current_player, temperature=1.0):
        """
        Run MCTS from (board, current_player) and return:
          - action: int chosen for play (argmax of visits if temp==0, else sample)
          - pi: 1D numpy array shape (7,) of visit-count distribution (normalized)
        """
        # MCTS expects env and network
        mcts = MCTS(env=self.env, network=self.network, c_puct=self.c_puct,
                    num_simulations=self.num_simulations, device=self.device)

        # run returns policy vector (visit count normalized)
        pi = mcts.run(root_state=board.copy(), current_player=current_player)

        # apply temperature:
        if temperature == 0:
            action = int(np.argmax(pi))
            return action, pi
        elif temperature == 1.0:
            probs = pi / (pi.sum() + 1e-12)
        else:
            # soften or sharpen distribution
            # raise to 1/temp then renormalize
            temp = max(1e-6, temperature)
            p = np.power(pi + 1e-12, 1.0 / temp)
            probs = p / (p.sum() + 1e-12)

        # Ensure valid distribution
        probs = np.array(probs, dtype=float)

        # If MCTS produced invalid or zero policy → fallback to uniform over valid moves
        if (probs < 0).any() or probs.sum() <= 1e-8 or np.isnan(probs).any():
            valid = self.env.get_valid_actions_from_board(board)
            probs = np.zeros_like(pi)
            probs[valid] = 1.0 / len(valid)

        else:
            # normalize for safety
            probs = probs / probs.sum()

        action = int(np.random.choice(len(probs), p=probs))
        return action, pi


    def select_action(self, env=None, temperature=1.0):
        """
        Convenience helper for play loops that pass the live env.
        Uses env.board and env.current_player by default.
        Returns only the action (for compatibility with other agents).
        """
        if env is None:
            env = self.env
        board = env.board.copy()
        current_player = env.current_player
        action, pi = self.policy_from_root(board, current_player, temperature=temperature)
        return action
