import math
import numpy as np
import torch


class MCTSNode:
    def __init__(self, state, parent, prior_prob):
        self.state = state            # board array (6x7)
        self.parent = parent
        self.prior_prob = prior_prob  # p(a|s) from network

        self.children = {}            # action -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, env, network, c_puct=1.4, num_simulations=200, device="cpu"):
        self.env = env
        self.network = network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = device

    def expand(self, node, current_player):
        board_tensor = self._board_to_tensor(node.state, current_player)
        board_tensor = board_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.network(board_tensor)
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            value = value.item()

        valid_moves = self.env.get_valid_actions_from_board(node.state)

        # normalize over valid actions only
        policy = policy / (np.sum(policy[valid_moves]) + 1e-8)

        for action in valid_moves:
            # copy board
            next_board = self.env.apply_action(node.state, action, current_player)
            node.children[action] = MCTSNode(
                state=next_board,
                parent=node,
                prior_prob=policy[action],
            )

        return value

    def run(self, root_state, current_player):
        root = MCTSNode(root_state, parent=None, prior_prob=1)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            player = current_player

            # 1. Select until leaf
            while node.children:
                action, node = self.select_child(node)
                search_path.append(node)
                player = -player

            # 2. Check if game is terminal at this node
            # (This is important - prevents expanding terminal states)
            if self._is_terminal(node.state):
                # If terminal, value is 0 (draw) or determined by winner
                winner = self._get_winner(node.state)
                if winner == 0:
                    value = 0
                elif winner == player:
                    value = 1
                else:
                    value = -1
            else:
                # 3. Expand non-terminal leaf
                value = self.expand(node, player)

            # 4. Backprop - value is from 'player' perspective at leaf
            self.backpropagate(search_path, value, player)

        # At end: policy = visit counts
        actions = list(root.children.keys())
        visits = np.array([root.children[a].visit_count for a in actions])
        policy = np.zeros(7)
        policy[actions] = visits / np.sum(visits)

        return policy

    def select_child(self, node):
        best_score = -1e9
        best_action = None
        best_child = None

        parent_visits = node.visit_count + 1e-8

        for action, child in node.children.items():
            u = self.c_puct * child.prior_prob * math.sqrt(parent_visits) / (1 + child.visit_count)
            score = child.value + u

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def backpropagate(self, search_path, value, leaf_player):
        """
        Backpropagate value from leaf to root.
        
        Args:
            search_path: list of nodes from root to leaf
            value: value from leaf_player's perspective
            leaf_player: which player is at the leaf node
        """
        # Start with value from leaf player's perspective
        current_value = value
        
        # Traverse back up the tree
        # Each level up represents the PARENT's turn (opposite player)
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += current_value
            # Flip value for parent (opponent's perspective)
            current_value = -current_value

    def _board_to_tensor(self, board, current_player):
        # channel 0 = current player's pieces
        # channel 1 = opponent's pieces
        p1 = (board == current_player).astype(np.float32)
        p2 = (board == -current_player).astype(np.float32)
        return torch.tensor(np.stack([p1, p2], axis=0), dtype=torch.float32)

    def _is_terminal(self, board):
        """Check if board state is terminal (win or draw)"""
        # Check if anyone won
        if self._check_win_on_board(board, 1) or self._check_win_on_board(board, -1):
            return True
        # Check if board is full (draw)
        if len(self.env.get_valid_actions_from_board(board)) == 0:
            return True
        return False

    def _get_winner(self, board):
        """Return winner (1, -1) or 0 for draw"""
        if self._check_win_on_board(board, 1):
            return 1
        if self._check_win_on_board(board, -1):
            return -1
        return 0

    def _check_win_on_board(self, board, player):
        """Check if player has won on given board"""
        rows, cols = board.shape
        # horizontal
        for r in range(rows):
            for c in range(cols - 3):
                if all(board[r][c+i] == player for i in range(4)):
                    return True
        # vertical
        for c in range(cols):
            for r in range(rows - 3):
                if all(board[r+i][c] == player for i in range(4)):
                    return True
        # positive diagonal
        for r in range(rows - 3):
            for c in range(cols - 3):
                if all(board[r+i][c+i] == player for i in range(4)):
                    return True
        # negative diagonal
        for r in range(3, rows):
            for c in range(cols - 3):
                if all(board[r-i][c+i] == player for i in range(4)):
                    return True
        return False