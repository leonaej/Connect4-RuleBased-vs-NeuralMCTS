import numpy as np
import math
import random



def score_window(window, player):
    """
    Scores a 4-cell window.
    Higher score = better for 'player'.
    """
    opp = -player
    score = 0

    if window.count(player) == 4:
        score += 100
    elif window.count(player) == 3 and window.count(0) == 1:
        score += 5
    elif window.count(player) == 2 and window.count(0) == 2:
        score += 2

    if window.count(opp) == 3 and window.count(0) == 1:
        score -= 4  # block opponent

    return score



def evaluate_board(board, player):
   
    score = 0
    rows, cols = board.shape

    # Score center column (players prefer middle)
    center_col = list(board[:, cols // 2])
    score += center_col.count(player) * 3

    #horizontal
    for r in range(rows):
        row_arr = list(board[r])
        for c in range(cols - 3):
            window = row_arr[c:c+4]
            score += score_window(window, player)

    # Vertical
    for c in range(cols):
        col_arr = list(board[:, c])
        for r in range(rows - 3):
            window = col_arr[r:r+4]
            score += score_window(window, player)

    #positive diagonal (/)
    for r in range(rows - 3):
        for c in range(cols - 3):
            window = [board[r+i][c+i] for i in range(4)]
            score += score_window(window, player)

    #negative diagonal (\)
    for r in range(rows - 3):
        for c in range(cols - 3):
            window = [board[r+3-i][c+i] for i in range(4)]
            score += score_window(window, player)

    return score



def check_win(board, player):
    """Returns True if player has won."""
    rows, cols = board.shape

    #horizontal
    for r in range(rows):
        for c in range(cols - 3):
            if all(board[r][c+i] == player for i in range(4)):
                return True

    # Vertical
    for c in range(cols):
        for r in range(rows - 3):
            if all(board[r+i][c] == player for i in range(4)):
                return True

    # positive diagonal
    for r in range(rows - 3):
        for c in range(cols - 3):
            if all(board[r+i][c+i] == player for i in range(4)):
                return True

    #negative diagonal
    for r in range(3, rows):
        for c in range(cols - 3):
            if all(board[r-i][c+i] == player for i in range(4)):
                return True

    return False












#minimax and alpha beta pruning 

def minimax(board, depth, alpha, beta, maximizing, player, env):
    valid_moves=env.get_valid_actions()

    #terminal checks here
    if check_win(board, player):
        return(999999, None)
    if check_win(board, -player):
        return(-999999, None)
    if depth==0 or len(valid_moves)==0:
        return (evaluate_board(board,player),None)
    

    best_move=random.choice(valid_moves)

    if maximizing:
        max_eval=-math.inf
        for col in valid_moves:
            temp_board=board.copy()

            for r in range (env.rows-1, -1,-1):
                 if temp_board[r][col]==0:
                     temp_board[r][col]=player
                     break
                 
            eval_score,_=minimax(temp_board, depth-1, alpha, beta, False, player,env)

            if eval_score>max_eval:
                max_eval=eval_score
                best_move=col
            
            alpha= max(alpha, eval_score)
            if alpha >=beta:
                break

        return max_eval, best_move
    
    else:
        min_eval=math.inf
        for col in valid_moves:
            temp_board=board.copy()
            for r in range (env.rows-1, -1, -1):
                if temp_board[r][col]==0:
                    temp_board[r][col]=-player
                    break
            
            eval_score,_=minimax(temp_board,depth-1,alpha,beta, True, player,env)

            if eval_score<min_eval:
                min_eval=eval_score
                best_move=col

            beta=min(beta, eval_score)
            if alpha >=beta:
                break
            
        return min_eval, best_move


            

  

#heuristic agent class

class HeuristicAgent:
    def __init__(self, depth=3):
        self.depth = depth

    def select_action(self, env):
        valid_moves = env.get_valid_actions()
        score, best_move = minimax(
            env.board.copy(),
            self.depth,
            -math.inf,
            math.inf,
            True,
            env.current_player,
            env
        )
        return best_move
    
