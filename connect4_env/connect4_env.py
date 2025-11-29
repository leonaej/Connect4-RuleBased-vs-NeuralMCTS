import numpy as np


class Connect4Env:

    def __init__(self):
        self.rows=6
        self.cols=7
        self.current_player=1
        self.board=np.zeros((self.rows, self.cols),dtype=int)

    def reset(self):
        self.board=np.zeros((self.rows, self.cols), dtype=int)
        self.current_player=1
        return self.board.copy()
    
    def get_valid_actions(self):
        valid=[]
        for c in range(self.cols):
            if self.board[0][c]==0:
                valid.append(c)
        return valid
    
    def step (self, action):
        for r in range(self.rows -1, -1, -1):
            if self.board[r][action]==0:
                self.board[r][action]=self.current_player
                break

        self.current_player = -1 if self.current_player==1 else 1

        done= len(self.get_valid_actions())==0
        reward=0
        return self.board.copy(), reward, done


   
    def check_win(self, player):
        board = self.board
        # horizontal
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if all(board[r][c+i] == player for i in range(4)):
                    return True
        # vertical
        for c in range(self.cols):
            for r in range(self.rows - 3):
                if all(board[r+i][c] == player for i in range(4)):
                    return True
        # positive diagonal
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if all(board[r+i][c+i] == player for i in range(4)):
                    return True
        # negative diagonal
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if all(board[r-i][c+i] == player for i in range(4)):
                    return True
        return False

    def is_draw(self):
        return len(self.get_valid_actions()) == 0