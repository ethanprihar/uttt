import numpy as np

from .Constants import *


class Cell:
    def __init__(self, n):
        self.n = n
        self.board = np.zeros((n, n, CHANNELS))
        self.board[:, :, EMPTY_CHANNEL] = 1
        self.open_cell = True
        self.self_win = False
        self.opponent_win = False
        self.tie = False

    def available_moves(self):
        if self.open_cell:
            return self.board[:, :, EMPTY_CHANNEL].flatten().astype(bool)
        else:
            return np.zeros((self.n ** 2,)).astype(bool)

    def move(self, move):
        self.board[move.sub_row, move.sub_column, SELF_CHANNEL] = 1
        self.board[move.sub_row, move.sub_column, EMPTY_CHANNEL] = 0
        if self.check_self_win():
            self.self_win = True
            self.open_cell = False
        elif self.check_opponent_win():
            self.opponent_win = True
            self.open_cell = False
        elif self.check_tie():
            self.tie = True
            self.open_cell = False

    def switch_self(self):
        channels_to_flip = [SELF_CHANNEL, OPPONENT_CHANNEL]
        self.board[:, :, channels_to_flip] = np.flip(self.board[:, :, channels_to_flip], axis=2)
        if self.self_win:
            self.self_win = False
            self.opponent_win = True
        elif self.opponent_win:
            self.opponent_win = False
            self.self_win = True

    def nn_input(self):
        open_cell_channel = np.ones((self.n, self.n, 1)) * int(self.open_cell)
        self_win_channel = np.ones((self.n, self.n, 1)) * int(self.self_win)
        opponent_win_channel = np.ones((self.n, self.n, 1)) * int(self.opponent_win)
        tie_channel = np.ones((self.n, self.n, 1)) * int(self.tie)
        return np.concatenate([self.board,
                               open_cell_channel,
                               self_win_channel,
                               opponent_win_channel,
                               tie_channel], axis=2)

    # Private Functions
    def check_self_win(self):
        self_board = self.board[:, :, SELF_CHANNEL]
        rows = self_board.sum(axis=1).tolist()
        columns = self_board.sum(axis=0).tolist()
        diagonals = [self_board.diagonal().sum(), np.fliplr(self_board).diagonal().sum()]
        return max(rows + columns + diagonals) == self.n

    def check_opponent_win(self):
        self_board = self.board[:, :, OPPONENT_CHANNEL]
        rows = self_board.sum(axis=1).tolist()
        columns = self_board.sum(axis=0).tolist()
        diagonals = [self_board.diagonal().sum(), np.fliplr(self_board).diagonal().sum()]
        return max(rows + columns + diagonals) == self.n

    def check_tie(self):
        return self.board[:, :, EMPTY_CHANNEL].sum() == 0
