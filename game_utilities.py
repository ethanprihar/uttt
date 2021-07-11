import numpy as np

# Channel Constants
CHANNELS = 3
SELF_CHANNEL = 0
OPPONENT_CHANNEL = 1
EMPTY_CHANNEL = 2


class Move:
    def __init__(self, super_row, super_column, sub_row, sub_column):
        self.super_row = super_row
        self.super_column = super_column
        self.sub_row = sub_row
        self.sub_column = sub_column


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
        return self.board[:, :, EMPTY_CHANNEL].flatten().astype(bool)

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


class Board:
    def __init__(self, n):
        self.n = n
        self.board = np.empty((n, n), dtype=object)
        for r in range(n):
            for c in range(n):
                self.board[r, c] = Cell(n)
        self.open_board = 1
        self.self_win = 0
        self.opponent_win = 0
        self.tie = 0
        self.last_move = None

    def available_moves(self):
        moves = np.zeros((self.n ** 4,)).astype(bool)
        if self.last_move is not None and self.board[self.last_move.sub_row, self.last_move.sub_column].open_cell:
            start = self.n ** 3 * self.last_move.sub_row + self.n ** 2 * self.last_move.sub_column
            end = start + self.n ** 2
            moves[start: end] = self.board[self.last_move.sub_row, self.last_move.sub_column].available_moves()
        else:
            for r in range(self.n):
                for c in range(self.n):
                    start = self.n ** 3 * r + self.n ** 2 * c
                    end = start + self.n ** 2
                    moves[start: end] = self.board[r, c].available_moves()
        return moves

    def move(self, move):
        self.board[move.super_row, move.super_column].move(move)
        if self.check_self_win():
            self.self_win = 1
            self.open_board = 0
        elif self.check_opponent_win():
            self.opponent_win = 1
            self.open_board = 0
        elif self.check_tie():
            self.tie = 1
            self.open_board = 0
        self.last_move = move

    def switch_self(self):
        for row in self.board:
            for cell in row:
                cell.switch_self()

    def first_players_turn(self):
        self_turns = 0
        opponent_turns = 0
        for r in range(self.n):
            for c in range(self.n):
                self_turns += np.sum(self.board[r, c].board[:, :, SELF_CHANNEL])
                opponent_turns += np.sum(self.board[r, c].board[:, :, OPPONENT_CHANNEL])
        return self_turns == opponent_turns

    def print(self):
        output = ''
        for sup_r in range(self.n):
            for sub_r in range(self.n):
                for sup_c in range(self.n):
                    for sub_c in range(self.n):
                        cell_open_cell = self.board[sup_r, sup_c].open_cell
                        cell_self_win = self.board[sup_r, sup_c].self_win
                        cell_opponent_win = self.board[sup_r, sup_c].opponent_win
                        self_val = self.board[sup_r, sup_c].board[sub_r, sub_c, SELF_CHANNEL]
                        opponent_val = self.board[sup_r, sup_c].board[sub_r, sub_c, OPPONENT_CHANNEL]
                        if cell_open_cell:
                            if self_val:
                                output += 'x'
                            elif opponent_val:
                                output += 'o'
                            else:
                                output += '*'
                        else:
                            if cell_self_win:
                                output += 'X'
                            elif cell_opponent_win:
                                output += 'O'
                            else:
                                output += '#'
                output += '\n'
        print(output)

    def nn_input(self):
        available_moves = self.available_moves()
        rows = []
        for r in range(self.n):
            cells = []
            for c in range(self.n):
                cell_nn_input = self.board[r, c].nn_input()
                start = self.n ** 3 * r + self.n ** 2 * c
                end = start + self.n ** 2
                move_options = available_moves[start:end].reshape((self.n, self.n, 1))
                cells.append(np.concatenate([cell_nn_input, move_options], axis=2))
            rows.append(np.concatenate(cells, axis=1))
        return np.expand_dims(np.concatenate(rows, axis=0), axis=0)

    # Private Functions
    def check_self_win(self):
        self_board = np.zeros((self.n, self.n))
        for r in range(self.n):
            for c in range(self.n):
                self_board[r, c] = int(self.board[r, c].self_win)
        rows = self_board.sum(axis=1).tolist()
        columns = self_board.sum(axis=0).tolist()
        diagonals = [self_board.diagonal().sum(), np.fliplr(self_board).diagonal().sum()]
        return max(rows + columns + diagonals) == self.n

    def check_opponent_win(self):
        opponent_board = np.zeros((self.n, self.n))
        for r in range(self.n):
            for c in range(self.n):
                opponent_board[r, c] = int(self.board[r, c].opponent_win)
        rows = opponent_board.sum(axis=1).tolist()
        columns = opponent_board.sum(axis=0).tolist()
        diagonals = [opponent_board.diagonal().sum(), np.fliplr(opponent_board).diagonal().sum()]
        return max(rows + columns + diagonals) == self.n

    def check_tie(self):
        open_cell_count = 0
        for r in range(self.n):
            for c in range(self.n):
                open_cell_count += int(self.board[r, c].open_cell)
        return open_cell_count == 0
