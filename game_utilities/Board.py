import numpy as np
import termcolor

from .Cell import Cell
from .Move import Move
from .Constants import SELF_CHANNEL, OPPONENT_CHANNEL

N = 3


class Board:
    def __init__(self, n=N):
        self.n = n
        self.board = np.empty((n, n), dtype=object)
        for r in range(n):
            for c in range(n):
                self.board[r, c] = Cell(n)
        self.open_board = True
        self.self_win = False
        self.opponent_win = False
        self.tie = False
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

    def get_move_index(self, move):
        move_index = self.n ** 3 * move.super_row
        move_index += self.n ** 2 * move.super_column
        move_index += self.n * move.sub_row
        move_index += move.sub_column
        return move_index

    def check_valid_move(self, move):
        return self.available_moves()[self.get_move_index(move)]

    def move(self, move):
        self.board[move.super_row, move.super_column].move(move)
        if self.check_self_win():
            self.self_win = True
            self.open_board = False
        elif self.check_opponent_win():
            self.opponent_win = True
            self.open_board = False
        elif self.check_tie():
            self.tie = True
            self.open_board = False
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

    def get_state(self):
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
        return np.concatenate(rows, axis=0)

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
                            elif self.check_valid_move(Move(sup_r, sup_c, sub_r, sub_c)):
                                output += termcolor.colored('*', 'cyan')
                            else:
                                output += '*'
                        else:
                            if cell_self_win:
                                output += 'X'
                            elif cell_opponent_win:
                                output += 'O'
                            else:
                                output += '#'
                    output += '|' if sup_c < self.n - 1 else ''
                output += '\n'
            output += ''.join(['-'] * (self.n ** 2 + self.n - 1)) + '\n' if sup_r < self.n - 1 else '\n'
        print(output)

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
