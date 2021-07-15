import os
import sys
import random
import colorama

from dqn import DQNModel
from game_utilities import Board, Move

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
colorama.init(autoreset=True)

model = DQNModel(board=Board(), purpose='playing')
if len(sys.argv) == 2:
    model.load(sys.argv[1])

computer_first = random.random() < 0.5
game_over = False

board = Board()
board.print()
if computer_first:
    move = model.get_move(board)
    board.move(move)
    board.print()
    board.switch_self()
    game_over = not board.open_board
while not game_over:
    valid_move = False
    while not valid_move:
        string_move = input('move (super_row super_column sub_row sub_column): ')
        try:
            move = Move(*[int(m) - 1 for m in string_move.split()])
            if board.check_valid_move(move):
                board.move(move)
                if computer_first:
                    board.switch_self()
                    board.print()
                else:
                    board.print()
                    board.switch_self()
                valid_move = True
            else:
                print('bad move')
        except:
            print('bad input')
    if board.open_board:
        move = model.get_move(board)
        board.move(move)
        if computer_first:
            board.print()
            board.switch_self()
        else:
            board.switch_self()
            board.print()
    game_over = not board.open_board

print('Game Over')
