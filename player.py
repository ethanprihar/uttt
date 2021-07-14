import sys
import random
import colorama

from dqn import *
from game_utilities import *


colorama.init(autoreset=True)

model = DQNModel(board=Board(), training=False)
if len(sys.argv) == 2:
    model.load(sys.argv[1])

computer_first = random.random() < 0.5
game_over = False

board = Board()
board.print()
if computer_first:
    model.move(board)
    board.print()
    game_over = not board.open_board
while not game_over:
    valid_move = False
    while not valid_move:
        string_move = input('move (super_row super_column sub_row sub_column): ')
        try:
            move = Move(*[int(m) - 1 for m in string_move.split()])
            if board.check_valid_move(move):
                board.switch_self()
                board.move(move)
                board.switch_self()
                board.print()
                valid_move = True
            else:
                print('bad move')
        except:
            print('bad input')
    if board.open_board:
        model.move(board)
        board.print()
    game_over = not board.open_board

if board.self_win:
    print('Computer Wins')
elif board.opponent_win:
    print('Human Wins')
else:
    print('Tie Game')
