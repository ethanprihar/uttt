import sys
import colorama

from dqn import *
from game_utilities import *


colorama.init(autoreset=True)

model = DQNModel(board=Board(), training=True)
if len(sys.argv) == 2:
    model.load(sys.argv[1])

'''
while True:
    board = Board()
    extra_print = False
    if not model.backup_count % model.backup_frequency:
        board.print()
        extra_print = True
    while board.open_board:
        model.move(board)
        if extra_print and not board.first_players_turn():
            board.print()
        board.switch_self()
        if extra_print and board.first_players_turn():
            board.print()
'''

while True:
    board = Board()
    while board.open_board:
        model.move(board)
        board.switch_self()
