import sys
import random
import colorama
from game_utilities import *
from dqn import *

colorama.init(autoreset=True)

# Game Constants
N = 3

# Model Constants
RANDOM_SETTLING_FACTOR = 1 - 1 / 2 ** 18
DISCOUNT_FACTOR = 1 - 1 / 2 ** 6
SAMPLE_BUFFER_SIZE = 2 ** 17
SAMPLES_PER_UPDATE = 2 ** 7
BATCHES_PER_UPDATE = 2 ** 5
BATCH_SIZE = 2 ** 5
BASE_REWARD = -2 ** 0
WIN_REWARD = 2 ** 7
SAVE_CHECKPOINT = 2 ** 10

model = Model(Board(N),
              RANDOM_SETTLING_FACTOR,
              DISCOUNT_FACTOR,
              SAMPLE_BUFFER_SIZE,
              SAMPLES_PER_UPDATE,
              BATCHES_PER_UPDATE,
              BATCH_SIZE,
              BASE_REWARD,
              WIN_REWARD,
              SAVE_CHECKPOINT,
              False)
model.load(sys.argv[1])
board = Board(N)

computer_first = random.random() < 0.5
game_over = False

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
