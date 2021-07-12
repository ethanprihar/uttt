import sys
from game_utilities import *
from dqn import *

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
TRAINING = bool(int(sys.argv[1]))

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
              TRAINING)

if len(sys.argv) == 3:
    model.load(sys.argv[2])

if TRAINING:
    while True:
        board = Board(N)
        extra_print = False
        if not model.game_count % model.save_checkpoint:
            board.print()
            extra_print = True
        while board.open_board:
            model.move(board)
            if extra_print and not board.first_players_turn():
                board.print()
            board.switch_self()
            if extra_print and board.first_players_turn():
                board.print()
else:
    board = Board(N)
    board.print()
    while board.open_board:
        model.move(board)
        board.print()
        valid_move = False
        while not valid_move:
            string_move = input('move (super_row super_column sub_row sub_column): ')
            move = Move(*[int(m) for m in string_move.split()])
            if board.check_valid_move(move):
                board.switch_self()
                board.move(move)
                board.switch_self()
                board.print()
                valid_move = True
