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
              True)
if len(sys.argv) == 2:
    model.load(sys.argv[1])

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
