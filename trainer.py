import sys
import colorama

from dqn import DQNModel
from game_utilities import Board


colorama.init(autoreset=True)

training_model = DQNModel(board=Board(), purpose='training')
if len(sys.argv) == 2:
    training_model.load(sys.argv[1])
helper_model = DQNModel(board=Board(), purpose='helping')
helper_model.record_keeper = training_model.record_keeper
helper_model.main_model = training_model.main_model
helper_model.target_model = training_model.target_model

while True:
    board = Board()
    # board.print()
    helper_move = None
    while board.open_board:
        training_move = training_model.get_move(board)
        board.move(training_move)
        if not board.open_board:
            training_model.set_result(board, training_move)
        # board.print()
        board.switch_self()
        helper_model.set_result(board, helper_move)
        if board.open_board:
            helper_move = helper_model.get_move(board)
            board.move(helper_move)
            if not board.open_board:
                helper_model.set_result(board, helper_move)
            board.switch_self()
            # board.print()
            training_model.set_result(board, training_move)
