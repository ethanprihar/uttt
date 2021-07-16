import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model, clone_model, load_model

from .Sample import Sample
from game_utilities import Move
from .RecordKeeper import RecordKeeper

MAXIMUM_BUFFER_SIZE = 100000
BATCH_SIZE = 32
INITIAL_RANDOM_CHANCE = 1
RANDOM_CHANCE_GAIN = 0.9999954
MINIMUM_RANDOM_CHANCE = 0.01
WIN_REWARD = 1000
LOSE_REWARD = -1000
CELL_REWARD = 0
BASE_REWARD = -10
DISCOUNT_FACTOR = 0.95
UPDATE_FREQUENCY = 4
TRANSFER_FREQUENCY = 512
BACKUP_FREQUENCY = 10000


class DQNModel:
    def __init__(self,
                 board,
                 purpose,
                 maximum_buffer_size=MAXIMUM_BUFFER_SIZE,
                 batch_size=BATCH_SIZE,
                 initial_random_chance=INITIAL_RANDOM_CHANCE,
                 random_chance_gain=RANDOM_CHANCE_GAIN,
                 minimum_random_chance=MINIMUM_RANDOM_CHANCE,
                 win_reward=WIN_REWARD,
                 lose_reward=LOSE_REWARD,
                 cell_reward=CELL_REWARD,
                 base_reward=BASE_REWARD,
                 discount_factor=DISCOUNT_FACTOR,
                 update_frequency=UPDATE_FREQUENCY,
                 transfer_frequency=TRANSFER_FREQUENCY,
                 backup_frequency=BACKUP_FREQUENCY):
        self.n = board.n
        self.purpose = purpose
        self.record_keeper = RecordKeeper(maximum_buffer_size, batch_size)
        self.random_chance = initial_random_chance
        self.random_chance_gain = random_chance_gain
        self.minimum_random_chance = minimum_random_chance
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.cell_reward = cell_reward
        self.base_reward = base_reward
        self.discount_factor = discount_factor
        self.update_frequency = update_frequency
        self.transfer_frequency = transfer_frequency
        self.backup_frequency = backup_frequency
        self.current_record = Sample()
        self.update_count = 0
        self.transfer_count = 0
        self.backup_count = 0
        self.loss_list = []
        self.mae_list = []
        self.backup_count_list = []
        input_layer = Input(shape=board.get_state().shape)
        conv_layer_1 = Conv2D(filters=32,
                              kernel_size=board.n,
                              strides=board.n,
                              activation='relu',
                              kernel_initializer='he_normal',
                              bias_initializer='he_normal')(input_layer)
        conv_layer_2 = Conv2D(filters=64,
                              kernel_size=3,
                              strides=1,
                              activation='relu',
                              kernel_initializer='he_normal',
                              bias_initializer='he_normal')(conv_layer_1)
        flatten_layer = Flatten()(conv_layer_2)
        dense_layer = Dense(units=128,
                            activation='relu',
                            kernel_initializer='he_normal',
                            bias_initializer='he_normal')(flatten_layer)
        output_layers = [Dense(units=1,
                               activation='linear',
                               kernel_initializer='he_normal',
                               bias_initializer='he_normal',
                               name=f'output_{i}')(dense_layer)
                         for i in range(board.n ** 4)]
        self.main_model = Model(input_layer, output_layers)
        self.target_model = clone_model(self.main_model)
        self.main_model.compile(optimizer=Adam(learning_rate=0.00025, clipnorm=1.0),
                                loss='huber',
                                metrics=['mae', 'acc'])
        self.target_model.compile(optimizer=Adam(learning_rate=0.00025, clipnorm=1.0),
                                  loss='huber',
                                  metrics=['mae', 'acc'])

    def get_move(self, board):
        self.current_record = Sample()

        # state
        state = board.get_state()
        self.current_record.state = state

        # action
        if self.purpose == 'playing' or np.random.rand() > self.random_chance:
            prediction = np.concatenate(self.main_model.predict(np.expand_dims(state, axis=0))).flatten()
            prediction[np.invert(board.available_moves())] = -np.inf
            action_index = np.argmax(prediction)
        else:
            action_index = np.random.choice(np.where(board.available_moves())[0])
        self.random_chance = np.max([self.random_chance * self.random_chance_gain, self.minimum_random_chance])
        action = np.zeros((self.n ** 4))
        action[action_index] = 1
        self.current_record.action = action

        # move
        super_cell_index = int(action_index / self.n ** 2)
        sub_cell_index = action_index % self.n ** 2
        move = Move(int(super_cell_index / self.n),
                    super_cell_index % self.n,
                    int(sub_cell_index / self.n),
                    sub_cell_index % self.n)
        return move

    def set_result(self, board, move):
        if self.current_record.state is not None:

            # reward
            reward = [self.get_reward(board, move)]
            self.current_record.reward = reward

            # next_state
            next_state = board.get_state()
            self.current_record.next_state = next_state

            # open_flag
            open_flag = [int(board.open_board)]
            self.current_record.open_flag = open_flag

            # record sample
            self.record_keeper.record(self.current_record)
            self.current_record = Sample()

            # perform scheduled tasks
            if self.purpose == 'training':
                self.update_count += 2  # it's two because the other model is also playing
                self.transfer_count += 2  # it's two because the other model is also playing
                self.backup_count += 0 if open_flag[0] else 1
                if not open_flag[0]:
                    print(f'{self.backup_count} games completed')
                update_check_1 = self.update_count >= self.update_frequency
                update_check_2 = self.record_keeper.get_buffer_size() >= self.record_keeper.batch_size
                if update_check_1 and update_check_2:
                    self.update_main_model()
                    self.update_count = 0
                if self.transfer_count >= self.transfer_frequency:
                    self.update_target_model()
                    self.transfer_count = 0
                if self.backup_count % self.backup_frequency == 0 and not open_flag[0]:
                    self.save()

    def get_reward(self, board, move):
        if board.self_win:
            return self.win_reward
        elif board.opponent_win:
            return self.lose_reward
        elif board.board[move.super_row, move.super_column].self_win:
            return self.cell_reward
        else:
            return self.base_reward

    def update_main_model(self):
        states, actions, rewards, next_states, open_flags = self.record_keeper.get_batch()
        target_model_prediction = np.concatenate(self.target_model.predict(next_states), axis=1)
        expected_reward = np.max(target_model_prediction, axis=1).reshape(-1, 1) * open_flags
        target_matrix = actions * (self.discount_factor * expected_reward + rewards)
        targets = [target_matrix[:, i] for i in range(self.n ** 4)]
        weights = dict(zip(self.main_model.output_names, [actions[:, i] for i in range(self.n ** 4)]))
        self.main_model.train_on_batch(states, targets, sample_weight=weights)

    def update_target_model(self):
        self.target_model.set_weights(self.main_model.get_weights())

    def save(self):
        # make folder
        os.mkdir(f'checkpoints/model_{self.backup_count}')

        # same metrics
        states, actions, rewards, next_states, open_flags = self.record_keeper.get_batch()
        target_model_prediction = np.concatenate(self.target_model.predict(next_states), axis=1)
        expected_reward = np.max(target_model_prediction, axis=1).reshape(-1, 1) * open_flags
        target_matrix = actions * (self.discount_factor * expected_reward + rewards)
        targets = [target_matrix[:, i] for i in range(self.n ** 4)]
        weights = dict(zip(self.main_model.output_names, [actions[:, i] for i in range(self.n ** 4)]))
        evaluation = self.main_model.evaluate(states, targets, sample_weight=weights, verbose=0, return_dict=True)
        loss = evaluation['loss']
        mae = np.mean([evaluation[key] for key in evaluation.keys() if 'mae' in key])
        self.loss_list.append(loss)
        self.mae_list.append(mae)
        self.backup_count_list.append(self.backup_count)
        fig, ax_loss = plt.subplots()
        ax_loss.set_title(f'Training Progress at Game {self.backup_count}')
        ax_loss.set_xlabel('Games Played')
        ax_loss.set_ylabel('Loss', color='orange')
        ax_loss.plot(self.backup_count_list, self.loss_list, color='orange')
        ax_loss.tick_params(axis='y', labelcolor='orange')
        ax_mae = ax_loss.twinx()
        ax_mae.set_ylabel('MAE', color='blue')
        ax_loss.plot(self.backup_count_list, self.mae_list, color='blue')
        ax_loss.tick_params(axis='y', labelcolor='blue')
        fig.tight_layout()
        plt.savefig(f'checkpoints/model_{self.backup_count}/progress.png', dpi=300)
        print(f'loss: {loss}, mean absolute error: {mae}')

        # save backup
        with open(f'checkpoints/model_{self.backup_count}/param.pkl', 'wb') as file:
            model_dict = {key: self.__dict__[key] for key in self.__dict__.keys()
                          if key not in ['main_model', 'target_model', 'purpose']}
            pickle.dump(model_dict, file)
        self.main_model.save(f'checkpoints/model_{self.backup_count}/main_model')
        self.target_model.save(f'checkpoints/model_{self.backup_count}/target_model')

    def load(self, path):
        with open(f'{path}/param.pkl', 'rb') as file:
            model_dict = pickle.load(file)
            self.__dict__.update(model_dict)
        self.main_model = load_model(f'{path}/main_model')
        self.target_model = load_model(f'{path}/target_model')
