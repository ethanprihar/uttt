import os
import pickle
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model, clone_model, load_model

from game_utilities import Move
from .RecordKeeper import RecordKeeper

MAXIMUM_BUFFER_SIZE = 2 ** 17
BATCH_SIZE = 2 ** 5
INITIAL_RANDOM_CHANCE = 2 ** 0
RANDOM_CHANCE_GAIN = 1 - 1 / 2 ** 18
MINIMUM_RANDOM_CHANCE = 1 / 2 ** 6
WIN_REWARD = 2 ** 7
CELL_REWARD = 2 ** 4
BASE_REWARD = -2 ** 0
DISCOUNT_FACTOR = 1 - 1 / 2 ** 6
UPDATE_FREQUENCY = 2 ** 2
TRANSFER_FREQUENCY = 2 ** 8
BACKUP_FREQUENCY = 2 ** 10


class DQNModel:
    def __init__(self,
                 board,
                 training,
                 maximum_buffer_size=MAXIMUM_BUFFER_SIZE,
                 batch_size=BATCH_SIZE,
                 initial_random_chance=INITIAL_RANDOM_CHANCE,
                 random_chance_gain=RANDOM_CHANCE_GAIN,
                 minimum_random_chance=MINIMUM_RANDOM_CHANCE,
                 win_reward=WIN_REWARD,
                 cell_reward=CELL_REWARD,
                 base_reward=BASE_REWARD,
                 discount_factor=DISCOUNT_FACTOR,
                 update_frequency=UPDATE_FREQUENCY,
                 transfer_frequency=TRANSFER_FREQUENCY,
                 backup_frequency=BACKUP_FREQUENCY):
        self.n = board.n
        self.training = training
        self.record_keeper = RecordKeeper(maximum_buffer_size, batch_size)
        self.random_chance = initial_random_chance
        self.random_chance_gain = random_chance_gain
        self.minimum_random_chance = minimum_random_chance
        self.win_reward = win_reward
        self.cell_reward = cell_reward
        self.base_reward = base_reward
        self.discount_factor = discount_factor
        self.update_frequency = update_frequency
        self.transfer_frequency = transfer_frequency
        self.backup_frequency = backup_frequency
        self.update_count = 0
        self.transfer_count = 0
        self.backup_count = 0
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

    def move(self, board):
        if not self.training:
            prediction = np.concatenate(self.main_model.predict(np.expand_dims(board.get_state(), axis=0))).flatten()
            prediction[np.invert(board.available_moves())] = -np.inf
            action_index = np.argmax(prediction)
            super_cell_index = int(action_index / self.n ** 2)
            sub_cell_index = action_index % self.n ** 2
            move = Move(int(super_cell_index / self.n),
                        super_cell_index % self.n,
                        int(sub_cell_index / self.n),
                        sub_cell_index % self.n)
            board.move(move)

        else:
            # state
            state = board.get_state()

            # action
            if np.random.rand() > self.random_chance:
                prediction = np.concatenate(self.main_model.predict(np.expand_dims(state, axis=0))).flatten()
                prediction[np.invert(board.available_moves())] = -np.inf
                action_index = np.argmax(prediction)
            else:
                action_index = np.random.choice(np.where(board.available_moves())[0])
            action = np.zeros((self.n ** 4))
            action[action_index] = 1
            self.random_chance = np.max([self.random_chance * self.random_chance_gain, self.minimum_random_chance])

            # move
            super_cell_index = int(action_index / self.n ** 2)
            sub_cell_index = action_index % self.n ** 2
            move = Move(int(super_cell_index / self.n),
                        super_cell_index % self.n,
                        int(sub_cell_index / self.n),
                        sub_cell_index % self.n)
            board.move(move)

            # reward
            reward = [self.get_reward(board, move)]

            # next_state
            next_state = board.get_state()

            # end_flag
            end_flag = [int(not board.open_board)]

            # record sample
            self.record_keeper.record(state, action, reward, next_state, end_flag)

            # perform scheduled tasks
            self.update_count += 1
            self.transfer_count += 1
            self.backup_count += 1 if end_flag[0] else 0
            if end_flag[0]:
                print(f'{self.backup_count} games completed')
            update_check_1 = self.update_count >= self.update_frequency
            update_check_2 = self.record_keeper.get_buffer_size() >= self.record_keeper.batch_size
            if update_check_1 and update_check_2:
                self.update_main_model()
                self.update_count = 0
            if self.transfer_count >= self.transfer_frequency:
                self.update_target_model()
                self.transfer_count = 0
            if self.backup_count % self.backup_frequency == 0 and end_flag[0]:
                self.save()

    def get_reward(self, board, move):
        if board.self_win:
            return self.win_reward
        elif board.board[move.super_row, move.super_column].self_win:
            return self.cell_reward
        else:
            return self.base_reward

    def update_main_model(self):
        states, actions, rewards, next_states, end_flags = self.record_keeper.get_batch()
        target_model_prediction = np.concatenate(self.target_model.predict(next_states), axis=1)
        expected_reward = np.max(target_model_prediction, axis=1).reshape(-1, 1) * end_flags
        target_matrix = actions * (self.discount_factor * expected_reward + rewards)
        targets = [target_matrix[:, i] for i in range(self.n ** 4)]
        weights = dict(zip(self.main_model.output_names, [actions[:, i] for i in range(self.n ** 4)]))
        self.main_model.train_on_batch(states, targets, sample_weight=weights)

    def update_target_model(self):
        self.target_model.set_weights(self.main_model.get_weights())

    def save(self):
        os.mkdir(f'checkpoints/model_{self.backup_count}')
        with open(f'checkpoints/model_{self.backup_count}/param.pkl', 'wb') as file:
            no_model_dict = {key: self.__dict__[key] for key in self.__dict__.keys()
                             if key not in ['main_model', 'target_model', 'training']}
            pickle.dump(no_model_dict, file)
        self.main_model.save(f'checkpoints/model_{self.backup_count}/main_model')
        self.target_model.save(f'checkpoints/model_{self.backup_count}/target_model')

    def load(self, path):
        with open(f'{path}/param.pkl', 'rb') as file:
            no_model_dict = pickle.load(file)
            self.__dict__.update(no_model_dict)
        self.main_model = load_model(f'{path}/main_model')
        self.target_model = load_model(f'{path}/target_model')
