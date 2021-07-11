# TODO stop only acting greedily

import os
import random
import pickle
import numpy as np
from random import sample
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from game_utilities import Move

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')


class Model:
    def __init__(self, board, rsf, df, sbs, spu, bpu, bs, br, wr, sc):
        self.random_settling_factor = rsf
        self.discount_factor = df
        self.sample_buffer_size = sbs
        self.samples_per_update = spu
        self.batches_per_update = bpu
        self.batch_size = bs
        self.base_reward = br
        self.win_reward = wr
        self.save_checkpoint = sc

        self.random_threshold = 1
        self.update_count = 0
        self.game_count = 0
        self.input_buffer = []
        self.target_buffer = []

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=board.nn_input().shape[1:]))
        self.model.add(Conv2D(filters=64, kernel_size=board.n, strides=board.n, activation='relu'))
        self.model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
        self.model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(units=512, activation='relu'))
        self.model.add(Dense(units=board.n ** 4, activation='linear'))
        self.model.compile(optimizer=Adam(learning_rate=0.00025, clipnorm=1.0), loss='huber')

    def move(self, board):
        self.input_buffer.append(board.nn_input())

        input_q_values = self.model.predict(board.nn_input()).flatten()
        if np.random.rand() > self.random_threshold:
            valid_moves = input_q_values.copy()
            valid_moves[np.invert(board.available_moves())] = -np.inf
            move_index = np.argmax(valid_moves)
        else:
            move_index = random.choice(np.where(board.available_moves())[0])
        super_cell_index = int(move_index / board.n ** 2)
        sub_cell_index = move_index % board.n ** 2
        move = Move(int(super_cell_index / board.n),
                    super_cell_index % board.n,
                    int(sub_cell_index / board.n),
                    sub_cell_index % board.n)
        self.random_threshold *= self.random_settling_factor

        board.move(move)

        reward = self.win_reward if board.self_win else self.base_reward
        expected_future_reward = np.max(self.model.predict(board.nn_input())) if board.open_board else 0
        target_q_values = input_q_values.copy()
        target_q_values[move_index] = reward + self.discount_factor * expected_future_reward

        self.target_buffer.append(target_q_values)

        if len(self.input_buffer) > self.sample_buffer_size:
            self.input_buffer.pop(0)
            self.target_buffer.pop(0)

        self.update_count += 1
        if self.update_count == self.samples_per_update:
            print('training')
            for i in range(self.batches_per_update):
                inputs, targets = zip(*sample(list(zip(self.input_buffer, self.target_buffer)), self.batch_size))
                self.model.train_on_batch(np.concatenate(inputs), np.stack(targets))
            self.update_count = 0

        if not board.open_board:
            self.game_count += 1
            print(f'game {self.game_count}')
            if self.game_count % self.save_checkpoint == 0:
                os.mkdir(f'checkpoints/model_{self.game_count}')
                with open(f'checkpoints/model_{self.game_count}/input_buffer.pkl', 'wb') as input_file:
                    pickle.dump(self.input_buffer, input_file)
                    with open(f'checkpoints/model_{self.game_count}/target_buffer.pkl', 'wb') as target_file:
                        pickle.dump(self.input_buffer, target_file)
                self.model.save(f'checkpoints/model_{self.game_count}/model_{self.game_count}')
