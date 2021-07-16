import random
import numpy as np
from collections import deque


class RecordKeeper:
    def __init__(self, maximum_buffer_size):
        self.buffer = deque(maxlen=maximum_buffer_size)

    def record(self, sample):
        self.buffer.append(sample.tuple())

    def get_buffer_size(self):
        return len(self.buffer)

    def get_batch(self, batch_size):
        states, actions, rewards, next_states, open_flags = zip(*random.sample(self.buffer, batch_size))
        return np.stack(states), np.stack(actions), np.stack(rewards), np.stack(next_states), np.stack(open_flags)

    def get_all(self):
        states, actions, rewards, next_states, open_flags = zip(*self.buffer)
        return np.stack(states), np.stack(actions), np.stack(rewards), np.stack(next_states), np.stack(open_flags)
