import numpy as np
from random import sample
from collections import deque

from .Sample import Sample


class RecordKeeper:
    def __init__(self, maximum_buffer_size, batch_size):
        self.buffer = deque(maxlen=maximum_buffer_size)
        self.batch_size = batch_size

    def record(self, state, action, reward, next_state, end_flag):
        self.buffer.append(Sample(state, action, reward, next_state, end_flag))

    def get_buffer_size(self):
        return len(self.buffer)

    def get_batch(self):
        states, actions, rewards, next_states, end_flags = zip(*sample(self.buffer, self.batch_size))
        return np.stack(states), np.stack(actions), np.stack(rewards), np.stack(next_states), np.stack(end_flags)
