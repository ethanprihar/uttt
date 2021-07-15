class Sample:
    def __init__(self, state=None, action=None, reward=None, next_state=None, end_flag=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.end_flag = end_flag

    def tuple(self):
        return self.state, self.action, self.reward, self.next_state, self.end_flag
