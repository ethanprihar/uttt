from collections import namedtuple


Sample = namedtuple('Sample', field_names=['state', 'action', 'reward', 'next_state', 'end_flag'])
