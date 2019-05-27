from collections import deque

import numpy as np

class AudioVaultPlayer(object):

    def __init__(self, n_hypothesis, positive_feedback_mp3_list, negative_feedback_mp3_list, target_index=None):
        self.n_hypothesis = n_hypothesis
        self.positive_feedback_mp3_deque = deque(positive_feedback_mp3_list)
        self.negative_feedback_mp3_deque = deque(negative_feedback_mp3_list)
        self.update_target_index(target_index)

    def update_target_index(self, new_target_index):
        if new_target_index is None:  # set it randomly
            self.target_index = np.random.randint(self.n_hypothesis)
        else:
            self.target_index = new_target_index

    def get_feedback_mp3(self, flash_pattern):
        is_target_flashed = flash_pattern[self.target_index]
        if is_target_flashed:
            deque_to_sample = self.positive_feedback_mp3_deque
        else:
            deque_to_sample = self.negative_feedback_mp3_deque

        deque_to_sample.rotate()
        return deque_to_sample[0]
