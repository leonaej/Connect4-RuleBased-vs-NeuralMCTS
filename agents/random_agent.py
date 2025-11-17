import random

class RandomAgent:
    def select_action(self, valid_actions):
        return random.choice(valid_actions)