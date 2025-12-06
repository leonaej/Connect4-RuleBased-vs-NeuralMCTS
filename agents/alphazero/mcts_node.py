
import math

class Node:
    def __init__(self, state, parent=None, prior=0.0):
        self.state = state               # board array or internal representation
        self.parent = parent             # parent node
        self.children = {}               # action -> Node

        self.prior = prior               # P(s, a) from network
        self.visit_count = 0             # N(s, a)
        self.value_sum = 0               # W(s, a)

        self.is_expanded = False         # has children been added?

    # Q(s,a) = W / N   (value estimate)
    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    # UCB = Q + U = Q + c * P * (sqrt(sum visits) / (1 + N))
    def ucb_score(self, parent_sum_visits, c_puct=1.0):
        u = c_puct * self.prior * math.sqrt(parent_sum_visits) / (1 + self.visit_count)
        return self.q_value + u

    # expand children using network policy
    def expand(self, action_priors, next_states):
        """
        action_priors: dict {action: prior_prob}
        next_states:   dict {action: next_state}
        """
        for action, prior in action_priors.items():
            self.children[action] = Node(
                state=next_states[action],
                parent=self,
                prior=prior
            )
        self.is_expanded = True

    # backprop values all the way up
    def backprop(self, value):
        """
        value: result from this state's perspective (win=1, loss=-1)
        """
        self.visit_count += 1
        self.value_sum += value

        if self.parent is not None:
            # value flips sign when going to parent (because turn alternates)
            self.parent.backprop(-value)
