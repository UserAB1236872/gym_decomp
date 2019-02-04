import numpy as np


class QWorld(object):
    def __init__(self, world, np_random):
        self.__np_random = np_random
        self.__world = world
        self.__shape = world.shape
        self.__actions = world.actions
        self.__reward_types = [*world.rewards.keys()]

    @property
    def np_random(self):
        return self.__np_random

    @np_random.setter
    def np_random(self, val):
        self.__np_random = val

    @property
    def shape(self):
        return self.__shape

    @property
    def actions(self):
        return self.__actions

    @property
    def reward_types(self):
        return self.__reward_types

    def reset(self):
        non_terms = [*self.__world.nonterminal_states()]
        idx = self.np_random.choice(len(non_terms), None)
        return non_terms[idx]

    def act(self, state, action):
        cum_prob = 0.0
        roll = self.np_random.rand()

        for nxt in self.__world.successors(state, action):
            cum_prob += self.__world.transition_prob(
                state, action, nxt)
            if roll < cum_prob:
                rewards = {}
                for (typ, vals) in self.__world.rewards.items():
                    rewards[typ] = vals[nxt]
                total = self.__world.total_reward[nxt]

                terminal = self.__world.terminals[nxt]

                return nxt, rewards, total, terminal

        raise Exception("No successor state")

    def statify(self, state):
        return self.__world.statify(state)

    def valid(self, state):
        return (0 <= state[0] <= self.shape[0]) \
               and (0 <= state[1] <= self.shape[1]) \
               and not self.__world.impassable[state]

    @property
    def states(self):
        return [self.__world.statify(s).flatten() for s in self.__world.states]

    def destatify(self,state):
        i = int(np.where(state == 1)[0][0])
        cord_state = (int(i / 5), i - int(i / 5) * 5)
        assert (self.statify(cord_state).flatten() == state).all()
        return cord_state

    def transition_prob(self, state, action, next_state):
        action = self.__actions[action]
        cord_state,cord_next_state = self.destatify(state), self.destatify(next_state)
        return self.__world.transition_prob(cord_state, action, cord_next_state)

    def reward(self, state):
        cord_state = self.destatify(state)
        reward_info = {}
        reward = 0
        for (typ, vals) in self.__world.rewards.items():
            reward_info[typ] = vals[cord_state]
            reward += vals[cord_state]
        return reward, reward_info
