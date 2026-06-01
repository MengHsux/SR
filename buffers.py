import numpy as np
import torch


class ReplayBuffer():

    def __init__(self, max_size=np.inf):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def batch_push(self, batch_data):
        for data in batch_data: self.push(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), \
               np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

    def reset(self, max_size=np.inf):
        self.storage = []
        self.ptr = 0
        self.max_size = max_size


class DiffusionMemory():

    def __init__(self, max_size=np.inf):
        self.capacity = int(max_size)

        self.states = []
        self.best_actions = []
        self.max_size = self.capacity
        self.ptr = 0

    def push(self, state, action):
        if len(self.states) == self.capacity:
            self.states[int(self.ptr)] = state
            self.best_actions[int(self.ptr)] = action
            self.ptr = (self.ptr + 1) % self.capacity
        else:
            self.states.append(state)
            self.best_actions.append(action)

    def batch_push(self, batch_data):
        for state, action in batch_data: self.push(state, action)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.states), size=batch_size)
        states = [np.array(self.states[i], copy=False) for i in ind]
        best_actions = [np.array(self.best_actions[i], copy=False) for i in ind]

        return np.array(states), np.array(best_actions), np.array(ind)

    def replace(self, idxs, best_actions):
        for i, idx in enumerate(idxs):
            self.best_actions[idx] = best_actions[i]

    def replace1(self, idxs, states):
        for i, idx in enumerate(idxs):
            self.states[idx] = states[i]

    def reset(self, max_size=np.inf):
        self.states = []
        self.best_actions = []
        self.ptr = 0
        self.max_size = max_size
