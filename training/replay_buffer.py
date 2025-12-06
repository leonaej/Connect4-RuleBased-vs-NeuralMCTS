import random
import torch
import os
import numpy as np
import pickle

class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state_tensor, policy_vector, value_scalar):
        """
        state_tensor: torch.FloatTensor (C,H,W) or numpy array
        policy_vector: length-7 numpy or list (visit counts normalized)
        value_scalar: float (-1,0,1)
        """
        item = (state_tensor, policy_vector, value_scalar)
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.position] = item
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, policies, values = zip(*batch)
        # convert states to torch tensors if needed
        states = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32) for s in states])
        policies = torch.tensor(policies, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        return states, policies, values

    def sample_batch(self, batch_size):
        """
        Return a batch dictionary with torch tensors suitable for NetworkTrainer.
        """
        batch_size = min(batch_size, len(self.buffer))
        batch_samples = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*batch_samples)

        # Stack states (already tensors from encode_board)
        states = torch.stack(list(states))
        
        # Convert policies and values to numpy first, then to tensor (faster)
        policies = torch.tensor(np.array(policies), dtype=torch.float32)
        values = torch.tensor(np.array(values), dtype=torch.float32)

        return {"states": states, "policies": policies, "values": values}
    
    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        d = {"buffer": self.buffer, "capacity": self.capacity, "position": self.position}
        with open(path, "wb") as f:
            pickle.dump(d, f)

    def load(self, path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                d = pickle.load(f)
            self.buffer = d.get("buffer", [])
            self.capacity = d.get("capacity", self.capacity)
            self.position = d.get("position", 0)
