
import torch
import random

def train_one_iteration(agent, replay_buffer, batch_size=64, steps=50):
    
    if len(replay_buffer) < batch_size:
        return 0.0  # not enough samples yet

    total_loss = 0

    for _ in range(steps):
        batch = replay_buffer.sample_batch(batch_size)
        loss, ploss, vloss = agent.trainer.train_step(batch)
        total_loss += loss

    return total_loss / steps
