import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NetworkTrainer:
    def __init__(self, network, lr=1e-3, weight_decay=1e-4, device="cpu"):
        self.network = network.to(device)
        self.device = device

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()

    def train_step(self, batch):
        
        # States are already tensors from encode_board, just move to device
        states = batch["states"].to(self.device)
        target_policies = batch["policies"].to(self.device)
        target_values = batch["values"].to(self.device)

        self.optimizer.zero_grad()

        pred_policies, pred_values = self.network(states)

        # Cross entropy for policy (soft-target CE for distributions)
        policy_loss = -(target_policies * torch.log_softmax(pred_policies, dim=1)).sum(dim=1).mean()

        # Value MSE
        value_loss = self.value_loss_fn(pred_values.squeeze(), target_values)

        loss = policy_loss + value_loss

        loss.backward()
        self.optimizer.step()

        return float(loss.item()), float(policy_loss.item()), float(value_loss.item())


    def save_checkpoint(self, path):
        torch.save(self.network.state_dict(), path)


def train_one_iteration(trainer, replay_buffer, batch_size=64, batches_per_iter=100):
    
    import random

    losses = []
    policy_losses = []
    value_losses = []

    for _ in range(batches_per_iter):
        batch = replay_buffer.sample(batch_size)

        loss, pol_loss, val_loss = trainer.train_step(batch)
        losses.append(loss)
        policy_losses.append(pol_loss)
        value_losses.append(val_loss)

    return (
        float(np.mean(losses)),
        float(np.mean(policy_losses)),
        float(np.mean(value_losses)),
    )
