

import torch
import torch.nn as nn
import torch.nn.functional as F


class AZNetwork2(nn.Module):
    
    def __init__(self, board_rows=6, board_cols=7, num_filters=64):
        super().__init__()

        self.board_rows = board_rows
        self.board_cols = board_cols

        # ---------- Shared Convolutional Body ----------
        self.conv_block = nn.Sequential(
            nn.Conv2d(2, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

        # TRUE residual blocks with 2 conv layers each
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(),
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_filters),
            )
            for _ in range(4)  # 4 residual blocks
        ])

        # ---------- Policy Head ----------
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )

        self.policy_fc = nn.Linear(2 * board_rows * board_cols, board_cols)

        # ---------- Value Head ----------
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

        self.value_fc1 = nn.Linear(board_rows * board_cols, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, state_tensor):
        

        x = self.conv_block(state_tensor)

        # Residual blocks WITH skip connections
        for block in self.res_blocks:
            identity = x  # Save input
            x = block(x)  # Apply transformations
            x = F.relu(x + identity)  # Add skip connection and activate

        # ---- Policy Head ----
        p = self.policy_head(x)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # ---- Value Head ----
        v = self.value_head(x)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v