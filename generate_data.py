import random
import numpy as np
import torch

# Data Generation
np.random.seed(42) # Comment out for random results.
x = np.random.rand(100, 1)

# y = 1 + 2x + Gaussian noise
y = 1 + 2 * x + .1 * np.random.randn(100, 1)


# Shuffles the indices to split train and validation datasets
idx = np.arange(100)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:80]

# Uses the remaining indices for validation
val_idx = idx[80:]

# Generates train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)