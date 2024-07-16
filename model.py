# code block 5 : construct model
import torch
import os
import numpy as np

def construct_model():
    print('New Model Construction Started')
    # training - check dimension
    input_dim = 0
    for path, dirs, files in os.walk(os.path.join('data', 'training', 'training_stock', 'X')):
        for filename in files:
            file_path = os.path.join(path, filename)
            X = np.load(file_path)
            input_dim = X.shape[1]
            break

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    new_model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.1),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.1),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.1),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.1),
        torch.nn.Linear(32, 1),
        torch.nn.Sigmoid(),
    )
    print('New Model Construction finished')
    return device, new_model

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, std=0.01)

device, model = construct_model()
model.apply(init_weights)
model.to(device)