import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def train_model(model, train_loader, val_loader, optimizer, device, epochs):
    model.train()
    train_losses, val_losses = [], []
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for data in train_loader:
            data = data.to(device)  # Move to GPU if available
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        val_loss = evaluate_model(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    return train_losses, val_losses


def evaluate_model(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = F.mse_loss(out, data.y)
            total_loss += loss.item()
    loss = total_loss / len(loader)
    return loss


def run_experiment(model, model_name, train_loader, val_loader, device, n_epochs):
    """
    Run a training experiment for a given model.
    """

    print(f"Running experiment for {model_name}, training on {len(train_loader)} samples for {n_epochs} epochs.")
    
    print("\nModel architecture:")
    print(model)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f'Total parameters: {total_param}')
    model = model.to(device)

    # Adam optimizer with LR 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print('\nTraining started.')
    train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, device, n_epochs)
    print('Training finished.')
    return train_losses, val_losses