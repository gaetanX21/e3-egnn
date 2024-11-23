import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import wandb


def train_model(model, train_loader, val_loader, optimizer, device, epochs, scheduler, use_wandb):
    model.train()
    train_losses, val_losses = [], []
    for epoch in tqdm(range(epochs)):
        if scheduler is not None:
            lr = scheduler.optimizer.param_groups[0]['lr']
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        val_loss = evaluate_model(model, val_loader, device)
        if scheduler is not None:
            scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if use_wandb:
            wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'lr': lr})
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


def run_experiment(model, train_loader, val_loader, device, n_epochs, lr=1e-3, use_wandb=False, use_scheduler=False):
    """
    Run a training experiment for a given model.
    """
    print(f"Running experiment for {model.__class__.__name__}, training on {len(train_loader.dataset)} samples for {n_epochs} epochs.")
    print("\nModel architecture:")
    print(model)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f'Total parameters: {total_param}')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, min_lr=1e-5)
    else:
        scheduler = None
    
    if use_wandb:
        wandb.init(
            project = 'egnn',
            config = {
                'model': model.__class__.__name__,
                'lr': lr,
                'n_epochs': n_epochs,
                'total_param': total_param
            }
        )
    print('\nTraining started.')
    train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, device, n_epochs, scheduler, use_wandb)
    print('Training finished.')
    return train_losses, val_losses