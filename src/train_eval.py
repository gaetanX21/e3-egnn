import torch
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_loss(predictions, targets):
    """
    Compute the loss between predicted values and the true target values.
    
    Parameters:
    - predictions (Tensor): The model's predicted values (shape: [batch_size, num_properties]).
    - targets (Tensor): The true target values (shape: [batch_size, num_properties]).
    
    Returns:
    - loss (Tensor): The computed MSE loss.
    """
    loss = F.mse_loss(predictions, targets)
    return loss

def train_model(model, train_loader, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in train_loader:
            data = data.to(device)  # Move to GPU if available
            optimizer.zero_grad()
            out = model(data)
            loss = compute_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

def evaluate_model(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = compute_loss(out, data.y)
            total_loss += loss.item()
            pred = (out > 0.5).float()  # Binary prediction
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    return total_loss / len(loader), correct / total