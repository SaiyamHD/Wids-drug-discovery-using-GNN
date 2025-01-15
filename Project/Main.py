import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

import torch
from torch_geometric.loader import DataLoader 
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, global_mean_pool


class LipophilicityGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LipophilicityGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        x = x.float()  
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        x = self.fc1(x).relu()
        x = self.dropout(x)
        x = self.fc2(x)
        return x
      
def load_dataset():
    dataset = MoleculeNet(root="./data", name="lipo")
    dataset.edge_attr = dataset.edge_attr.to(torch.float32)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    return DataLoader(train_dataset, batch_size=64, shuffle=True), DataLoader(test_dataset, batch_size=64, shuffle=False)


def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out.squeeze(), batch.y.view(-1))  
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate_model(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            y_true.append(batch.y.cpu().numpy())
            y_pred.append(out.squeeze().cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)  
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae
  
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 9  
    hidden_dim = 64

    
    train_loader, test_loader = load_dataset()
    model = LipophilicityGNN(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

  
    epochs = 50
    train_losses = []
    eval_metrics = []

    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        r2, rmse, mae = evaluate_model(model, test_loader, device)
        eval_metrics.append((r2, rmse, mae))

        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    epochs_range = range(1, epochs + 1)
    r2_scores, rmse_scores, mae_scores = zip(*eval_metrics)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs_range, train_losses, label="Training Loss")
    plt.plot(epochs_range, r2_scores, label="R2 Score")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.legend()
    plt.title("Training and Evaluation Metrics")
    plt.savefig("./outputs/performance_plot.png")
    plt.show()
