#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd


class DiagnosisModel(nn.Module):
    def __init__(self, input_size=15):
        super(DiagnosisModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 32)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for inputs, targets in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate_model(model, loader):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            predictions = (outputs >= 0.5).float()
            all_predictions.append(predictions)
            all_targets.append(targets)
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    correct = (all_predictions == all_targets).sum().item()
    total = all_targets.size(0)
    accuracy = correct / total
    
    # Positive class metrics
    true_positives = ((all_predictions == 1) & (all_targets == 1)).sum().item()
    false_positives = ((all_predictions == 1) & (all_targets == 0)).sum().item()
    false_negatives = ((all_predictions == 0) & (all_targets == 1)).sum().item()
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    return accuracy, precision, recall


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load dataset
    data = pd.read_csv('/workspace/data/patient_records.csv')
    
    # Split features and target
    X = data[[f'feature_{i}' for i in range(1, 16)]].values
    y = data['diagnosis'].values
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)
    
    # Split train/test (last 4000 for test)
    X_train, X_test = X_tensor[:-4000], X_tensor[-4000:]
    y_train, y_test = y_tensor[:-4000], y_tensor[-4000:]
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model, criterion, and optimizer
    model = DiagnosisModel(input_size=15)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    num_epochs = 50
    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, train_loader, optimizer, criterion)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # Evaluate on test set
    accuracy, precision, recall = evaluate_model(model, test_loader)
    print(f"\nTest Set Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Positive Precision: {precision:.4f}")
    print(f"Positive Recall: {recall:.4f}")