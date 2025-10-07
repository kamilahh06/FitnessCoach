import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class Trainer:
    """
    Trainer class for training and evaluating PyTorch models.
    Args:
        model: PyTorch model to train.
        train_dataset: Training dataset (PyTorch Dataset).
        test_dataset: Testing dataset (PyTorch Dataset).
        lr: Learning rate for the optimizer.
        batch_size: Batch size for training and evaluation.
        device: Device to run the model on ('cuda' or 'cpu'). If None, automatically selects.
    """
    def __init__(self, model, train_dataset, test_dataset, lr=1e-3, batch_size=32, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, epochs=10):
        for epoch in range(epochs):
            self.model.train()
            total_loss, correct, total = 0, 0, 0
            
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

            acc = 100 * correct / total
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(self.train_loader):.4f}, Acc: {acc:.2f}%")

    def evaluate(self):
        self.model.eval()
        correct, total, eval_loss = 0, 0, 0
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                eval_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        print(f"Test Loss: {eval_loss/len(self.test_loader):.4f}, Test Accuracy: {100 * correct / total:.2f}%")