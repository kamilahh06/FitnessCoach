import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class RegressorTrainer:
    def __init__(self, model, train_dataset, test_dataset, lr=1e-3, batch_size=32, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader  = DataLoader(test_dataset, batch_size=batch_size)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _compute_metrics(self, preds, targets, tol=0.1):
        """Compute R², correlation, and approximate accuracy."""
        preds = preds.squeeze()
        targets = targets.squeeze()
        ss_res = torch.sum((targets - preds) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        if torch.std(preds) == 0 or torch.std(targets) == 0:
            corr = torch.tensor(0.0)
        else:
            corr = torch.corrcoef(torch.stack([targets, preds]))[0, 1]

        # Approximate accuracy: % predictions within `tol` range of target
        within_tol = torch.abs(preds - targets) <= tol
        acc = within_tol.float().mean() * 100

        return r2.item(), corr.item(), acc.item()

    def train(self, epochs=10):
        for epoch in range(epochs):
            self.model.train()
            total_loss, preds_all, y_all = 0, [], []

            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                preds = self.model(X)
                loss = self.criterion(preds, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                preds_all.append(preds.detach().cpu())
                y_all.append(y.detach().cpu())

            preds_all = torch.cat(preds_all)
            y_all = torch.cat(y_all)

            r2, corr, acc_strict = self._compute_metrics(preds_all, y_all, tol=0.1)
            _, _, acc_lil_loose = self._compute_metrics(preds_all, y_all, tol=0.25)
            _, _, acc_loose = self._compute_metrics(preds_all, y_all, tol=0.5)
            print(f"Epoch: {epoch} | R²: {r2:.3f} | Corr: {corr:.3f} | Acc±0.1: {acc_strict:.1f}% | Acc±0.25: {acc_lil_loose:.1f}% | Acc±0.5: {acc_loose:.1f}%")

    def evaluate(self):
        self.model.eval()
        total_loss, preds_all, y_all = 0, [], []

        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                preds = self.model(X)
                loss = self.criterion(preds, y)
                total_loss += loss.item()
                preds_all.append(preds.cpu())
                y_all.append(y.cpu())

        preds_all = torch.cat(preds_all)
        y_all = torch.cat(y_all)
        r2, corr, acc = self._compute_metrics(preds_all, y_all)
        _, _, acc_lil_loose = self._compute_metrics(preds_all, y_all, tol=0.25)
        _, _, acc_loose = self._compute_metrics(preds_all, y_all, tol=0.5)

        print(f"Test MSE: {total_loss/len(self.test_loader):.4f} | "
              f"R²: {r2:.3f} | Corr: {corr:.3f} | Acc(±0.1): {acc:.1f}% | Acc±0.25: {acc_lil_loose:.1f}% | Acc±0.5: {acc_loose:.1f}%")