# # import torch
# # import torch.nn as nn
# # from torch.utils.data import DataLoader

# # class RegressorTrainer:
# #     def __init__(self, model, train_dataset, test_dataset, lr=1e-3, batch_size=32, device=None):
# #         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
# #         self.model = model.to(self.device)
# #         self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# #         self.test_loader  = DataLoader(test_dataset, batch_size=batch_size)
# #         self.criterion = nn.MSELoss()
# #         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

# #     def _compute_metrics(self, preds, targets, tol=0.1):
# #         """Compute R², correlation, and approximate accuracy."""
# #         preds = preds.squeeze()
# #         targets = targets.squeeze()
# #         ss_res = torch.sum((targets - preds) ** 2)
# #         ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
# #         r2 = 1 - ss_res / (ss_tot + 1e-8)

# #         if torch.std(preds) == 0 or torch.std(targets) == 0:
# #             corr = torch.tensor(0.0)
# #         else:
# #             corr = torch.corrcoef(torch.stack([targets, preds]))[0, 1]

# #         # Approximate accuracy: % predictions within `tol` range of target
# #         within_tol = torch.abs(preds - targets) <= tol
# #         acc = within_tol.float().mean() * 100

# #         return r2.item(), corr.item(), acc.item()

# #     def train(self, epochs=10):
# #         for epoch in range(epochs):
# #             self.model.train()
# #             total_loss, preds_all, y_all = 0, [], []

# #             for X, y in self.train_loader:
# #                 X, y = X.to(self.device), y.to(self.device)
# #                 self.optimizer.zero_grad()

# #                 preds = self.model(X)
# #                 loss = self.criterion(preds, y)
# #                 loss.backward()
# #                 self.optimizer.step()

# #                 total_loss += loss.item()
# #                 preds_all.append(preds.detach().cpu())
# #                 y_all.append(y.detach().cpu())

# #             preds_all = torch.cat(preds_all)
# #             y_all = torch.cat(y_all)

# #             r2, corr, acc_strict = self._compute_metrics(preds_all, y_all, tol=0.1)
# #             _, _, acc_lil_loose = self._compute_metrics(preds_all, y_all, tol=0.25)
# #             _, _, acc_loose = self._compute_metrics(preds_all, y_all, tol=0.5)
# #             print(f"Epoch: {epoch} | R²: {r2:.3f} | Corr: {corr:.3f} | Acc±0.1: {acc_strict:.1f}% | Acc±0.25: {acc_lil_loose:.1f}% | Acc±0.5: {acc_loose:.1f}%")

# #     def evaluate(self):
# #         self.model.eval()
# #         total_loss, preds_all, y_all = 0, [], []

# #         with torch.no_grad():
# #             for X, y in self.test_loader:
# #                 X, y = X.to(self.device), y.to(self.device)
# #                 preds = self.model(X)
# #                 loss = self.criterion(preds, y)
# #                 total_loss += loss.item()
# #                 preds_all.append(preds.cpu())
# #                 y_all.append(y.cpu())

# #         preds_all = torch.cat(preds_all)
# #         y_all = torch.cat(y_all)
# #         r2, corr, acc = self._compute_metrics(preds_all, y_all)
# #         _, _, acc_lil_loose = self._compute_metrics(preds_all, y_all, tol=0.25)
# #         _, _, acc_loose = self._compute_metrics(preds_all, y_all, tol=0.5)

# #         print(f"Test MSE: {total_loss/len(self.test_loader):.4f} | "
# #               f"R²: {r2:.3f} | Corr: {corr:.3f} | Acc(±0.1): {acc:.1f}% | Acc±0.25: {acc_lil_loose:.1f}% | Acc±0.5: {acc_loose:.1f}%")

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt


# class RegressorTrainer:
#     def __init__(self, model, train_dataset, test_dataset, lr=1e-3, batch_size=32, device=None):
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = model.to(self.device)
#         self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         self.test_loader  = DataLoader(test_dataset, batch_size=batch_size)
#         self.criterion = nn.MSELoss()
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

#         # --- history tracking for plots ---
#         self.history = {
#             "loss": [], "r2": [],
#             "acc_01": [], "acc_025": [], "acc_05": []
#         }

#     def _compute_metrics(self, preds, targets, tol=0.1):
#         """Compute R², correlation, and approximate accuracy."""
#         preds = preds.squeeze()
#         targets = targets.squeeze()
#         ss_res = torch.sum((targets - preds) ** 2)
#         ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
#         r2 = 1 - ss_res / (ss_tot + 1e-8)

#         if torch.std(preds) == 0 or torch.std(targets) == 0:
#             corr = torch.tensor(0.0)
#         else:
#             corr = torch.corrcoef(torch.stack([targets, preds]))[0, 1]

#         # Approximate accuracy: % predictions within `tol` range of target
#         within_tol = torch.abs(preds - targets) <= tol
#         acc = within_tol.float().mean() * 100

#         return r2.item(), corr.item(), acc.item()

#     def train(self, epochs=10):
#         for epoch in range(epochs):
#             self.model.train()
#             total_loss, preds_all, y_all = 0, [], []

#             for X, y in self.train_loader:
#                 X, y = X.to(self.device), y.to(self.device)
#                 self.optimizer.zero_grad()

#                 preds = self.model(X)
#                 loss = self.criterion(preds, y)
#                 loss.backward()
#                 self.optimizer.step()

#                 total_loss += loss.item()
#                 preds_all.append(preds.detach().cpu())
#                 y_all.append(y.detach().cpu())

#             preds_all = torch.cat(preds_all)
#             y_all = torch.cat(y_all)

#             r2, corr, acc_strict = self._compute_metrics(preds_all, y_all, tol=0.1)
#             _, _, acc_lil_loose = self._compute_metrics(preds_all, y_all, tol=0.25)
#             _, _, acc_loose = self._compute_metrics(preds_all, y_all, tol=0.5)

#             self.history["loss"].append(total_loss / len(self.train_loader))
#             self.history["r2"].append(r2)
#             self.history["acc_01"].append(acc_strict)
#             self.history["acc_025"].append(acc_lil_loose)
#             self.history["acc_05"].append(acc_loose)

#             print(f"Epoch: {epoch+1} | "
#                   f"R²: {r2:.3f} | Corr: {corr:.3f} | "
#                   f"Acc±0.1: {acc_strict:.1f}% | Acc±0.25: {acc_lil_loose:.1f}% | Acc±0.5: {acc_loose:.1f}%")

#         self._plot_training_curves()

#     def _plot_training_curves(self):
#         """Visualize metrics across training epochs."""
#         epochs = range(1, len(self.history["loss"]) + 1)
#         plt.figure(figsize=(15, 5))

#         # --- Loss ---
#         plt.subplot(1, 3, 1)
#         plt.plot(epochs, self.history["loss"], label="Train Loss")
#         plt.title("Training Loss Over Epochs")
#         plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.legend()

#         # --- R² ---
#         plt.subplot(1, 3, 2)
#         plt.plot(epochs, self.history["r2"], label="Train R²", color="green")
#         plt.title("R² Over Epochs")
#         plt.xlabel("Epoch"); plt.ylabel("R²"); plt.legend()

#         # --- Accuracy thresholds ---
#         plt.subplot(1, 3, 3)
#         plt.plot(epochs, self.history["acc_01"], label="Acc ±0.1")
#         plt.plot(epochs, self.history["acc_025"], label="Acc ±0.25")
#         plt.plot(epochs, self.history["acc_05"], label="Acc ±0.5")
#         plt.title("Accuracy Across Epochs")
#         plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.legend()

#         plt.tight_layout()
#         plt.show()
#         plt.savefig("training_curves.png")

#     def evaluate(self):
#         self.model.eval()
#         total_loss, preds_all, y_all = 0, [], []

#         with torch.no_grad():
#             for X, y in self.test_loader:
#                 X, y = X.to(self.device), y.to(self.device)
#                 preds = self.model(X)
#                 loss = self.criterion(preds, y)
#                 total_loss += loss.item()
#                 preds_all.append(preds.cpu())
#                 y_all.append(y.cpu())

#         preds_all = torch.cat(preds_all)
#         y_all = torch.cat(y_all)
#         r2, corr, acc = self._compute_metrics(preds_all, y_all)
#         _, _, acc_lil_loose = self._compute_metrics(preds_all, y_all, tol=0.25)
#         _, _, acc_loose = self._compute_metrics(preds_all, y_all, tol=0.5)

#         print(f"Test MSE: {total_loss/len(self.test_loader):.4f} | "
#               f"R²: {r2:.3f} | Corr: {corr:.3f} | "
#               f"Acc(±0.1): {acc:.1f}% | Acc±0.25: {acc_lil_loose:.1f}% | Acc±0.5: {acc_loose:.1f}%")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

class RegressorTrainer:
    def __init__(self, model, dataset, test_dataset, lr=1e-3, batch_size=32,
                 val_split=0.2, weight_decay=1e-4, patience=10, model_name="best_model", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.model_name = f"{model_name.replace(' ', '_')}.pt"

        # --- Split training set into train/validation ---
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader   = DataLoader(self.val_dataset, batch_size=batch_size)
        self.test_loader  = DataLoader(test_dataset, batch_size=batch_size)

        # --- Early stopping ---
        self.best_val_r2 = -np.inf
        self.epochs_no_improve = 0
        self.patience = patience

        # --- Track metrics for plotting ---
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_r2": [], "val_r2": [],
            "val_acc_01": [], "val_acc_025": [], "val_acc_05": []
        }

    def _compute_metrics(self, preds, targets, tol=0.25):
        preds, targets = preds.squeeze(), targets.squeeze()
        ss_res = torch.sum((targets - preds) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        corr = torch.corrcoef(torch.stack([targets, preds]))[0, 1] if torch.std(preds) > 0 and torch.std(targets) > 0 else torch.tensor(0.0)
        acc = (torch.abs(preds - targets) <= tol).float().mean() * 100
        return r2.item(), corr.item(), acc.item()

    def _run_epoch(self, loader, training=True):
        """Run one training or validation epoch."""
        self.model.train() if training else self.model.eval()
        total_loss, preds_all, y_all = 0, [], []

        with torch.set_grad_enabled(training):
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                preds = self.model(X)
                loss = self.criterion(preds, y)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                preds_all.append(preds.detach().cpu())
                y_all.append(y.detach().cpu())

        preds_all = torch.cat(preds_all)
        y_all = torch.cat(y_all)

        # compute metrics for all tolerances
        r2, corr, acc01 = self._compute_metrics(preds_all, y_all, tol=0.1)
        _, _, acc025 = self._compute_metrics(preds_all, y_all, tol=0.25)
        _, _, acc05 = self._compute_metrics(preds_all, y_all, tol=0.5)

        return total_loss / len(loader), r2, corr, acc01, acc025, acc05

    def train(self, epochs=100):
        print(f"🚀 Starting training with {len(self.train_dataset)} samples "
              f"(validation: {len(self.val_dataset)})")
        for epoch in range(1, epochs + 1):
            train_loss, train_r2, _, _, _, _ = self._run_epoch(self.train_loader, training=True)
            val_loss, val_r2, val_corr, acc01, acc025, acc05 = self._run_epoch(self.val_loader, training=False)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_r2"].append(train_r2)
            self.history["val_r2"].append(val_r2)
            self.history["val_acc_01"].append(acc01)
            self.history["val_acc_025"].append(acc025)
            self.history["val_acc_05"].append(acc05)

            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val R²: {val_r2:.3f} | Corr: {val_corr:.3f} | "
                  f"Acc±0.1: {acc01:.1f}% | Acc±0.25: {acc025:.1f}% | Acc±0.5: {acc05:.1f}%")

            # --- Early stopping ---
            if val_r2 > self.best_val_r2:
                self.best_val_r2 = val_r2
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.model_name)
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print("🛑 Early stopping — validation R² plateaued.")
                    break

        print(f"✅ Training complete. Best Val R² = {self.best_val_r2:.3f}")
        self._plot_training_curves()

    # def evaluate(self):
    #     """Evaluate final test performance."""
    #     self.model.load_state_dict(torch.load("best_model_3.pt"))
    #     test_loss, r2, corr, acc01, acc025, acc05 = self._run_epoch(self.test_loader, training=False)
    #     print(f"📊 Test MSE: {test_loss:.4f} | R²: {r2:.3f} | Corr: {corr:.3f} | "
    #           f"Acc±0.1: {acc01:.1f}% | Acc±0.25: {acc025:.1f}% | Acc±0.5: {acc05:.1f}%")
    #     return {"test_r2": r2, "test_corr": corr, "acc01": acc01, "acc025": acc025, "acc05": acc05}

    def evaluate(self):
        """Evaluate final test performance."""
        checkpoint_path = self.model_name
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # handle both full checkpoints and plain model weights
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print("✅ Loaded model weights from checkpoint.")
        else:
            self.model.load_state_dict(checkpoint)
            print("✅ Loaded raw state_dict (weights only).")

        test_loss, r2, corr, acc01, acc025, acc05 = self._run_epoch(self.test_loader, training=False)
        print(f"📊 Test MSE: {test_loss:.4f} | R²: {r2:.3f} | Corr: {corr:.3f} | "
              f"Acc±0.1: {acc01:.1f}% | Acc±0.25: {acc025:.1f}% | Acc±0.5: {acc05:.1f}%")
        return {"test_r2": r2, "test_corr": corr, "acc01": acc01, "acc025": acc025, "acc05": acc05}
    
    def _plot_training_curves(self):
        epochs = range(1, len(self.history["train_loss"]) + 1)
        plt.figure(figsize=(14, 6))

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history["train_loss"], label="Train Loss")
        plt.plot(epochs, self.history["val_loss"], label="Val Loss")
        plt.title("Loss Over Epochs"); plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend()

        # R² + Accuracies
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history["val_r2"], label="Val R²", color="green")
        plt.plot(epochs, self.history["val_acc_01"], label="Acc ±0.1")
        plt.plot(epochs, self.history["val_acc_025"], label="Acc ±0.25")
        plt.plot(epochs, self.history["val_acc_05"], label="Acc ±0.5")
        plt.title("Validation Accuracy and R²"); plt.xlabel("Epoch"); plt.legend()

        plt.tight_layout()
        plt.savefig("training_validation_curves.png")
        plt.show()
        
    
    def resume_training(self, epochs=50):
        """
        Resume training from a saved model checkpoint.
        Keeps optimizer, learning rate, and history intact if available.
        """
        print(f"🔁 Resuming training from {checkpoint_path} ...")
        checkpoint_path = self.model_name

        # Load model weights
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        # If the checkpoint is just the model's state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" not in checkpoint:
            self.model.load_state_dict(checkpoint)
            print("✅ Loaded model weights only.")
        else:
            # Otherwise, it's a full training checkpoint
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.best_val_r2 = checkpoint.get("best_val_r2", -np.inf)
            self.history = checkpoint.get("history", self.history)
            print(f"✅ Loaded full checkpoint (previous best R²: {self.best_val_r2:.3f})")

        # Continue training
        start_r2 = self.best_val_r2
        for epoch in range(1, epochs + 1):
            train_loss, train_r2, _, _, _, _ = self._run_epoch(self.train_loader, training=True)
            val_loss, val_r2, val_corr, acc01, acc025, acc05 = self._run_epoch(self.val_loader, training=False)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_r2"].append(train_r2)
            self.history["val_r2"].append(val_r2)
            self.history["val_acc_01"].append(acc01)
            self.history["val_acc_025"].append(acc025)
            self.history["val_acc_05"].append(acc05)

            print(f"[Resume] Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val R²: {val_r2:.3f} | Corr: {val_corr:.3f} | "
                  f"Acc±0.1: {acc01:.1f}% | Acc±0.25: {acc025:.1f}% | Acc±0.5: {acc05:.1f}%")

            # Early stopping logic remains
            if val_r2 > self.best_val_r2:
                self.best_val_r2 = val_r2
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_val_r2": self.best_val_r2,
                    "history": self.history
                }, self.model_name)
                print("💾 Checkpoint updated.")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print("🛑 Early stopping — validation R² plateaued.")
                    break

        print(f"✅ Resumed training complete. Best Val R² improved from {start_r2:.3f} → {self.best_val_r2:.3f}")
        self._plot_training_curves()