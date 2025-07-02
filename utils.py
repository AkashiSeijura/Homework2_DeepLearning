import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.datasets import make_classification # импортируем из библиотеки
#make_classification — это генератор из sklearn, создающий искусственные данные для многоклассовой классификации "Для себя"

class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_regression_data(n=100, noise=0.1, source='random'):
    if source == 'random':
        X = torch.rand(n, 1)
        w, b = 2.0, -1.0
        y = w * X + b + noise * torch.randn(n, 1)
        return X, y
    elif source == 'diabetes':
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X = torch.tensor(data['data'], dtype=torch.float32)
        y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
        return X, y
    else:
        raise ValueError('Unknown source')

# Реализован класс make_classification_data для многоклассовой классификации
def make_classification_data(n=200, num_classes=2, n_features=2):
    if num_classes <= 1:
        raise ValueError("num_classes должен быть >= 2 для классификации")

    X, y = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_classes=num_classes,
        n_clusters_per_class=1,
        random_state=42
    )

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return X, y

def mse(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean().item()

def accuracy(y_pred, y_true):
    y_pred_bin = (y_pred > 0.5).float()
    return (y_pred_bin == y_true).float().mean().item()

def log_epoch(epoch, loss, **metrics):
    msg = f"Epoch {epoch}: loss={loss:.4f}"
    for k, v in metrics.items():
        msg += f", {k}={v:.4f}"
    print(msg)