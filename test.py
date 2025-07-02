import urllib.request
import torch
from torch.utils.data import DataLoader
from homework2.data.homework_datasets import CSVDataset
from homework_model_modification import LinearRegressionManual
from utils import mse, log_epoch
import matplotlib.pyplot as plt


url = "https://raw.githubusercontent.com/Ankit152/Fish-Market/main/Fish.csv"
file_path = "fish.csv"
urllib.request.urlretrieve(url, file_path)

dataset = CSVDataset(file_path, target_column="Weight", categorical_columns=["Species"])
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print(f"Размер датасета: {len(dataset)}")
print(f"Количество батчей: {len(dataloader)}")

in_features = dataset[0][0].shape[0]
model = LinearRegressionManual(in_features=in_features, l1_lambda=0.01, l2_lambda=0.01)

lr = 0.01
epochs = 1000
early_stopping_patience = 10
best_loss = float('inf')
patience_counter = 0

for epoch in range(1, epochs + 1):
    total_loss = 0
    for i, (batch_X, batch_y) in enumerate(dataloader):
        y_pred = model(batch_X)
        loss = mse(y_pred, batch_y)
        total_loss += loss

        model.zero_grad()
        model.backward(batch_X, batch_y, y_pred)
        model.step(lr)

    avg_loss = total_loss / (i + 1)
    log_epoch(epoch, avg_loss)

    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping on epoch {epoch}")
            break


model.save("linear_regression_fish.pth")