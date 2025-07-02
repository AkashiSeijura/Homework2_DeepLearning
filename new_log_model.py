import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from utils import make_classification_data, ClassificationDataset

class LogisticRegression(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)

if __name__ == '__main__':
    # Генерируем данные
    X, y = make_classification_data(n=200, num_classes=3, n_features=2)

    # Убеждаемся что y  это одномерный тензор меток классов
    y = y.argmax(dim=1) if y.ndim > 1 else y

    # Создаём датасет и даталоадер
    dataset = ClassificationDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')

    model = LogisticRegression(in_features=2, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1)

    # Обучаем модель
    epochs = 100
    for epoch in range(1, epochs + 1):
        all_preds = []
        all_targets = []
        total_loss = 0

        for i, (batch_X, batch_y) in enumerate(dataloader):

            batch_y = batch_y.argmax(dim=1) if batch_y.ndim > 1 else batch_y

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).tolist())
            all_targets.extend(batch_y.tolist())

        avg_loss = total_loss / (i + 1)
        precision = precision_score(all_targets, all_preds, average='macro')
        recall = recall_score(all_targets, all_preds, average='macro')
        f1 = f1_score(all_targets, all_preds, average='macro')
        roc = roc_auc_score(
            torch.nn.functional.one_hot(torch.tensor(all_targets), num_classes=3),
            torch.nn.functional.one_hot(torch.tensor(all_preds), num_classes=3),
            average='macro', multi_class='ovr')
        cm = confusion_matrix(all_targets, all_preds)

        print(f"Epoch {epoch:>2}: loss={avg_loss:.4f}, precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}, ROC-AUC={roc:.3f}")
        print("Confusion Matrix:")
        print(cm)

    torch.save(model.state_dict(), 'logreg_multiclass.pth')
