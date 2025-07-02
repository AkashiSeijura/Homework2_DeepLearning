import csv
import torch
from torch.utils.data import Dataset


class CSVDataset(Dataset):
    def __init__(self, file_path, target_column, categorical_columns=None):
        self.data = []
        self.targets = []
        self.categorical_columns = categorical_columns or []

        # Чтение данных из CSV
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        self.header = header
        target_idx = header.index(target_column)

        raw_data = [row[:target_idx] + row[target_idx+1:] for row in rows]
        raw_target = [row[target_idx] for row in rows]


        cat_indices = [header.index(col) for col in self.categorical_columns if col in header]


        self.cat_maps = [{} for _ in cat_indices]
        cat_encoded = []

        for row in raw_data:
            encoded_row = []
            for i, val in enumerate(row):
                if i in cat_indices and val not in ("", None):
                    idx = cat_indices.index(i)
                    if val not in self.cat_maps[idx]:
                        self.cat_maps[idx][val] = len(self.cat_maps[idx])
                    one_hot = [0] * len(self.cat_maps[idx])
                    one_hot[self.cat_maps[idx][val]] = 1
                    encoded_row.extend(one_hot)
                else:
                    try:
                        encoded_row.append(float(val))
                    except ValueError:
                        encoded_row.append(0.0)
            cat_encoded.append(encoded_row)


        features_tensor = torch.tensor(cat_encoded, dtype=torch.float32)
        self.min = features_tensor.min(0).values
        self.max = features_tensor.max(0).values
        self.features = (features_tensor - self.min) / (self.max - self.min + 1e-6)

        try:
            self.targets = torch.tensor([float(y) for y in raw_target], dtype=torch.float32).unsqueeze(1)
        except ValueError:

            classes = list(set(raw_target))
            self.class_map = {label: i for i, label in enumerate(classes)}
            self.targets = torch.tensor([self.class_map[y] for y in raw_target], dtype=torch.long)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]

    def __len__(self):
        return len(self.features)