from torch.utils.data import Dataset, Subset, DataLoader
import pandas as pd
from pytorch_lightning import LightningDataModule
import numpy as np
from typing import Iterable


class TradeDataSet(Dataset):
    main_dir = r'C:/Users/yonio/PycharmProjects/trade/'

    def __init__(self, look_back: int, threshold:float):
        data = pd.read_csv(self.main_dir + r'data/FFAData.csv')
        aux_data = pd.read_csv(self.main_dir + r'data/FFADataAux.csv')
        self.data, self.full_data = self.normalize_df(data, aux_data)

        self.look_back = look_back
        self.threshold = threshold

    @staticmethod
    def normalize_df(data, aux_data):
        merged = pd.merge(data, aux_data)
        merged = merged.drop('Date', axis=1)
        data = data.drop('Date', axis=1)

        res = [df.diff().dropna() for df in (data, merged)]
        return res

    def __len__(self): return len(self.data) - self.look_back

    def __getitem__(self, item):
        x = self.full_data.iloc[item:item + self.look_back].to_numpy(dtype='float32')
        y = self.data.iloc[item + self.look_back].to_numpy(dtype='float32')

        y = np.array([self.calc_ground_truth(y) for y in y], dtype='int64')
        return {
            'y': y,
            'x': x
        }

    def calc_ground_truth(self, y):
        if y > self.threshold:
            return 2
        elif y < -self.threshold:
            return 0
        else:
            return 1


class TradeDataModule(LightningDataModule):
    def __init__(self, look_back, threshold, train_ratio, batch_size):
        super().__init__()

        self.ds = TradeDataSet(look_back, threshold)
        self.train_ds, self.test_ds = self.train_test_split(train_ratio)
        self.batch_size = batch_size

    def train_test_split(self, train_ratio):
        train_samples_n = int(len(self.ds) * train_ratio)
        return (
            Subset(self.ds, list(range(0, train_samples_n))),
            Subset(self.ds, list(range(train_samples_n, len(self.ds))))
        )

    def train_dataloader(self): return DataLoader(self.train_ds, self.batch_size, shuffle=True)
    def val_dataloader(self): return DataLoader(self.test_ds, drop_last=True)
