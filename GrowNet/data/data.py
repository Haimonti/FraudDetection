import numpy as np
from torch.utils.data import Dataset


class AAERComp(Dataset):
    def __init__(self, data_df, target_col, pos=1, neg=0, out_pos=1, out_neg=-1):
        self.data_df = data_df
        self.target_col = target_col

        self.feat = data_df.drop(columns=[target_col]).values.astype(np.float32)
        self.label = data_df[target_col].values.astype(np.float32)

        idx_pos = self.label == pos
        idx_neg = self.label == neg
        self.label[idx_pos] = out_pos
        self.label[idx_neg] = out_neg

    def __getitem__(self, index):
        arr = self.feat[index, :]
        return arr, self.label[index]

    def __len__(self):
        return len(self.label)

