import torch
import pandas as pd


class CodeRefactoringDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer):
        self.samples = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        code_string = self.samples.iloc[idx].func_code_string
        tokenized_code = self.tokenizer(code_string)

        return tokenized_code
