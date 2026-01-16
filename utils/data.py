import pandas as pd

from torch.utils.data import DataLoader
from functools import partial

from dataset.code_datasets import CodeRefactoringDataset


def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


class DatasetBuilder:
    def __init__(self, parser, code_masker, remove_docstring):
        self.dataset_partial = partial(
            CodeRefactoringDataset,
            parser=parser,
            code_masker=code_masker,
            remove_docstring=remove_docstring,
        )

    def build(
        self,
        data: pd.DataFrame,
        splits: list[str] = ["train", "val", "test"],
    ) -> dict[str, CodeRefactoringDataset]:

        assert len(splits) != 0, "need to provide at least one split"

        data_dict = {}

        for split in splits:
            data_split = data[data.split_name == split]
            data_dict[split] = self.dataset_partial(data=data_split)

        return data_dict


class DataLoaderBuilder:
    def __init__(self, collate_fn, batch_size):
        self.collate_fn = collate_fn
        self.batch_size = batch_size

    def build(
        self, datasets: dict[str, CodeRefactoringDataset]
    ) -> dict[str, DataLoader]:
        dataloader_dict = {}

        for split, data in datasets.items():
            dataloader_dict[split] = DataLoader(
                data,
                shuffle=True if split == "train" else False,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
            )
        return dataloader_dict
