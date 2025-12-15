import os
from functools import partial
from torchtext.data import get_tokenizer  # dummy tokenizer

from dataset.code_datasets import CodeRefactoringDataset
from utils.data import load_data


path = os.path.dirname(os.path.realpath(__file__))
DATASET_FILE_NAME = "dataset.parquet"

filepath = os.path.join(path, DATASET_FILE_NAME)

# path = os.path
data = load_data(filepath)


# can be bytes(string, encoding) or b'string'
code_bytes = bytes("""hello world""", "utf-8")

# dummy tokenizer
tokenizer = get_tokenizer("basic_english")

dataset_partial = partial(CodeRefactoringDataset, tokenizer=tokenizer)

splits = ["train", "valid", "test"]
data_dict = {}
for split in splits:
    data_split = data[data.split_name == split]
    data_dict[split] = dataset_partial(data=data_split)

# print(data.split_name.unique())
#
# print(data_dict.keys(), len(data_dict["valid"]))
