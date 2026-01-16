import torch
import pandas as pd

from utils.parser import CodeParser


class CodeRefactoringDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        parser: CodeParser,
        code_masker,
        mask_percentage: int = 15,
        remove_docstring: bool = False,
    ):
        self.samples = data
        self.parser = parser
        self.mask_percentage = mask_percentage
        self.code_masker = code_masker
        self.remove_docstring = remove_docstring

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        code_string = self.samples.iloc[idx].func_code_string

        if self.remove_docstring:
            code_string = self.code_masker.remove_docstrings_regex(code_string)

        # parse the code string, needs to be bytes
        code_tree = self.parser.parse(bytes(code_string, "utf-8"))

        code_identifiers = self.code_masker.extract_identifiers(code_tree.root_node)

        num_tokens_to_mask = (len(code_identifiers) * self.mask_percentage) // 100

        sampled_identifiers = self.code_masker.sample_identifiers(
            code_identifiers, num_tokens_to_mask
        )

        input, labels = self.code_masker.apply_masking(
            code_string,
            sampled_identifiers,
        )

        sample = {
            "label": labels,
            "input": input,
        }

        return sample
