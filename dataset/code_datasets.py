import torch
import pandas as pd

from utils.parser import CodeParser
from utils.processor import CodeMaskingProcessor


class CodeRefactoringDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        parser: CodeParser,
        tokenizer,
        mask_percentage: int = 15,
    ):
        self.samples = data
        self.tokenizer = tokenizer
        self.parser = parser
        self.mask_percentage = mask_percentage
        # this could be injected but because only using python right now this is fine
        self.code_masker = CodeMaskingProcessor(tokenizer)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        code_string = self.samples.iloc[idx].func_code_string

        # parse the code string, needs to be bytes
        code_tree = self.parser.parse(bytes(code_string, "utf-8"))

        code_identifiers = self.code_masker.extract_identifiers(code_tree.root_node)

        num_tokens_to_mask = (len(code_identifiers) * self.mask_percentage) // 100

        sampled_identifiers = self.code_masker.sample_identifiers(
            code_identifiers, num_tokens_to_mask
        )

        masked_input = self.code_masker.apply_masking(
            code_string,
            sampled_identifiers,
        )

        sample = {
            "label": code_string,
            "masked_input": masked_input,
        }

        return sample
