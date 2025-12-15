import random
import torch
import pandas as pd
from utils.parser import CodeParser


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

        # self.tokenizer_mask = tokenizer.mask_token
        self.tokenizer_mask = b"<mask>"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        code_string = self.samples.iloc[idx].func_code_string
        code_label = self.tokenizer(code_string)

        num_tokens_to_mask = (len(code_label) * self.mask_percentage) // 100

        code_tree = self.parser.parse(bytes(code_string, "utf-8"))
        code_identifiers = self._get_identifiers(code_tree)

        # use all identifiers in case there is less than number of tokens
        num_tokens_to_mask = (
            num_tokens_to_mask
            if len(code_identifiers) > num_tokens_to_mask
            else len(code_identifiers)
        )

        sampled_identifiers = self.sample_identifiers(
            code_identifiers, num_tokens_to_mask
        )

        masked_string = self.mask_identifiers(code_string, sampled_identifiers)
        masked_input = self.tokenizer(masked_string)

        return code_label, masked_input

    @staticmethod
    def _get_identifiers(root_node):
        identifiers = []

        def tree_traversal(node):
            for children_node in node.children:
                # identifiers hold variables and method names
                if children_node.type == "identifier":
                    # get the string position of each identifier
                    identifiers.append(
                        (children_node.start_byte, children_node.end_byte)
                    )
                tree_traversal(children_node)
            return identifiers

        tree_traversal(root_node)
        return identifiers

    @staticmethod
    def sample_identifiers(identifiers, num_masks):
        sampled_identifiers = random.sample(identifiers, num_masks)
        # need to sort the values as to modify the string from left to right
        sampled_identifiers = sorted(sampled_identifiers, key=lambda x: x[0])
        return sampled_identifiers

    def mask_identifiers(self, code_string, masks):
        offset = 0
        for mask in masks:
            start, end = mask[0], mask[1]

            # label = code_string[start + offset : end + offset]
            # mask the code string
            code_string = (
                code_string[: start + offset]
                + self.tokenizer_mask
                + code_string[end + offset :]
            )

            # because we modify the string, we need an offest for the modifications
            # the len will be 6 because the mask is "<mask>"" in codebert minus the difference
            offset += len(self.tokenizer_mask) - end - start
        return code_string
