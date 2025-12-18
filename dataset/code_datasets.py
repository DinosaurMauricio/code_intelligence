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

        self.tokenizer_mask = tokenizer.mask_token

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        code_string = self.samples.iloc[idx].func_code_string

        # parse the code string
        code_tree = self.parser.parse(bytes(code_string, "utf-8"))

        code_identifiers = self._get_identifiers(code_tree.root_node)
        num_tokens_to_mask = (len(code_identifiers) * self.mask_percentage) // 100

        sampled_identifiers = self.sample_identifiers(
            code_identifiers, num_tokens_to_mask
        )

        masked_string = self.mask_identifiers(code_string, sampled_identifiers)

        sample = {"labels": code_string, "masked_input": masked_string}

        return sample

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

                if children_node.type == "comment":
                    # Ignore comments, maybe the dataset could be preprocessed and remove comments...
                    continue

                tree_traversal(children_node)

        tree_traversal(root_node)

        return identifiers

    @staticmethod
    def sample_identifiers(identifiers, num_masks):
        sampled_identifiers = random.sample(identifiers, num_masks)
        # need to sort the values as to modify the string from left to right
        sampled_identifiers = sorted(sampled_identifiers, key=lambda x: x[0])
        return sampled_identifiers

    def mask_identifiers(self, code_string, masks, single_token=False):
        # single_token flag is just in to predict the first token
        # else it will multiply the mask token
        offset = 0
        labels = []
        for mask in masks:
            start, end = mask[0], mask[1]

            label = code_string[start + offset : end + offset]

            # labels.append(label)

            label_tokens = len(
                self.tokenizer("demand", add_special_tokens=False)["input_ids"]
            )

            # mask the code string
            code_string = (
                code_string[: start + offset]
                + self.tokenizer_mask
                * label_tokens  # TODO: Worth the shot just taking the first token of the label
                + code_string[end + offset :]
            )

            # because we modify the string, we need an offest for the modifications
            # the len will be 6 because the mask is "<mask>"" in codebert minus the difference
            difference = end - start
            offset += len(self.tokenizer_mask) * label_tokens - difference
        return code_string  # , labels
