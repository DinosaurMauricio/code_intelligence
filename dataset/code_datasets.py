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
        # add to tokenizer vocabulary <mask_holder>
        self.tokenizer.add_tokens(["<mask_holder>"])

        self.parser = parser
        self.mask_percentage = mask_percentage

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

        sample = {
            "label": self.samples.iloc[idx].func_code_string,
            "masked_input": masked_string,
        }

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

    def mask_identifiers(self, code_string, masks):
        labels = []
        offset = 0

        for mask in masks:  # masks are the identifiers
            start, end = mask[0], mask[1]

            if code_string[start + offset - 1] == " ":
                # This is because the tokenizer before a space it might tokenize it together for example kwargs without
                # space is kw, args. if  before there was a space it would make it Ġk, wa, rgs which is completly differnt
                # in this case we just attach the space if before there was a space
                start -= 1

            identifier_label = code_string[start + offset : end + offset]
            labels.append(identifier_label)

            # set a temporary token to get the position, this is because if we mask and
            # then tokenize, its more difficult to find the positions. This way we can
            # add a place holder token on the identifiers and after tokenizing it
            # its possible to change the holder with the true tokenized identifiers
            code_string = (
                code_string[: start + offset]
                + "<mask_holder>"
                + code_string[end + offset :]
            )

            difference = end - start
            offset += len("<mask_holder>") - difference

        tokenized_code = self.tokenizer.tokenize(code_string)

        # replace the mask place holder with the actual mask
        index = 0
        while len(labels) > 0:
            token = tokenized_code[index]
            if token == "<mask_holder>":
                identifier = labels.pop(0)
                ## TODO: Worth the shot just taking the first token of the label when it tokenizes
                ## for example long variables and just masks part of it
                tokenized_code = (
                    tokenized_code[:index]
                    + [self.tokenizer.mask_token]
                    * len(self.tokenizer.tokenize(identifier))
                    + tokenized_code[index + 1 :]
                )
            index += 1

        return tokenized_code
