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

    # TODO: Clean this method
    def mask_identifiers(self, code_string, masks):
        labels = []
        offset = 0

        # tree-sitter works with bytes, when working with english code is fine
        # if we use the relative byte positon, but when working for example
        # with chinese characters the position might differ, encoding it
        # fixes this issue
        code_bytes = code_string.encode("utf8")
        mask_holder = "<mask_holder>".encode("utf-8")

        for mask in masks:  # masks are the identifiers
            start, end = mask[0], mask[1]

            if code_bytes[start + offset - 1] == 32:  # 32 is ' ' in bytes
                # This is because the tokenizer before a space it might tokenize it together for example kwargs without
                # space is kw, args. if  before there was a space it would make it Ġk, wa, rgs which is completly differnt
                # in this case we just attach the space if before there was a space
                start -= 1

            identifier_label = code_bytes[start + offset : end + offset]
            labels.append(identifier_label.decode("utf-8"))

            # set a temporary token to get the position, this is because if we mask and
            # then tokenize, its more difficult to find the positions. This way we can
            # add a place holder token on the identifiers and after tokenizing it
            # its possible to change the holder with the true tokenized identifiers
            code_bytes = (
                code_bytes[: start + offset] + mask_holder + code_bytes[end + offset :]
            )

            difference = end - start
            offset += len(mask_holder) - difference

        code_string = code_bytes.decode("utf-8")
        tokenized_code = self.tokenizer.tokenize(code_string)

        index = 0
        # TODO: Check here because it seems its eating up some tokens....
        # the labels should be already be ordered but it might be that reason why
        # or either the mask_holder is not being set
        while len(labels) > 0:
            token = tokenized_code[index]
            if token == "<mask_holder>":
                identifier = labels.pop(0)
                ## TODO: Worth the shot just taking the first token of the label when it tokenizes
                ## for example long variables and just masks part of it

                print(self.tokenizer.tokenize(identifier))
                print(identifier)
                tokenized_code = (
                    tokenized_code[:index]
                    + [self.tokenizer.mask_token]
                    * len(self.tokenizer.tokenize(identifier))
                    + tokenized_code[index + 1 :]
                )
            index += 1

        max = 256

        if len(tokenized_code) >= max - 2:
            tokenized_code = tokenized_code[
                : max - 2
            ]  #  truncate it. -2 because need to add bos and eos token

        tokenized_code = (
            [self.tokenizer.bos_token] + tokenized_code + [self.tokenizer.eos_token]
        )

        padding = [self.tokenizer.pad_token] * (max - len(tokenized_code))
        tokenized_code = tokenized_code + padding

        tokenized_code = self.tokenizer.convert_tokens_to_ids(tokenized_code)

        return tokenized_code
