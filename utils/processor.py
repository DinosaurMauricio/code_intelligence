import random

from utils.constants import MASK_PLACEHOLDER


class CodeMaskingProcessor:

    SPACE_BYTE = 32
    IGNORE_LABEL_ID = -100

    def __init__(self, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        # TODO: check how truncating affects as some codes are to big, they have +1000 words
        # maybe I can first truncate and then mask. Or I can just filter the dataset
        self.max_length = max_length

    @staticmethod
    def extract_identifiers(root_node):
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
        # need to sort the values to modify the string from left to right
        sampled_identifiers = sorted(sampled_identifiers, key=lambda x: x[0])
        return sampled_identifiers

    def apply_masking(self, code_string, masks_positions):

        original_identifiers, code_with_placeholders = self._insert_placeholders(
            masks_positions,
            code_string,
        )

        code_tokens = self.tokenizer.tokenize(code_with_placeholders)

        masked_tokens, label = self._replace_placeholders_with_masks(
            original_identifiers,
            code_tokens,
        )

        padded_tokens = self._add_special_tokens_and_padding(masked_tokens)

        attention = [
            1 if token != self.tokenizer.pad_token else 0 for token in padded_tokens
        ]

        input = {
            "token_ids": self.tokenizer.convert_tokens_to_ids(padded_tokens),
            "attention": attention,
        }

        return input, label

    def _insert_placeholders(self, masks_positions, code_string: str):

        # tree-sitter works with bytes, when working with english codes its fine
        # if we use the relative byte positon, but when working for example
        # with chinese characters the position might differ, encoding it
        # fixes this issue
        code_bytes = code_string.encode("utf8")
        mask_holder = MASK_PLACEHOLDER.encode("utf-8")

        offset = 0
        labels = []

        for start, end in masks_positions:  # positions are in bytes
            if code_bytes[start + offset - 1] == self.SPACE_BYTE:  # 32 is ' ' in bytes
                # This is because the tokenizer might tokenize the word with the space, for example:
                # kwargs without space is [kw, args] with a space in the previous position
                # [Ġk, wa, rgs] which is completly different. For this reason,
                # in these cases i attach the space into the identifier
                start -= 1

            identifier_label = code_bytes[start + offset : end + offset]
            labels.append(identifier_label.decode("utf-8"))

            # set a temporary token (i.e "<mask_holder>"). This is because if we mask and
            # then tokenize, its more difficult to find the positions of the words.
            # by placing placeholder token on the identifiers position its more simple to
            # find the positions of the masks and ground truth labels
            code_bytes = (
                code_bytes[: start + offset] + mask_holder + code_bytes[end + offset :]
            )

            # because we modify the length of the string we need an offset for the positions
            difference = end - start
            offset += len(mask_holder) - difference

        code_string = code_bytes.decode("utf-8")
        return labels, code_string

    def _replace_placeholders_with_masks(self, labels, tokenized_code):
        index = 0
        # problem again is the tokenizer, for example "._iterate" will tokenize "._", "iterate"
        # messing the positions if naming starting with "_" exists on the method, so
        # building the labels here
        label = [self.IGNORE_LABEL_ID]  # bos token but set it already to -100 to ignore
        while len(labels) > 0:
            token = tokenized_code[index]
            if token == MASK_PLACEHOLDER:
                identifier = labels.pop(0)
                identifier_tokens = self.tokenizer(
                    identifier, add_special_tokens=False
                )["input_ids"]
                label.extend(identifier_tokens)
                ## TODO: Worth the shot just taking the first token of the label when it tokenizes
                ## for example long variables and just masks part of it
                identifier_len = len(identifier_tokens)
                tokenized_code = (
                    tokenized_code[:index]
                    + [self.tokenizer.mask_token] * identifier_len
                    + tokenized_code[index + 1 :]
                )

                index += identifier_len - 1
            else:
                label.append(self.IGNORE_LABEL_ID)
            index += 1

        # pad it or truncate it
        if len(label) >= self.max_length:
            label = label[: self.max_length]
        else:
            label.extend([self.IGNORE_LABEL_ID] * (self.max_length - len(label)))

        return tokenized_code, label

    def _add_special_tokens_and_padding(self, tokenized_code):

        if len(tokenized_code) >= self.max_length - 2:
            #  truncate it. -2 because need to add bos and eos token
            tokenized_code = tokenized_code[: self.max_length - 2]

        tokenized_code = (
            [self.tokenizer.bos_token] + tokenized_code + [self.tokenizer.eos_token]
        )

        padding = [self.tokenizer.pad_token] * (self.max_length - len(tokenized_code))
        tokens = tokenized_code + padding
        return tokens
