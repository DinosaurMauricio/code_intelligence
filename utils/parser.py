import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Tree


class CodeParser:
    PY_LANGUAGE = Language(tspython.language())

    def __init__(self):
        self.parser = Parser(self.PY_LANGUAGE)

    def parse(self, code: bytes) -> Tree:
        return self.parser.parse(code, encoding="utf8")
