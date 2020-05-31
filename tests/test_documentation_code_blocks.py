from pathlib import Path

import black
import pytest

from tests.utils import extract_code_blocks_from_md

docs_path = Path("./docs")
doc_files = [f"{file}" for file in docs_path.glob("*.md")]

doc_files_string = []
for file in doc_files:
    with open(f"{file}", "r", encoding="utf8") as f:
        doc_files_string.append(f.read().replace("../tests/data", "tests/data"))


@pytest.mark.parametrize("doc_file_string", doc_files_string, ids=doc_files)
def test_code_blocks(doc_file_string):
    exec(extract_code_blocks_from_md(doc_file_string), {}, {})


@pytest.mark.parametrize("doc_file_string", doc_files_string, ids=doc_files)
def test_lint_code_blocks(doc_file_string):
    code_blocks = extract_code_blocks_from_md(doc_file_string)
    if code_blocks:
        code_blocks = f"{code_blocks}\n"
        assert code_blocks == black.format_str(code_blocks, mode=black.FileMode())
