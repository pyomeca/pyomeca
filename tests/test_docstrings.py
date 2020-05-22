import inspect

import black
import pytest

import pyomeca
from tests.utils import (
    extract_code_blocks_from_md,
    function_has_return,
    DocStringError,
    get_available_methods,
    do_we_generate_doc_files,
    generate_api_json,
)

generate_doc_files = do_we_generate_doc_files()
methods = get_available_methods(pyomeca)
if generate_doc_files:
    generate_api_json(pyomeca)


@pytest.mark.parametrize("method", methods)
def test_has_docstring(method):
    if not method.__doc__:
        raise DocStringError(f"Missing docstring in `{method}`")


@pytest.mark.parametrize("method", methods)
def test_docstring_has_example(method):
    if "```python" not in method.__doc__:
        raise DocStringError(f"Missing example in `{method}` docstring")


@pytest.mark.parametrize("method", methods)
def test_docstring_example(method):
    plt_show_replacer = (
        f'plt.savefig("docs/images/api/{method.__name__}.svg", bbox_inches="tight")\nplt.figure()'
        if generate_doc_files
        else ""
    )
    code_block = extract_code_blocks_from_md(method.__doc__).replace(
        "plt.show()", plt_show_replacer,
    )
    exec(code_block, {}, {})


@pytest.mark.parametrize("method", methods)
def test_docstring_lint_code_blocks(method):
    code_blocks = extract_code_blocks_from_md(method.__doc__)
    if code_blocks:
        code_blocks = f"{code_blocks}\n"
        assert code_blocks == black.format_str(code_blocks, mode=black.FileMode())


@pytest.mark.parametrize("method", methods)
def test_docstring_return(method):
    if function_has_return(method):
        if "Returns:" not in method.__doc__:
            raise DocStringError(f"Missing returns in `{method}` docstring")
        if "return" not in inspect.getfullargspec(method).annotations:
            raise DocStringError(
                f"Type annotation missing for the `return` type in {method} docstring"
            )


@pytest.mark.parametrize("method", methods)
def test_docstring_parameters(method):
    funct_with_ignored_args = "cls", "self"
    argspec = inspect.getfullargspec(method)
    args = [a for a in argspec.args if a not in funct_with_ignored_args]
    if args and "Arguments:" not in method.__doc__:
        raise DocStringError(f"`Arguments` block missing in `{method}` docstring")
    for arg in args:
        if arg in funct_with_ignored_args:
            continue
        if arg not in method.__doc__:
            raise DocStringError(f"{arg} not described in {method} docstring")
        if arg not in argspec.annotations:
            raise DocStringError(f"{arg} not type annotated in {method}")
