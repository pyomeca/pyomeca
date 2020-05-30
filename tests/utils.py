import inspect
import json
import re

import numpy as np
import xarray as xr

mkdocs_server = "http://127.0.0.1:8000"


class DocStringError(Exception):
    pass


def is_expected_array(
    array,
    shape_val: tuple,
    first_last_val: tuple,
    mean_val: float,
    median_val: float,
    sum_val: float,
    nans_val: int,
    decimal: int = 6,
):
    np.testing.assert_array_equal(
        x=array.shape, y=shape_val, err_msg="Shape does not match"
    )
    raveled = array.values.ravel()
    np.testing.assert_array_almost_equal(
        x=raveled[0],
        y=first_last_val[0],
        err_msg="First value does not match",
        decimal=decimal,
    )
    np.testing.assert_array_almost_equal(
        x=raveled[-1],
        y=first_last_val[-1],
        err_msg="Last value does not match",
        decimal=decimal,
    )
    np.testing.assert_array_almost_equal(
        x=array.mean(), y=mean_val, decimal=decimal, err_msg="Mean does not match"
    )
    np.testing.assert_array_almost_equal(
        x=array.median(skipna=True),
        y=median_val,
        decimal=decimal,
        err_msg="Median does not match",
    )
    np.testing.assert_allclose(
        actual=array.sum(), desired=sum_val, rtol=0.05, err_msg="Sum does not match"
    )
    np.testing.assert_array_equal(
        x=array.isnull().sum(), y=nans_val, err_msg="Nans value value does not match"
    )


def print_expected_values(array: xr.DataArray):
    shape_val = array.shape
    print(f"shape_val={shape_val}")

    ravel = array.values.ravel()
    first_last_val = ravel[0], ravel[-1]
    print(f"first_last_val={first_last_val}")

    mean_val = array.mean().item()
    print(f"mean_val={mean_val}")

    median_val = array.median(skipna=True).item()
    print(f"median_val={median_val}")

    sum_val = array.sum().item()
    print(f"sum_val={sum_val}")

    nans_val = array.isnull().sum().item()
    print(f"nans_val={nans_val}")


def is_function_or_method_or_new(method):
    return (inspect.isfunction(method) or inspect.ismethod(method)) and (
        method.__name__[0] != "_" or method.__name__ == "__new__"
    )


def get_available_methods(module):
    return [
        method_obj
        for class_name, class_obj in inspect.getmembers(module, inspect.isclass)
        for method_name, method_obj in inspect.getmembers(class_obj)
        if is_function_or_method_or_new(method_obj)
    ]


def do_we_generate_doc_files():
    try:
        import requests

        try:
            return requests.get(mkdocs_server).status_code == 200
        except requests.exceptions.ConnectionError:
            return False
    except ModuleNotFoundError:
        return False


def function_has_return(func):
    """Caution: this will return True if the function contains the word 'return'"""
    lines, _ = inspect.getsourcelines(func)
    return any("return" in line for line in lines)


def extract_code_blocks_from_md(
    docstring: str, start_code_block: str = "```python", end_code_block: str = "```"
) -> str:
    return inspect.cleandoc(
        "\n".join(
            re.findall(
                fr"{start_code_block}(.*?){end_code_block}", docstring, re.DOTALL
            )
        )
    )


def generate_api_json(module):
    api_json = {
        "name": "pyomeca",
        "docstring": "<p>Base module</p>",
        "link": "/",
        "children": [],
    }
    for class_name, class_obj in inspect.getmembers(module, inspect.isclass):
        class_dict = {
            "name": class_name,
            "link": f"/api/{class_obj.__module__.split('.')[-1]}/#{class_obj.__module__}.{class_name}",
            "children": [],
        }
        if class_name == "DataArrayAccessor":
            class_dict["docstring"] = f"<p>{class_obj.__doc__}</p>"
        for method_name, method_obj in inspect.getmembers(class_obj):
            if is_function_or_method_or_new(method_obj):
                method_dict = {
                    "name": method_name,
                    "link": f"{class_dict['link']}.{method_name}",
                    "value": 1,
                }
                method_docstring = get_method_generated_docstring(method_dict)
                if method_name == "__new__":
                    class_dict["docstring"] = method_docstring
                else:
                    method_dict["docstring"] = method_docstring
                    class_dict["children"].append(method_dict)

        api_json["children"].append(class_dict)

    with open("docs/api/api.json", "w") as api_file:
        json.dump(api_json, api_file, indent=2)


def get_method_generated_docstring(method_dict):
    import requests
    from bs4 import BeautifulSoup

    to_delete = '<p class="admonition-title">Example</p>'
    html = requests.get(f"{mkdocs_server}{method_dict['link']}")
    html.raise_for_status()
    soup = BeautifulSoup(html.text)
    doc_bloc = soup.find(
        "h3", {"id": method_dict["link"].split("#")[-1]}
    ).find_next_sibling()
    description = f"{doc_bloc.find('p')}"
    example = "".join(
        [f"{i}" for i in doc_bloc.find("div", {"class": "admonition example"}).children]
    )
    generated_docstring = description + example
    if generated_docstring:
        return generated_docstring.replace(to_delete, "").replace(
            "<img ", '<img class="center"'
        )
    raise ValueError(f"could not process {method_dict['name']}")
