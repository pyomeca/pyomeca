import csv
from typing import Optional


def col_spliter(x, p, s):
    if p and s:
        return x.split(p)[-1].split(s)[0]
    if p:
        return x.split(p)[-1]
    if s:
        return x.split(s)[0]
    return x


def find_end_header_in_opensim_file(filename: str, end_header: Optional[int] = None):
    with open(filename, "rt") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if row[0] == "endheader":
                end_header = idx
                break
    if end_header is None:
        raise IndexError(
            "endheader not detected in your file. Try to specify the `end_header` parameter"
        )
    return end_header
