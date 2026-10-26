"""Monthly returns from the Kenneth R. French Data Library, downloaded on first use.

The library is free for research use but not redistributed here: files are cached in data/.
All returns are converted from percent to decimals; missing values (-99.99, -999) become NaN.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data"
BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "industries": "49_Industry_Portfolios_CSV.zip",
    "ff3": "F-F_Research_Data_Factors_CSV.zip",
    "ff5": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "mom": "F-F_Momentum_Factor_CSV.zip",
}
_ROW = re.compile(r"^\s*(\d{6})\s*,")


def _text(name: str) -> str:
    path = CACHE / FILES[name]
    if not path.exists():
        resp = requests.get(BASE + FILES[name], timeout=120)
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
    with zipfile.ZipFile(path) as zf:
        return zf.read(zf.namelist()[0]).decode("latin-1")


def _monthly_table(text: str, title: str | None = None) -> pd.DataFrame:
    """The first block of YYYYMM rows after `title` (or after the first header row)."""
    lines = text.splitlines()
    start = 0
    if title is not None:
        start = next(i for i, l in enumerate(lines) if title in l)
    header_i = next(i for i in range(start, len(lines)) if lines[i].lstrip().startswith(","))
    cols = [c.strip() for c in lines[header_i].split(",")[1:]]
    rows = []
    for line in lines[header_i + 1:]:
        if not _ROW.match(line):
            break
        parts = line.split(",")
        rows.append([parts[0].strip()] + [float(x) for x in parts[1:len(cols) + 1]])
    df = pd.DataFrame(rows, columns=["month"] + cols)
    df.index = pd.PeriodIndex(pd.to_datetime(df.pop("month"), format="%Y%m"), freq="M")
    return df.replace([-99.99, -999.0], np.nan) / 100


def industries() -> pd.DataFrame:
    """Value-weighted monthly returns of the 49 industry portfolios (July 1926 onwards)."""
    return _monthly_table(_text("industries"), "Average Value Weighted Returns -- Monthly")


def factors() -> pd.DataFrame:
    """Monthly Mkt-RF, SMB, HML, RF (1926+), RMW, CMA (1963+) and UMD momentum (1927+)."""
    ff3 = _monthly_table(_text("ff3"))
    ff5 = _monthly_table(_text("ff5"))[["RMW", "CMA"]]
    mom = _monthly_table(_text("mom")).rename(columns={"Mom": "UMD"})
    return ff3.join(ff5, how="left").join(mom, how="left")
