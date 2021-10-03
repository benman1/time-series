"""Utility functions for data loading."""
import requests
import pyreadr
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from fastcache import lru_cache


@lru_cache(maxsize=1, typed=False)
def get_energy_demand(scale: bool = True):
    resp = requests.get(
        "https://github.com/camroach87/gefcom2017data/raw/master/data/gefcom.rda",
        allow_redirects=True,
    )
    open("gefcom.rda", "wb").write(resp.content)
    result = pyreadr.read_r("gefcom.rda")
    df = result["gefcom"].pivot(index="ts", columns="zone", values="demand")
    if not scale:
        return df
    return pd.DataFrame(data=StandardScaler().fit_transform(df), columns=df.columns)


@lru_cache(maxsize=1, typed=False)
def get_ford(train: bool = True):
    """Classification dataset."""
    root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
    filename = root_url + "FordA_TRAIN.tsv"
    if not train:
        filename = root_url + "FordA_TEST.tsv"
    data = pd.read_csv(filename, sep="\t")
    y = data.values[:, 0].astype(int)
    x = data.values[:, 1:]
    y[y == -1] = 0
    return np.expand_dims(x, -1), y
