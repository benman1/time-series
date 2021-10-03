import requests
import pyreadr
import pandas as pd
from sklearn.preprocessing import StandardScaler
from fastcache import lru_cache


@lru_cache(maxsize=128, typed=False)
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
