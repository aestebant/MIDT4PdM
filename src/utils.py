from pathlib import Path
import pandas as pd
import numpy as np


def load_CMAPSS(key, split) -> (pd.MultiIndex, list): # type: ignore
    path = Path('../datasets/turbofan_case_study')
    file = path / f'{key}_{split}.csv'
    raw = pd.read_csv(file)

    multi_idx = pd.MultiIndex.from_arrays(raw[['unit', 'cycle']].to_numpy().transpose()-1, names=['', 'timepoints'])
    X = raw[[f's{i}' for i in range(1, 22)]]
    X.index = multi_idx

    y = raw[['unit', 'HS']].to_numpy()
    _, idxs = np.unique(y[:, 0], return_index=True)
    y = y[idxs, -1]

    return X, list(y)


def load_BearingCWRU(key, split) -> (pd.MultiIndex, list): # type: ignore
    path = Path('../datasets/ballbearing_case_study')
    file = path / f'{key}_{split}.csv'
    raw = pd.read_csv(file)

    multi_idx = pd.MultiIndex.from_arrays(raw[['unit', 'cycle']].to_numpy().transpose().astype(int)-1, names=['', 'timepoints'])
    X = raw[['DE', 'FE', 'BA', 'RPM']]
    X.index = multi_idx

    y = raw[['unit', 'HS']].to_numpy()
    _, idxs = np.unique(y[:, 0], return_index=True)
    y = y[idxs, -1]

    return X, list(y)