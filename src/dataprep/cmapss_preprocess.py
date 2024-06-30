from ctypes import Union
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import linear_model


def read_original_data(key_data, source) -> Union[pd.DataFrame, pd.DataFrame]:
    labels = ['unit', 'cycle', 'Altitud', 'TRA', 'Mach'] + [f's{i}' for i in range(1, 25)]

    df_train = pd.read_csv(source / f"train_{key_data}.txt", sep=' ', header=None, names=labels)
    df_train.dropna(axis=1, how='all', inplace=True)

    df_test = pd.read_csv(source / f"test_{key_data}.txt", sep=' ', header=None, names=labels)
    df_test.dropna(axis=1, how='all', inplace=True)
    rul = pd.read_csv(source / f"RUL_{key_data}.txt", header=None, names=['max_cycle'])
    rul.index = rul.index + 1
    max_cycle = df_test.groupby(by='unit')['cycle'].max()
    rul['max_cycle'] += max_cycle
    df_test = df_test.merge(rul, left_on='unit', right_index=True)

    return df_train, df_test


def save_processed_data(key_data, df_train: pd.DataFrame, df_test: pd.DataFrame, destination):
    removable = ['Altitud', 'TRA', 'Mach', 'RUL', 'HI', 'HI_smooth', 'max_cycle']
    df_train.drop(removable, axis=1, errors='ignore', inplace=True)
    df_test.drop(removable, axis=1, errors='ignore', inplace=True)
    df_train.to_csv(destination / f"{key_data}_train.csv", index=False)
    df_test.to_csv(destination / f"{key_data}_test.csv", index=False)


def regression_modeling(df_train) -> linear_model.LinearRegression:
    max_cycle = df_train.groupby(by='unit')['cycle'].max()
    aux = df_train.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit', right_index=True)
    aux['HI'] = np.nan
    aux.loc[aux.cycle <= aux.max_cycle * 0.1, 'HI'] = 1
    aux.loc[aux.cycle >= aux.max_cycle * 0.9, 'HI'] = 0
    aux = aux.dropna()

    inputs = ['Altitud', 'TRA', 'Mach'] + [f's{i+1}' for i in range(21)]
    x = aux[~aux['HI'].isnull()][inputs].values
    y = aux[~aux['HI'].isnull()][['HI']].values
    model = linear_model.LinearRegression()
    model.fit(x, y)

    return model


def _moving_average(df, unit):
    aux = df.query(f"unit=={unit}")
    roll = aux['HI'].rolling(window=5)
    return roll.mean()


def get_health_state(df, model) -> pd.DataFrame:
    inputs = ['Altitud', 'TRA', 'Mach'] + [f's{i+1}' for i in range(21)]
    df['HI'] = model.predict(df[inputs].values)
    roll_means = [_moving_average(df, i+1) for i in range(df.unit.max())]
    df = df.assign(HI_smooth=list(itertools.chain.from_iterable(roll_means)))
    df[['HI_smooth']] = df[['HI_smooth']].fillna(method='bfill')

    df[['HS']] = 1
    df.loc[df.HI_smooth >= 0.8, 'HS'] = 0
    df.loc[df.HI_smooth < 0.5, 'HS'] = 2

    return df


def split_in_classes(df) -> pd.DataFrame:
    units = np.array(df.unit.unique())
    np.random.seed(0)
    np.random.shuffle(units)
    states = {
        0: pd.DataFrame(columns=df.columns),
        1: pd.DataFrame(columns=df.columns),
        2: pd.DataFrame(columns=df.columns)
    }

    idx = 0
    # State 2
    while len(states[2].unit.unique()) < len(units) // 3 and idx < len(units):
        u = units[idx]
        unit_data = df.query(f"unit=={u}")
        if len(unit_data.HS.unique()) == 1:
            states[unit_data.HS.iloc[0]] = states[unit_data.HS.iloc[0]].append(unit_data)
        elif 2 in unit_data.HS.unique():
            states[2] = states[2].append(unit_data)
        else:
            cuts = unit_data.loc[unit_data['HS'].shift() != unit_data['HS']]
            cuts = cuts.append(unit_data.iloc[-1])
            length = (cuts['cycle'] - cuts['cycle'].shift())
            extend_cuts = pd.DataFrame(zip(length, cuts['HS'].shift()), index=cuts.index, columns=['length', 'HS'])
            mid_state_cut_point = extend_cuts.loc[:extend_cuts[extend_cuts.HS == 2].first_valid_index()].index[-2]
            mid_state = unit_data.loc[:mid_state_cut_point].iloc[:-1]
            if len(mid_state) > 20 and 1 in mid_state.HS.unique():
                states[1] = states[1].append(mid_state)
            elif extend_cuts.loc[extend_cuts.HS == 0]['length'].max() > 20:
                end = extend_cuts.loc[extend_cuts.HS == 0]['length'].idxmax()
                start = cuts.loc[:end].index[-2]
                states[0] = states[0].append(df.iloc[start:end])
            else:
                states[unit_data.HS.max()] = states[unit_data.HS.max()].append(unit_data)
        idx += 1
    # State 0
    while len(states[0].unit.unique()) < len(units) // 3 and idx < len(units):
        u = units[idx]
        unit_data = df.query(f"unit=={u}")
        cuts = unit_data.loc[unit_data['HS'].shift() != unit_data['HS']]
        cuts = cuts.append(unit_data.iloc[-1])
        if len(cuts) == 1 or len(unit_data.HS.unique()) == 1:
            states[cuts.HS.iloc[0]] = states[cuts.HS.iloc[0]].append(unit_data)
        else:
            length = (cuts['cycle'] - cuts['cycle'].shift())
            extend_cuts = pd.DataFrame(zip(length, cuts['HS'].shift()), index=cuts.index, columns=['length', 'HS'])
            if extend_cuts.loc[extend_cuts.HS == 0]['length'].max() > 20:
                end = extend_cuts.loc[extend_cuts.HS == 0]['length'].idxmax()
                start = cuts.loc[:end].index[-2]
                states[0] = states[0].append(df.iloc[start:end])
            else:
                mid_state_cut_point = extend_cuts.loc[:extend_cuts[extend_cuts.HS == 2].first_valid_index()].index[-2]
                mid_state = unit_data.loc[:mid_state_cut_point].iloc[:-1]
                if len(mid_state) > 20 and 1 in mid_state.HS.unique():
                    states[1] = states[1].append(mid_state)
                else:
                    states[2] = states[2].append(unit_data)
        idx += 1
    # State 1
    for u in units[idx:]:
        unit_data = df.query(f"unit=={u}")
        cuts = unit_data.loc[unit_data['HS'].shift() != unit_data['HS']]
        cuts = cuts.append(unit_data.iloc[-1])
        if len(cuts) == 1 or len(unit_data.HS.unique()) == 1:
            states[cuts.HS.iloc[0]] = states[cuts.HS.iloc[0]].append(unit_data)
        else:
            length = (cuts['cycle'] - cuts['cycle'].shift())
            extend_cuts = pd.DataFrame(zip(length, cuts['HS'].shift()), index=cuts.index, columns=['length', 'HS'])
            mid_state_cut_point = extend_cuts.loc[:extend_cuts[extend_cuts.HS == 2].first_valid_index()].index[-2]
            mid_state = unit_data.loc[:mid_state_cut_point].iloc[:-1]
            if len(mid_state) > 20 and 1 in mid_state.HS.unique():
                states[1] = states[1].append(mid_state)
            else:
                states[unit_data.HS.max()] = states[unit_data.HS.max()].append(unit_data)

    # Label all instances with the bag class
    for s in [0, 1, 2]:
        states[s].HS = s

    # Check if there are inconsistencies
    for s in [0, 1, 2]:
        if states[s].HS.max() > s:
            print(f"ERROR: state {states[s].HS.max()} in class {s}")
        for u in states[s].unit.unique():
            if s not in states[s].loc[states[s].unit == u].HS.unique():
                print(f"ERROR: state {s} not in unit {u}")
        for u in states[s].unit.unique():
            if len(states[s].loc[states[s].unit == u]) < 20:
                print(f"WARNING: unit {s} of length {len(states[s].loc[states[s].unit == u])} (class {s})")
    print(f"INFO: Class distribution in dataset: {[len(states[s].unit.unique()) for s in [0, 1, 2]]}, (labels: 0, 1, 2)")

    result = pd.concat(states.values())
    result.sort_values(by=['unit', 'cycle'], inplace=True)
    return result


if __name__ == "__main__":
    orig = Path('<path to the original data>')
    dest = Path('datasets/CMAPSS')

    for key in ['FD001', 'FD002', 'FD003', 'FD004']:
        print(f"INFO: Processing dataset {key}")
        df_train, df_test = read_original_data(key, orig)
        regression = regression_modeling(df_train)
        df_train = get_health_state(df_train, regression)
        df_test = get_health_state(df_test, regression)
        df_train = split_in_classes(df_train)
        df_test = split_in_classes(df_test)
        save_processed_data(key, df_train, df_test, dest)
