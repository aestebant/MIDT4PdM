from ctypes import Union
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


def original2sequences(experiments: list, source: Path, sample_size: int, test_percent=0.5) -> Union[pd.DataFrame, pd.DataFrame]:
    cols = ['cycle', 'DE', 'FE', 'BA', 'RPM', 'HS']
    train_data = list()
    test_data = list()
    for file in experiments:
        mat = loadmat(source / file)
        numid = int(file.split('_')[-1].split('.')[0])
        baseline = len(file.split('_')) == 3
        if file.split('_')[0] == '028':
            mat_cols = [f'X{numid:03}_DE_time']
        else:
            mat_cols = [f'X{numid:03}_DE_time', f'X{numid:03}_FE_time']
            if not baseline:
                mat_cols += [f'X{numid:03}_BA_time']
        sequence = np.column_stack([mat[col] for col in mat_cols])
        samples = np.array_split(sequence, len(sequence)//sample_size)
        if baseline:
            health = 0
        else:
            switcher = {'007': 1, '014': 2, '021': 3, '028': 4}
            health = switcher[(file.split('_')[0])]
        for i, sample in enumerate(samples):
            destination = 'train' if random.randrange(0, 100) > test_percent * 100 else 'test'
            length = len(sample)
            sample = np.column_stack((np.arange(1, length+1), sample))  # cycles
            if baseline:
                sample = np.column_stack((sample, np.zeros(length)))  # BA
            if file.split('_')[0] == '028':
                sample = np.column_stack((sample, np.zeros((length, 2))))  # FE, BA
            if f'X{numid:03}RPM' not in mat.keys():
                switcher = {'0': 1797, '1': 1772, '2': 1750, '3': 1730}
                if baseline:
                    mat[f'X{numid:03}RPM'] = switcher[file.split('_')[0]]
                else:
                    mat[f'X{numid:03}RPM'] = switcher[file.split('_')[1]]

            sample = pd.DataFrame(np.column_stack((sample, np.full(length, mat[f'X{numid:03}RPM']), np.full(length, health))), columns=cols)
            if destination == 'train':
                train_data.append(sample)
            else:
                test_data.append(sample)

            if baseline and i > 100:
                break

    random.shuffle(train_data)
    random.shuffle(test_data)
    for i in range(len(train_data)):
        train_data[i].insert(loc=0, column='unit', value=i+1)
    for i in range(len(test_data)):
        test_data[i].insert(loc=0, column='unit', value=i+1)

    return pd.concat(train_data), pd.concat(test_data)


if __name__ == "__main__":
    orig = Path('<path to the original data>')
    dest = Path('datasets/BearingCWRU')

    # DE-BB datasets
    experiments = [
        '0_normal_97.mat', '007_0_innerrace_105.mat', '014_0_innerrace_169.mat', '021_0_innerrace_209.mat', '028_0_innerrace_56.mat',
        '1_normal_98.mat', '007_1_innerrace_106.mat', '014_1_innerrace_170.mat', '021_1_innerrace_210.mat', '028_1_innerrace_57.mat',
        '2_normal_99.mat', '007_2_innerrace_107.mat', '014_2_innerrace_171.mat', '021_2_innerrace_211.mat', '028_2_innerrace_58.mat',
        '3_normal_100.mat', '007_3_innerrace_108.mat', '014_3_innerrace_172.mat', '021_3_innerrace_212.mat', '028_3_innerrace_59.mat'
    ]
    df_train, df_test = original2sequences(experiments, orig / 'drive', 1200)
    df_train.to_csv(dest / 'drive-innerrace_train.csv', index=False)
    df_test.to_csv(dest / 'drive-innerrace_test.csv', index=False)

    # DE-IR datasets
    experiments = [
        '0_normal_97.mat', '007_0_ball_118.mat', '014_0_ball_185.mat', '021_0_ball_222.mat', '028_0_ball_48.mat',
        '1_normal_98.mat', '007_1_ball_119.mat', '014_1_ball_186.mat', '021_1_ball_223.mat', '028_1_ball_49.mat',
        '2_normal_99.mat', '007_2_ball_120.mat', '014_2_ball_187.mat', '021_2_ball_224.mat', '028_2_ball_50.mat',
        '3_normal_100.mat', '007_3_ball_121.mat', '014_3_ball_188.mat', '021_3_ball_225.mat', '028_3_ball_51.mat'
    ]
    df_train, df_test = original2sequences(experiments, orig / 'drive', 1200)
    df_train.to_csv(dest / 'drive-ball_train.csv', index=False)
    df_test.to_csv(dest / 'drive-ball_test.csv', index=False)

    # FE-BB datasets
    experiments = [
        '0_normal_97.mat', '007_0_innerrace_278.mat', '014_0_innerrace_274.mat', '021_0_innerrace_270.mat',
        '1_normal_98.mat', '007_1_innerrace_279.mat', '014_1_innerrace_275.mat', '021_1_innerrace_271.mat',
        '2_normal_99.mat', '007_2_innerrace_280.mat', '014_2_innerrace_276.mat', '021_2_innerrace_272.mat',
        '3_normal_100.mat', '007_3_innerrace_281.mat', '014_3_innerrace_277.mat', '021_3_innerrace_273.mat'
    ]
    df_train, df_test = original2sequences(experiments, orig / 'fan', 1200)
    df_train.to_csv(dest / 'fan-innerrace_train.csv', index=False)
    df_test.to_csv(dest / 'fan-innerrace_test.csv', index=False)

    # FE-IR datasets
    experiments = [
        '0_normal_97.mat', '007_0_ball_282.mat', '014_0_ball_286.mat', '021_0_ball_290.mat',
        '1_normal_98.mat', '007_1_ball_283.mat', '014_1_ball_287.mat', '021_1_ball_291.mat',
        '2_normal_99.mat', '007_2_ball_284.mat', '014_2_ball_288.mat', '021_2_ball_292.mat',
        '3_normal_100.mat', '007_3_ball_285.mat', '014_3_ball_289.mat', '021_3_ball_293.mat'
    ]
    df_train, df_test = original2sequences(experiments, orig / 'fan', 1200)
    df_train.to_csv(dest / 'fan-ball_train.csv', index=False)
    df_test.to_csv(dest / 'fan-ball_test.csv', index=False)
