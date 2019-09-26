import os
from tqdm import tqdm
import numpy as np
from setup_fm import DPMovieLens100k
from setup_fm import DPMallzee


def create_mallzee():
    seed = 1
    num_splits = 5

    set_names = ['train_{}'.format(i) for i in range(num_splits)]
    set_names += ['test_{}'.format(i) for i in range(num_splits)]
    set_names += ['valid_{}'.format(i) for i in range(num_splits)]
    set_names += ['train_full_{}'.format(i) for i in range(num_splits)]
    set_name2filename = {name: 'mallzee/{}.txt'.format(name) for name in set_names}

    raw_data_filename = 'mallzee/mallzee.txt'
    num_lines = sum(1 for _ in open(raw_data_filename))
    with open(raw_data_filename, 'r') as f:
        header = next(f)
        lines = []

        with tqdm(total=num_lines) as pbar:
            for line in f:
                lines.append(line)
                pbar.update(1)

    np.random.seed(seed)
    np.random.shuffle(lines)  # in place.

    from sklearn.model_selection import KFold
    indices = np.array([i for i in range(len(lines))])
    kf = KFold(n_splits=num_splits)

    for i, (train_val_idxs, test_idxs) in enumerate(kf.split(indices)):
        kf_2 = KFold(n_splits=20, shuffle=False)  # hold out 5 percent for validation. note: that I shuffle!
        train_idxs, valid_idxs = next(kf_2.split(train_val_idxs))

        train_lines = [lines[i] for i in train_idxs]
        valid_lines = [lines[i] for i in valid_idxs]
        test_lines = [lines[i] for i in test_idxs]
        train_full_lines = train_lines + valid_lines

        with open(set_name2filename['train_{}'.format(i)], 'w+') as file:
            file.write(header)
            for line in train_lines:
                file.write(line)

        with open(set_name2filename['valid_{}'.format(i)], 'w+') as file:
            file.write(header)
            for line in valid_lines:
                file.write(line)

        with open(set_name2filename['train_full_{}'.format(i)], 'w+') as file:
            file.write(header)
            for line in train_full_lines:
                file.write(line)

        with open(set_name2filename['test_{}'.format(i)], 'w+') as file:
            file.write(header)
            for line in test_lines:
                file.write(line)

        print('finished creating train, valid and test files {}'.format(i))


def create_ml100k(has_header=False):
    seed = 1
    num_splits = 5

    set_names = ['train_{}'.format(i) for i in range(num_splits)]
    set_names += ['train_full_{}'.format(i) for i in range(num_splits)]
    set_names += ['test_{}'.format(i) for i in range(num_splits)]
    set_names += ['valid_{}'.format(i) for i in range(num_splits)]
    set_name2filename = {name: 'ml-100k/{}.txt'.format(name) for name in set_names}

    raw_data_filename = 'ml-100k/u.data'
    num_lines = sum(1 for _ in open(raw_data_filename))
    with open(raw_data_filename, 'r') as f:
        if has_header:
            header = next(f)

        lines = []

        with tqdm(total=num_lines) as pbar:
            for line in f:
                lines.append(line)
                pbar.update(1)

    np.random.seed(seed)
    np.random.shuffle(lines)  # in place.

    from sklearn.model_selection import KFold
    indices = np.array([i for i in range(len(lines))])
    kf = KFold(n_splits=num_splits, shuffle=False)  # too much overlap between folds if not shuffled i think.

    for i, (train_val_idxs, test_idxs) in enumerate(kf.split(indices)):
        kf_2 = KFold(n_splits=20)  # hold out 5 percent for validation.
        train_idxs, valid_idxs = next(kf_2.split(train_val_idxs))

        train_lines = [lines[i] for i in train_idxs]
        valid_lines = [lines[i] for i in valid_idxs]
        train_full_lines = train_lines + valid_lines
        test_lines = [lines[i] for i in test_idxs]

        with open(set_name2filename['train_{}'.format(i)], 'w+') as file:
            #file.write(header)
            for line in train_lines:
                file.write(line)

        with open(set_name2filename['train_full_{}'.format(i)], 'w+') as file:
            #file.write(header)
            for line in train_full_lines:
                file.write(line)

        with open(set_name2filename['valid_{}'.format(i)], 'w+') as file:
            #file.write(header)
            for line in valid_lines:
                file.write(line)

        with open(set_name2filename['test_{}'.format(i)], 'w+') as file:
            #file.write(header)
            for line in test_lines:
                file.write(line)

        print('finished creating train, train_full, valid and test files {}'.format(i))

def create_data_files():
    dp_classes = [DPMovieLens100k, DPMallzee]
    for dp_class in dp_classes:
        for i in range(1):
            train_dp = dp_class('train_{}'.format(i))
            test_dp = dp_class('test_{}'.format(i))
            valid_dp = dp_class('valid_{}'.format(i))



# create_mallzee()
# create_ml100k()

create_data_files()




