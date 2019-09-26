import sys
import os
from tqdm import tqdm
import numpy as np
import pickle
import torch


def get_item_id2item(users_dict, make_tens=False):
    items, item_ids = get_items(users_dict)
    if make_tens:
        import torch
        item_id2item = {item_id: torch.from_numpy(item).float().reshape(1, -1) for item_id, item in zip(item_ids, items)}
    else:
        item_id2item = {item_id: item for item_id, item in zip(item_ids, items)}
    return item_id2item


def get_items(users_dict):
    """
    return: items, items_ids
    """
    items = []
    item_ids = []

    for user_id, user in users_dict.items():
        for i in range(len(user.items)):
            if user.item_ids[i] not in item_ids:
                items.append(user.items[i])
                item_ids.append(user.item_ids[i])

    return items, item_ids


class _UsersDictBuilderMallzee(object):
    """
    basically responsible for how items get stored. different models typically take in different types of input.
    """

    class _MallzeeUser(object):
        possible_num_ratings = 2

        def __init__(self, user_id):
            self.user_id = user_id
            self.items = []
            self.item_ids = []
            self.ratings = []
            self.support_ids_dict = {0: [], 1: []}  # lambda dict can't be pickled!

        def add_data(self, item_id, item, rating):
            self.items.append(item)
            self.item_ids.append(item_id)
            self.ratings.append(rating)
            self.support_ids_dict[rating].append(item_id)

    def __init__(self, which_set):
        self.which_set = which_set
        self.raw_data_filename = os.path.join('C:/advers/data/mallzee', '{}.txt'.format(self.which_set))

    def __call__(self):
        """
        returns {user_id: user}. items are stored as numpy arrays shape (1, d).
        :return:
        """
        sys.stderr.write('building users_dict from file {}\n'.format(self.raw_data_filename))
        import csv
        reader = csv.reader(open(self.raw_data_filename), delimiter=',')
        header = next(reader)
        num_lines = sum(1 for _ in open(self.raw_data_filename))
        users_dict = {}
        item_id = 0
        num_users = 0
        # colnames_of_features = ['userID', 'CurrentPrice', 'DiscountFromOriginal', 'Currency', 'TypeOfClothing', 'Gender', 'InStock', 'Brand', 'Colour']
        colnames_of_features = ['CurrentPrice', 'DiscountFromOriginal', 'Currency', 'TypeOfClothing', 'Gender', 'InStock', 'Brand', 'Colour']

        with tqdm(total=num_lines) as pbar:  # e.g. = ['3', '345', '3', '889237004']
            for row in reader:
                user_id = int(row[header.index('userID')]) - 1
                if user_id not in users_dict.keys():
                    users_dict[user_id] = self._MallzeeUser(user_id=user_id)  # user_ids must start from 0.
                    num_users += 1

                columns_of_features = [header.index(col_name) for col_name in colnames_of_features]
                raw_item = {col_name: float(row[col_idx]) for col_name, col_idx in zip(colnames_of_features, columns_of_features)}

                # raw_item = {header[i]: float(row[i]) for i in columns_of_features}
                item = self.process_raw_item(raw_item)
                rating = int(row[header.index('userResponse')])
                users_dict[user_id].add_data(item_id=item_id, item=item, rating=rating)
                item_id += 1
                pbar.update(1)

        return users_dict

    def process_raw_item(self, raw_item):
        item = []
        for column_name, value in raw_item.items():
            if column_name in ['userID', 'Currency', 'TypeOfClothing', 'Gender', 'InStock', 'Brand', 'Colour']:
                value_onehot_list = self._make_onehot(value, column_name)
                item += value_onehot_list
            else:
                item += [value]
        item = np.array(item).reshape(1, -1)
        return item

    @staticmethod
    def _make_onehot(value, column_name):
        num_classes = {
            'userID': 200,
            'Currency': 4,
            'TypeOfClothing': 22,
            'Gender': 4,
            'InStock': 2,
            'Brand': 132,
            'Colour': 16
        }

        value2idx = {i: i - 1 for i in range(1, num_classes[column_name] + 1)}
        onehot_list = [0] * num_classes[column_name]
        onehot_list[value2idx[value]] = 1

        return onehot_list


class _UsersDictBuilderMovieLens100k(object):
    ml_100k_raw_data_dir = 'C:/advers/data/ml-100k'
    num_items = 1682
    num_users = 943

    class _MovieLensUser(object):
        possible_num_ratings = 5

        def __init__(self, user_id, ratings_format):
            self.user_id = user_id
            self.items = []
            self.item_ids = []
            self.ratings = []

            if ratings_format == 'binary':
                self.support_ids_dict = {0: [], 1: []}
            else:
                self.support_ids_dict = {i: [] for i in range(5)}

        def add_data(self, item_id, item, rating):
            self.items.append(item)
            self.item_ids.append(item_id)
            self.ratings.append(rating)
            self.support_ids_dict[rating].append(item_id)

    def __init__(self, item_format, rating_format, onehot_encode, which_set):

        """
        options:
            include_user_id=False, categories_onehot=False)
            ratings_format: 'binary' or otherwise.
            item_format: either 'with_content' or 'just_id'.
            onehot_encode: whether to onehot encode categorical fields or not. - the only "categorical" field is the
            item_id - you can definitely not onehot encode it with regression tree.
            categories_onehot = False.
        """

        self.item_format = item_format  # either 'with_content' or 'just_id'.
        self.rating_format = rating_format
        self.onehot_encode = onehot_encode

        self.which_set = which_set
        self.raw_data_filename = os.path.join('C:/advers/data/ml-100k', '{}.txt'.format(self.which_set))

        if self.rating_format == 'binary':
            self.possible_num_ratings = 2
        else:
            self.possible_num_ratings = 5

    def __call__(self):
        """
        a row: user_id, item_id, rating, timestamp row e.g. = ['3', '345', '3', '889237004']
        data is stored as numpy array - the data providers converts to tensors.
        """
        import csv
        from collections import OrderedDict

        reader_user = csv.reader(open(self.raw_data_filename), delimiter='\t')
        reader_item = csv.reader(open('C:/advers/data/ml-100k/u.item', 'r', encoding='ISO-8859-1'), delimiter='|')
        num_lines = sum(1 for _ in open(self.raw_data_filename))
        users_dict = {}

        item_id2arr = {}  # {item_id: arr} arr: (1, d), d = num_features.
        feature2col_idx = OrderedDict({
            'unknown': 5,
            'Action': 6,
            'Adventure': 7,
            'Animation': 8,
            'Children': 9,
            'Comedy': 10,
            'Crime': 11,
            'Documentary': 12,
            'Drama': 13,
            'Fantasy': 14,
            'Film-Noir': 15,
            'Horror': 16,
            'Musical': 17,
            'Mystery': 18,
            'Romance': 19,
            'Sci-Fi': 20,
            'Thriller': 21,
            'War': 22,
            'Western': 23
        })

        for row in reader_item:
            features_list = []
            item_id = int(row[0])

            if self.item_format == 'with_content':
                item_id_onehot = self._item_id2one_hot(item_id)  # returns an array but I need a list!
                features_list += item_id_onehot

                for genre, col_idx in feature2col_idx.items():
                    feature = int(row[col_idx])
                    features_list.append(feature)
            else:
                if self.onehot_encode:
                    item_id_onehot = self._item_id2one_hot(item_id)  # returns an array but I need a list!
                    features_list += item_id_onehot
                else:
                    features_list = [item_id]

            item_arr = np.array(features_list).reshape(1, -1)
            item_id2arr[item_id] = item_arr

        with tqdm(total=num_lines) as pbar:  #
            for row in reader_user:
                user_id = int(row[0]) - 1  # must start from 0.
                item_id = int(row[1])
                rating = int(row[2])
                if self.rating_format == 'binary':
                    rating = 0 if rating < 4 else 1
                else:
                    rating = rating - 1  # i.e. 0, 1, 2, 3, 4

                if user_id not in users_dict.keys():
                    users_dict[user_id] = self._MovieLensUser(user_id, self.rating_format)

                item = item_id2arr[item_id]
                users_dict[user_id].add_data(item_id=item_id, item=item, rating=rating)
                pbar.update(1)

        return users_dict

    def _item_id2one_hot(self, item_id):
        item = [0] * self.num_items
        item_idx = item_id - 1
        item[item_idx] = 1
        return item


class _DataProvider(object):
    """
    vars:
        data_list: [(user_id, item_id, rating)].
        item_id2tens: {item_id: item}, where item is a tensor (1, d).
    """

    def __init__(self, num_users, data_filename_pickle, users_dict_builder):
        self.data_filename_pickle = data_filename_pickle
        self.users_dict_builder = users_dict_builder
        self.num_users = num_users
        self.item_num_features = None

    def _build(self):
        if os.path.isfile(self.data_filename_pickle):
            output = self.load_from_storage()
            data_list = output['data_list']
            item_id2tens = output['item_id2tens']
        else:
            users_dict = self.users_dict_builder()  # call.
            item_id2tens = get_item_id2item(users_dict, make_tens=True)

            data_list = []
            for user_id, user in users_dict.items():
                for i in range(len(user.item_ids)):
                    observation = (user_id, user.item_ids[i], user.ratings[i])
                    data_list.append(observation)

            output_to_save = {'data_list': data_list, 'item_id2tens': item_id2tens}
            self.save_to_storage(output_to_save)

        for item_id in item_id2tens.keys():
            item_tens = item_id2tens[item_id]
            self.item_num_features = item_tens.shape[1]
            print('item num features: {}'.format(self.item_num_features))
            break

        self.data_list = data_list
        self.item_id2tens = item_id2tens

    def save_to_storage(self, input_dict):
        with open(self.data_filename_pickle, 'wb') as file:
            pickle.dump(input_dict, file)
            print('finished creating data. saved to {}'.format(self.data_filename_pickle))

    def load_from_storage(self):
        if os.path.isfile(self.data_filename_pickle):
            with open(self.data_filename_pickle, 'rb') as file:
                output = pickle.load(file)  # output contains data, and users = {user}.
                print('loaded data from {}'.format(self.data_filename_pickle))
                return output


class DPMallzee(_DataProvider):
    def __init__(self, which_set):
        num_users = 200
        data_filename_pickle = os.path.join('C:/advers/data/mallzee', 'fm_{}'.format(which_set))
        users_dict_builder = _UsersDictBuilderMallzee(which_set)
        super(DPMallzee, self).__init__(num_users, data_filename_pickle, users_dict_builder)
        self._build()


class DPMovieLens100k(_DataProvider):
    def __init__(self, which_set):
        num_users = _UsersDictBuilderMovieLens100k.num_users
        data_filename_pickle = os.path.join('C:/advers/data/ml-100k', 'fm_{}'.format(which_set))
        users_dict_builder = _UsersDictBuilderMovieLens100k(item_format='with_content',
                                                            rating_format='full',
                                                            onehot_encode=True,  # this one-hot encodes item_id.
                                                            which_set=which_set)

        super(DPMovieLens100k, self).__init__(num_users, data_filename_pickle, users_dict_builder)
        self._build()


class DefaultSampler(object):
    def __init__(self, data_list, batch_size=1, shuffle_first=True):
        """
        :param data_list: list of tuples; each tuple corresponds to an observation e.g. (xi,yi). an observation is defined as
        the information needed to perform a single training iteration.
        """

        self.data_list = data_list
        self.data_copy = self.data_list
        self.batch_size = batch_size
        self.shuffle_first = shuffle_first
        self.reset()

    def __iter__(self):  # implement iterator interface.
        return self

    def __next__(self):
        return self.sample()

    def __len__(self):
        # return max_num_batches.
        actual = len(self.data_copy) / self.batch_size
        rounded = len(self.data_copy) // self.batch_size

        if actual - rounded == 0:
            return rounded

        return rounded + 1

    def sample(self):
        if self.has_next():
            # note: if 0 < len(x) < batch_size then x[batch_size:] returns [] (is len 0)
            batch = self.data_list[:self.batch_size] # return top
            self.data_list = self.data_list[self.batch_size:] # remove top

            return batch
        else:
            raise StopIteration()

    def has_next(self):
        if len(self.data_list) > 0:
            return True
        self.reset()  # fill up again for next time you want to iterate over it.
        return False

    def reset(self):
        self.data_list = self.data_copy
        if self.shuffle_first:
            self.shuffle()

    def shuffle(self):
        perm = np.random.permutation(len(self.data_list))
        self.data_list = [self.data_list[i] for i in perm]


class Sampler(DefaultSampler):
    def __init__(self, data_provider, batch_size):

        """
        vars:
            data_list: [(user_id, item_id, rating)].
            item_id2tens: {item_id: item}, where item is a tensor (1, d).
        data provider used is from this class.
        """

        self.data_list = data_provider.data_list
        self.item_id2tens = data_provider.item_id2tens
        self.num_users = data_provider.num_users
        self.user_onehot_matrix = torch.eye(self.num_users, self.num_users)

        super(Sampler, self).__init__(self.data_list, batch_size, shuffle_first=True)

    def sample(self):
        """
        vars:
            batch: data provider contains [(user_id, item_id, rating)].
        return:
            batch_new: (x, y).
                x (tensor): (batch_size, x_dim).
                y (tensor): (batch_size, 1). (targets i.e. ratings).
        """

        if self.has_next():
            # note: if 0 < len(x) < batch_size then x[batch_size:] returns [] (is len 0)
            batch = self.data_list[:self.batch_size] # return top
            self.data_list = self.data_list[self.batch_size:] # remove top

            items = []
            ratings = []
            u_list = []
            for user_id, item_id, rating in batch:
                items.append(self.item_id2tens[item_id])
                rating_tens = torch.tensor(rating).reshape(1, 1)
                u_list.append(self.user_onehot_matrix[user_id, :].reshape(1, -1))
                ratings.append(rating_tens)

            x = torch.cat(items, dim=0)
            y = torch.cat(ratings, dim=0)
            u = torch.cat(u_list, dim=0)
            batch_new = (x, u, y)

            return batch_new
        else:
            raise StopIteration()



