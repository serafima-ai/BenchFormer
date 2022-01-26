from torch.utils.data import random_split
import pandas as pd
from . import FeaturesProcessorBuilder


class DataProcessor(object):

    def __init__(self, configs):

        self.configs = configs

        self.FeaturesProcessor = FeaturesProcessorBuilder.build(self.configs.features)

    def get_train_examples(self, df):
        raise NotImplementedError

    def get_dev_examples(self, df):
        raise NotImplementedError

    def get_test_examples(self, df):
        raise NotImplementedError

    def prepare_dataset(self, tokenizer):
        raise NotImplementedError

    def get_dataset(self, path='', data_type='csv'):
        if path is '':
            path = self.configs.get('train_dataset', path)

        if data_type == 'csv':
            return pd.read_csv(path, index_col=0)

        if data_type == 'json':
            return path

        return None

    def split_dataset(self, train_dataset, train_set=0.87, val_set=0.13, test_set=0.0, default_dataset=list):
        train_set_split_prop = float(self.configs.get('train_set_proportion', train_set))
        val_set_split_prop = float(self.configs.get('val_set_proportion', val_set))
        test_set_split_prop = float(self.configs.get('test_set_proportion', test_set))

        if train_set_split_prop + val_set_split_prop > 1 and test_set_split_prop == 0:
            val_set_split_prop = 1 - train_set_split_prop
        if train_set_split_prop + val_set_split_prop + test_set_split_prop != 1:
            raise ValueError("Train, validation, test set proportions are not equal to 1")

        nb_train_samples = int(train_set_split_prop * len(train_dataset))
        nb_val_samples = int(val_set_split_prop * len(train_dataset))  # len(train_dataset) - nb_train_samples
        nb_test_samples = len(train_dataset) - nb_train_samples - nb_val_samples

        if nb_test_samples == 0:
            return random_split(train_dataset, [nb_train_samples, nb_val_samples]), default_dataset()

        return random_split(train_dataset, [nb_train_samples, nb_val_samples, nb_test_samples])


class FeaturesProcessor(object):

    def __init__(self, configs):
        self.configs = configs

    def convert_examples_to_features(self, examples, tokenizer):
        raise NotImplementedError
