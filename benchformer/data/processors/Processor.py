import torch
from torch.utils.data import TensorDataset
import pandas as pd


class DataProcessor(object):

    def __init__(self, configs):

        self.configs = configs

        self.FeaturesProcessor = FeaturesProcessor(self.configs.features)

    def get_train_examples(self, df):
        raise NotImplementedError

    def get_dev_examples(self, df):
        raise NotImplementedError

    def get_test_examples(self, df):
        raise NotImplementedError

    def get_dataset(self, path='', data_type='csv'):
        if path is '':
            path = self.configs.get('train_dataset', path)

        if data_type == 'csv':
            return pd.read_csv(path, index_col=0)

        if data_type == 'json':
            return path

        return None

    def prepare_dataset(self, tokenizer):
        raise NotImplementedError

    def create_tensor_dataset(self, features) -> TensorDataset:
        return TensorDataset(torch.tensor([f.input_ids for f in features], dtype=torch.long),
                             torch.tensor([f.attention_mask for f in features], dtype=torch.long),
                             torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
                             torch.tensor([f.label for f in features], dtype=torch.long))


class FeaturesProcessor(object):

    def __init__(self, configs):
        self.configs = configs

    def convert_examples_to_features(self, data, tokenizer):
        raise NotImplementedError
