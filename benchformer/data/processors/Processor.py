#  Copyright (c) 2022 The Serafima.ai Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from dotmap import DotMap
from torch.utils.data import random_split
import pandas as pd
from . import FeaturesProcessorBuilder


class DataProcessor(object):
    """
    Abstract class for data processors is responsible for data loading, preprocessing and splitting on train/dev/test
    sets depending on the configs.

    Args:
        configs (DotMap): Contains parameters required for the initialization and usage of the data processor.

    Attributes:
        configs (DotMap): Stores data processor configuration.
        FeaturesProcessor (FeaturesProcessor): Converts dataset examples to the input features.
    """

    def __init__(self, configs: DotMap):

        self.configs = configs

        self.FeaturesProcessor = FeaturesProcessorBuilder.build(self.configs)

    def get_train_examples(self, df):
        raise NotImplementedError

    def get_dev_examples(self, df):
        raise NotImplementedError

    def get_test_examples(self, df):
        raise NotImplementedError

    def prepare_dataset(self, tokenizer):
        raise NotImplementedError

    def get_dataset(self, path: str = '', data_type: str = 'csv'):
        if path == '':
            path = self.configs.get('train_dataset', path)

        if data_type == 'csv':
            return pd.read_csv(path, index_col=0)

        if data_type == 'json':
            return path

        return None

    def split_dataset(self, train_dataset, train_set: int = 0.87, val_set: int = 0.13, test_set: int = 0.0) -> list:
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

        return random_split(train_dataset, [nb_train_samples, nb_val_samples, nb_test_samples])


class FeaturesProcessor(object):
    """
    Abstract class for features processors is responsible for converting dataset examples into features.

    Args:
        configs (DotMap): Contains parameters required for the initialization and usage of the features processor.

    Attributes:
        configs (DotMap): Stores features processor configuration.
    """

    def __init__(self, configs: DotMap):
        self.configs = configs

    def convert_examples_to_features(self, examples, tokenizer):
        raise NotImplementedError
