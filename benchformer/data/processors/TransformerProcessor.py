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

import torch
from dotmap import DotMap
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import InputExample

from benchformer.data.processors import DataProcessor, register_processor


@register_processor('TransformerProcessor')
class TransformerDataProcessor(DataProcessor):
    """
    Data processing for Transformer architecture models.

    Args:
        configs (DotMap): Contains parameters required for the initialization and usage of the data processor.
    """

    def __init__(self, configs: DotMap):
        super().__init__(configs)

    def get_train_examples(self, df) -> list:
        return self.create_examples(df, "train")

    def get_dev_examples(self, df) -> list:
        return self.create_examples(df, "dev")

    def get_test_examples(self, df) -> list:
        return self.create_examples(df, "test")

    def create_examples(self, df, set_type: str) -> list:
        """Creates examples for the training and dev sets."""
        examples = []
        for idx, row in df.iterrows():
            guid = "%s-%s" % (set_type, idx)

            default_input_columns = ['query', 'text']
            default_output_columns = ['label']

            input_columns = self.configs.get('input', default_input_columns)
            output_columns = self.configs.get('output', default_output_columns)

            try:
                text_a, text_b = row[input_columns[0]], None
                if len(input_columns) > 1:
                    text_b = row[input_columns[1]]

                label = row[output_columns[0]]

            except KeyError:
                print('No corresponding columns found for config keys {}'.format(input_columns))

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def prepare_dataset(self, tokenizer) -> tuple:
        train_dataloader, val_dataloader, test_dataloader = self.prepare_train_dataset(tokenizer)

        val_dataloader = self.resolve_dataset(val_dataloader, tokenizer)

        test_dataloader = self.resolve_dataset(test_dataloader, tokenizer, 'test')

        return train_dataloader, val_dataloader, test_dataloader

    def prepare_train_dataset(self, tokenizer) -> tuple:
        dataframe = self.get_dataset()
        dataset = self.get_train_examples(dataframe)

        features = self.FeaturesProcessor.convert_examples_to_features(dataset, tokenizer)
        train_dataset = self.create_tensor_dataset(features)
        train_subset, val_subset, test_subset = self.split_dataset(train_dataset)

        train_dataloader, val_dataloader, test_dataloader = None, None, None

        # train loader
        if train_subset.indices:
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.configs.batch_size)

        # val loader
        if val_subset.indices:
            val_sampler = RandomSampler(val_subset)
            val_dataloader = DataLoader(val_subset, sampler=val_sampler, batch_size=self.configs.batch_size)

        # test loader
        if test_subset.indices:
            test_sampler = RandomSampler(test_subset)
            test_dataloader = DataLoader(test_subset, sampler=test_sampler, batch_size=self.configs.batch_size)

        return train_dataloader, val_dataloader, test_dataloader

    def resolve_dataset(self, dataloader: DataLoader, tokenizer, stage: str = 'val') -> DataLoader:
        dataset_path = self.configs.test_dataset if stage == 'test' else self.configs.val_dataset

        if dataset_path:
            dataframe = self.get_dataset(dataset_path)
            dataset = self.get_test_examples(dataframe) if stage == 'test' else self.get_dev_examples(dataframe)
            features = self.FeaturesProcessor.convert_examples_to_features(dataset, tokenizer)
            dataset = self.create_tensor_dataset(features)

            test_sampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=test_sampler,
                                    batch_size=self.configs.batch_size)

        return dataloader

    @staticmethod
    def create_tensor_dataset(features) -> TensorDataset:
        return TensorDataset(torch.tensor([f.input_ids for f in features], dtype=torch.long),
                             torch.tensor([f.attention_mask for f in features], dtype=torch.long),
                             torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
                             torch.tensor([f.label for f in features], dtype=torch.long))
