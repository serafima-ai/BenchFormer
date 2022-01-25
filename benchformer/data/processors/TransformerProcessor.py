import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import InputExample

from benchformer.data.processors import DataProcessor, FeaturesProcessor


class TransformerDataProcessor(DataProcessor):

    def __init__(self, configs):
        super().__init__(configs)

        self.configs = configs

        self.FeaturesProcessor = FeaturesProcessor(self.configs.features)

    def get_train_examples(self, df):
        return self.create_examples(df, "train")

    def get_dev_examples(self, df):
        return self.create_examples(df, "dev")

    def get_test_examples(self, df):
        return self.create_examples(df, "test")

    def create_examples(self, df, set_type):
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

    def prepare_dataset(self, tokenizer):
        train_dataloader, val_dataloader, test_dataloader = self.prepare_train_dataset(tokenizer)

        bert_test_dataloader = self.resolve_test_dataset(test_dataloader, tokenizer)

        return train_dataloader, val_dataloader, bert_test_dataloader

    def prepare_train_dataset(self, tokenizer):
        dataframe = self.get_dataset()
        dataset = self.get_train_examples(dataframe)

        features = self.FeaturesProcessor.convert_examples_to_features(dataset, tokenizer)
        train_dataset = self.create_tensor_dataset(features)
        train_dataset, val_dataset, test_dataset = self.split_dataset(train_dataset, default_dataset=TensorDataset)

        # train loader
        train_sampler = RandomSampler(train_dataset)
        bert_train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                           batch_size=self.configs.batch_size)

        # val loader
        val_sampler = RandomSampler(val_dataset)
        bert_val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=self.configs.batch_size)

        # test loader
        test_sampler = RandomSampler(test_dataset)
        bert_test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                          batch_size=self.configs.batch_size)

        return bert_train_dataloader, bert_val_dataloader, bert_test_dataloader

    def resolve_test_dataset(self, test_dataloader, tokenizer):
        if self.configs.test_dataset:
            test_dataframe = self.get_dataset(self.configs.test_dataset)
            dataset = self.get_test_examples(test_dataframe)
            features = self.FeaturesProcessor.convert_examples_to_features(dataset, tokenizer)
            test_dataset = self.create_tensor_dataset(features)

            test_sampler = RandomSampler(test_dataset)
            test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                         batch_size=self.configs.batch_size)

        return test_dataloader

    @staticmethod
    def create_tensor_dataset(features) -> TensorDataset:
        return TensorDataset(torch.tensor([f.input_ids for f in features], dtype=torch.long),
                             torch.tensor([f.attention_mask for f in features], dtype=torch.long),
                             torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
                             torch.tensor([f.label for f in features], dtype=torch.long))
