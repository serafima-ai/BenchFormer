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

import codecs
import json
from pathlib import Path
from dotmap import DotMap
from transformers import InputFeatures, InputExample

from benchformer.data.processors import TransformerDataProcessor, FeaturesProcessor, register_processor


@register_processor('GluePARusProcessor')
class PARusDataProcessor(TransformerDataProcessor):
    """
    Data processing for the PARus task (Choice of Plausible Alternatives for Russian language) from RussianSuperGlue.
    For more info: https://russiansuperglue.com/tasks/task_info/PARus

    Args:
        configs (DotMap): Contains parameters required for the initialization and usage of the data processor.
    """

    def __init__(self, configs: DotMap):
        super().__init__(configs)

    def get_dataset(self, path: str = '', data_type: str = 'csv') -> list:
        if not path:
            print('Dataset path not specified! Training path from config file would be used instead.')
            path = self.configs.get('train_dataset', path)

        if not Path(path).is_file():
            raise FileNotFoundError("Dataset file was not found!")

        with codecs.open(path, encoding='utf-8-sig') as reader:
            lines = reader.read().split("\n")
            lines = list(map(json.loads, filter(None, lines)))

        return lines

    @staticmethod
    def build_features(row) -> tuple:
        premise = str(row["premise"]).strip()
        choice1 = row["choice1"]
        # choice2 = row["choice2"]
        label = row.get("label")
        label = label if label is not None else 0

        question = "Что было причиной этого?" if row["question"] == "cause" else "Что случилось в результате?"
        # text_a = f"{premise} {question} {choice1}"
        # text_b = f"{choice2}"
        text_a = f"{premise} {question}"
        text_b = f"{choice1}"
        # res = f"{premise} {question} {choice1} {choice2}"
        return (text_a, text_b), label

    def create_examples(self, df, set_type: str) -> list:
        res = list(map(self.build_features, df))
        texts = list(map(lambda x: x[0], res))
        labels = list(map(lambda x: x[1], res))
        ids = [x["idx"] for x in df]

        examples = []

        for num, text in enumerate(texts):
            idx = "%s-%s" % (set_type, ids[num])
            examples.append(InputExample(guid=idx, text_a=text[0], text_b=text[1], label=labels[num]))

        return examples


@register_processor('GluePARusFeaturesProcessor')
class PARusFeaturesProcessor(FeaturesProcessor):
    """
    Features processing for the PARus task (Choice of Plausible Alternatives for Russian language) from
    RussianSuperGlue. For more info: https://russiansuperglue.com/tasks/task_info/PARus

    Args:
        configs (DotMap): Contains parameters required for the initialization and usage of the features processor.
    """

    def __init__(self, configs: DotMap):
        super().__init__(configs)

        self.configs = configs

    @staticmethod
    def truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def convert_examples_to_features(self, examples, tokenizer, max_length=512) -> list:
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for example in examples:
            # print(example.text_a)
            tokens_a = tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self.truncate_seq_pair(tokens_a, tokens_b, max_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_length - 2:
                    tokens_a = tokens_a[:(max_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_length
            assert len(input_mask) == max_length
            assert len(segment_ids) == max_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=input_mask,
                              token_type_ids=segment_ids,
                              label=example.label))
        return features
