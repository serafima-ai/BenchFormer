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
import pytorch_lightning as pl
from dotmap import DotMap
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from . import ModelBuilder
from benchformer.data import DataProcessorBuilder


class Model(pl.LightningModule):
    """
    Abstract class that inherits LightningModule logic and implements additional methods for forward pass, dataset
    loading, and metrics calculation.

    Args:
        configs (DotMap): Object contains parameters required for model initialization and usage.

    Attributes:
        configs (DotMap): Stores model configuration.
        device (str): Specifies device type for responsible to load a tensor into memory.
        net_config (DotMap): Used to store network architecture configuration.
        net (Model): Network object with the specified type of architecture (Bert, FNet, etc.).
        tokenizer (AutoTokenizer): Used to translate text into a sequence of integers. Multilingual cased Bert tokenizer
            is used by default if no value is specified.
        data_processor (DataProcessor): Object responsible for dataset preprocessing.
    """
    def __init__(self, configs: DotMap):
        super().__init__()

        self.configs = configs

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.net_config = self.configs.net_config

        self.net = ModelBuilder.build(self.net_config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.configs.get('tokenizer', 'bert-base-multilingual-cased'))

        self.data_processor = DataProcessorBuilder.build(self.configs.data)

        self.train_loader, self.val_loader, self.test_loader = None, None, None

    def forward(self, **params):
        """
        The forward pass method, which places batch input data on the device, passes it through the network and returns
            net`s output.

        Args:
            **params (dict): Input data with strings as keys and tensors as values.

        Returns:
            outputs (ModelOutput): Object contains network`s forward pass output.
        """
        device_params = {k: v.to(self.device) if v is not None else v for k, v in params.items()}

        outputs = self.net(**device_params)

        return outputs

    def training_step(self, batch, batch_nb) -> dict:
        """
        This method is used to compute the training loss.

        Args:
            batch: The output of DataLoader in the form of list, tuple, or tensor.
            batch_nb (int): Displays index of the current batch.

        Returns:
            dict: Dictionary contains calculated loss after the network`s forward pass.
        """
        input_ids, attention_mask, token_type_ids, labels = batch[:4]

        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               labels=labels)

        return {'loss': outputs.loss}

    def validation_step(self, batch, batch_nb) -> dict:
        """
        This method is used to compute the validation loss.

        Args:
            batch: The output of DataLoader in the form of list, tuple, or tensor.
            batch_nb (int): Displays index of the current batch.

        Returns:
            dict: Dictionary contains calculated loss after the network`s forward pass.
        """
        input_ids, attention_mask, token_type_ids, labels = batch[:4]

        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               labels=labels)

        return {'val_loss': outputs.loss}

    def validation_end(self, outputs) -> dict:
        """
        Aggregates output of validation steps. Returns averaged values of loss and metrics.

        Args:
            outputs: Outputs of validation steps.

        Returns:
            dict: Dictionary contains average validation loss and metrics.
        """
        avg_loss, avg_metrics = self.aggregate_metrics(outputs, stage='val')

        tensorboard_logs = {**{'val_loss': avg_loss}, **avg_metrics}

        return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_nb) -> dict:
        """
        This method is used to compute the test loss.

        Args:
            batch: The output of DataLoader in the form of list, tuple, or tensor.
            batch_nb (int): Displays index of the current batch.

        Returns:
            dict: Dictionary contains calculated loss after the network`s forward pass.
        """
        input_ids, attention_mask, token_type_ids, labels = batch[:4]

        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               labels=labels)

        return {'test_loss': outputs.loss}

    def test_end(self, outputs) -> dict:
        """
        Aggregates output of test steps. Returns averaged values of loss and metrics.

        Args:
            outputs: Outputs of test steps.

        Returns:
            dict: Dictionary contains average test loss and metrics.
        """
        avg_loss, avg_metrics = self.aggregate_metrics(outputs, stage='test')

        tensorboard_logs = {**{'test_loss': avg_loss}, **avg_metrics}

        return {'avg_test_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad],
                                lr=self.configs.optimizer.learning_rate,
                                eps=self.configs.optimizer.epsilon)

    @pl.data_loader
    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    @pl.data_loader
    def val_dataloader(self) -> DataLoader:
        return self.val_loader

    @pl.data_loader
    def test_dataloader(self) -> DataLoader:
        return self.test_loader

    def load_dataset(self) -> bool:
        dataset = self.data_processor.prepare_dataset(self.tokenizer)
        self.train_loader, self.val_loader, self.test_loader = dataset

        return self.train_loader is not None and self.val_loader is not None

    def aggregate_metrics(self, outputs, stage='val'):
        """
        Aggregates output of val/test steps. Returns averaged values of loss and metrics.

        Args:
            outputs: Outputs of val/test steps.
            stage (str): Specifies the stage of obtained outputs.

        Returns:
            dict: Dictionary contains average val/test loss and metrics.
        """
        avg_loss = torch.stack([x[stage + '_loss'] for x in outputs]).mean()
        avg_metrics = {}

        for metric in self.metrics:
            key_name = stage + '_' + metric
            try:
                avg_metrics.update(
                    {'avg_' + key_name: torch.stack(
                        [x[key_name] if type(x[key_name]) is torch.Tensor
                         else torch.tensor(x[key_name]) for x in outputs if x[key_name] >= 0.0]
                    ).mean()
                     })
            except RuntimeError:
                avg_metrics.update({'avg_' + key_name: torch.tensor(0.0)})

        return avg_loss, avg_metrics


class ModelOutput(object):
    """
    The class used to store model`s forward pass outputs.

    Args:
        loss: The calculated difference between the prediction and the ground truth after the forward pass.

    Attributes:
        loss: Stores calculated loss.
    """

    def __init__(self, loss=None):
        self.loss = loss

    def __getitem__(self, key: str):
        if key == "loss":
            return self.loss

        return None


class ModelForLMOutput(ModelOutput):
    """
    The class is used to store the language model`s forward pass outputs.

    Args:
        loss: The calculated difference between the prediction and the ground truth after the forward pass.
        prediction_logits: Model output used for the masked language modeling.
        seq_relationship_logits: Model output used for the next sentence prediction.

    Attributes:
        loss: Stores calculated loss.
        prediction_logits: Stores calculated logits for the masked language modeling task.
        seq_relationship_logits: Stores calculated logits for the next sentence prediction task.
    """
    def __init__(self,
                 loss=None,
                 prediction_logits=None,
                 seq_relationship_logits=None):

        super().__init__(loss)

        self.prediction_logits = prediction_logits
        self.seq_relationship_logits = seq_relationship_logits

    def __getitem__(self, key: str):

        if key == "prediction_logits":
            return self.prediction_logits
        if key == "seq_relationship_logits":
            return self.seq_relationship_logits

        return super().__getitem__(key)
