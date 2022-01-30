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
import torch.nn as nn
from dotmap import DotMap

from . import register_task
from ..models import TransformerForLM


@register_task('GluePARusModel')
class PARusTaskModel(TransformerForLM):
    """
    Task model for the PARus task (Choice of Plausible Alternatives for Russian language) from RussianSuperGlue. For
    more info: https://russiansuperglue.com/tasks/task_info/PARus

    Args:
        configs (DotMap): Contains parameters required for the initialization and usage of the task model.

    Attributes:
        proba_fn: Activation function to be applied to the output of the next sentence prediction task.
        metrics (list): Metrics to be calculated after validation and test steps.
        """

    def __init__(self, configs: DotMap):
        super().__init__(configs.model)

        self.proba_fn = nn.Softmax(dim=1)
        self.metrics = ['accuracy']

    def training_step(self, batch, batch_nb) -> dict:
        """
        This method is used to compute the training loss.

        Args:
            batch: The output of DataLoader in the form of list, tuple, or tensor.
            batch_nb (int): Displays index of the current batch.

        Returns:
            dict: Dictionary contains calculated loss after the network`s forward pass.
        """
        input_ids, attention_mask, token_type_ids, nsp_labels = batch[:4]

        outputs = self.forward(input_ids, attention_mask, token_type_ids, next_sentence_label=nsp_labels)

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
        return self.calculate_metrics(batch)

    def test_step(self, batch, batch_nb) -> dict:
        """
        This method is used to compute the test loss.

        Args:
            batch: The output of DataLoader in the form of list, tuple, or tensor.
            batch_nb (int): Displays index of the current batch.

        Returns:
            dict: Dictionary contains calculated loss after the network`s forward pass.
        """
        return self.calculate_metrics(batch, stage='test')

    def calculate_metrics(self, batch, stage: str = 'val') -> dict:
        """
        This method is used to compute the val/test loss and metrics.

        Args:
            batch: The output of DataLoader in the form of list, tuple, or tensor.
            stage (str): Specifies the current stage of processing (validation or test).

        Returns:
            dict: Dictionary contains calculated loss and metrics after the network`s forward pass.
        """
        input_ids, attention_mask, token_type_ids, nsp_labels = batch[:4]

        outputs = self.forward(input_ids, attention_mask, token_type_ids, next_sentence_label=nsp_labels)

        nsp_probas = self.proba_fn(outputs.seq_relationship_logits)
        nsp_labels_pred = torch.tensor([int(probas[0] > probas[1]) for probas in nsp_probas])

        accuracy = sum(nsp_labels == nsp_labels_pred) / len(nsp_labels_pred)

        return {stage + '_loss': outputs.loss, stage + '_accuracy': accuracy}
