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

import torch.nn as nn
from dotmap import DotMap
from transformers.models.bert.modeling_bert import BertPreTrainingHeads, BertOnlyNSPHead, BertOnlyMLMHead
from benchformer.models import Model, ModelForLMOutput


class TransformerForLM(Model):
    """
    Transformer architecture model for language modeling task.

    Args:
        configs (DotMap): Object contains parameters required for model initialization and usage.

    Attributes:
        configs (DotMap): Stores model configuration.
        nsp_head (BertOnlyNSPHead): Next sentence prediction (classification) head.
        mlm_head (BertOnlyMLMHead): Masked language modeling head.
        heads (BertPreTrainingHeads): Next sentence prediction and masked language modeling head together.
        loss: Function for calculating the distance between predictions and ground truth.
        vocab_size (DataProcessor): Number of tokens in vocabulary.
        """
    def __init__(self, configs: DotMap):
        super().__init__(configs)

        self.nsp_head = None
        self.mlm_head = None
        self.heads = None

        if 'nsp_head' in self.configs.heads and 'mlm_head' in self.configs.heads:
            self.heads = BertPreTrainingHeads(self.net_config)
        elif 'nsp_head' in self.configs.heads:
            self.nsp_head = BertOnlyNSPHead(self.net_config)
        elif 'mlm_head' in self.configs.heads:
            self.mlm_head = BertOnlyMLMHead(self.net_config)

        self.loss = nn.CrossEntropyLoss()

        self.vocab_size = self.net_config.vocab_size

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                next_sentence_label=None,
                ):
        """
        The forward pass method.

        Args:
            input_ids (torch.LongTensor): Indices of input sequence tokens in the vocabulary.
            attention_mask (torch.FloatTensor): Mask to avoid performing attention on padding token indices.
            token_type_ids (torch.LongTensor): Segment token indices to indicate first and second portions of the input.
            position_ids (torch.LongTensor): Indices of positions of each input sequence tokens in the position
                embeddings.
            head_mask (torch.FloatTensor): Mask to nullify selected heads of the self-attention modules.
            inputs_embeds (torch.FloatTensor): Optionally, instead of passing `input_ids` you can choose to directly
                pass an embedded representation.
            labels (torch.LongTensor): Labels for computing the masked language modeling loss.
            next_sentence_label (torch.LongTensor): Labels for computing the next sequence prediction loss.

        Returns:
            outputs (TransformerForLMOutput): Object contains network`s forward pass output with calculated losses for
                masked language modeling, next sentence prediction tasks, and their sum.
        """
        params = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                  'position_ids': position_ids, 'head_mask': head_mask, 'inputs_embeds': inputs_embeds}

        outputs = super().forward(**params)
        sequence_output, pooled_output = outputs[:2]

        mlm_scores, nsp_score = None, None

        if self.heads:
            mlm_scores, nsp_score = self.heads(sequence_output, pooled_output)
        elif self.mlm_head:
            mlm_scores = self.mlm_head(sequence_output)
        elif self.nsp_head:
            nsp_score = self.nsp_head(pooled_output)
        else:
            raise ValueError("No output head found for language model")

        loss = None

        if labels is not None and mlm_scores is not None:
            mlm_loss = self.loss(mlm_scores.view(-1, self.vocab_size).cpu(), labels.view(-1))
            loss = mlm_loss

        if next_sentence_label is not None and nsp_score is not None:
            nsp_loss = self.loss(nsp_score.view(-1, 2).cpu(), next_sentence_label.view(-1))
            loss = loss + nsp_loss if loss else nsp_loss

        return TransformerForLMOutput(
            loss=loss,
            hidden_states=sequence_output,
            prediction_logits=mlm_scores,
            seq_relationship_logits=nsp_score
        )

    def training_step(self, batch, batch_nb) -> dict:
        """
        This method is used to compute the training loss.

        Args:
            batch: The output of DataLoader in the form of list, tuple, or tensor.
            batch_nb (int): Displays index of the current batch.

        Returns:
            dict: Dictionary contains calculated loss after the network`s forward pass.
        """
        input_ids, attention_mask, token_type_ids, mlm_labels, nsp_labels = batch[:5]

        outputs = self.forward(input_ids, attention_mask, token_type_ids, labels=mlm_labels,
                               next_sentence_label=nsp_labels)

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
        input_ids, attention_mask, token_type_ids, mlm_labels, nsp_labels = batch[:5]

        outputs = self.forward(input_ids, attention_mask, token_type_ids, labels=mlm_labels,
                               next_sentence_label=nsp_labels)

        return {'loss': outputs.loss}

    def test_step(self, batch, batch_nb) -> dict:
        """
        This method is used to compute the test loss.

        Args:
            batch: The output of DataLoader in the form of list, tuple, or tensor.
            batch_nb (int): Displays index of the current batch.

        Returns:
            dict: Dictionary contains calculated loss after the network`s forward pass.
        """
        input_ids, attention_mask, token_type_ids, mlm_labels, nsp_labels = batch[:5]

        outputs = self.forward(input_ids, attention_mask, token_type_ids, labels=mlm_labels,
                               next_sentence_label=nsp_labels)

        return {'loss': outputs.loss}


class TransformerForLMOutput(ModelForLMOutput):
    """
    The class is used to store the Transformer language model`s forward pass outputs.

    Args:
        loss: The calculated difference between the prediction and the ground truth after the forward pass.
        hidden_states: Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions: Attentions weights after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        prediction_logits: Model output used for the masked language modeling.
        seq_relationship_logits: Model output used for the next sentence prediction.

    Attributes:
        loss: Stores calculated loss.
        prediction_logits: Stores calculated logits for the masked language modeling task.
        seq_relationship_logits: Stores calculated logits for the next sentence prediction task.
    """
    def __init__(self,
                 loss=None,
                 hidden_states=None,
                 attentions=None,
                 prediction_logits=None,
                 seq_relationship_logits=None):

        super().__init__(loss, prediction_logits, seq_relationship_logits)

        self.hidden_states = hidden_states
        self.attentions = attentions

    def __getitem__(self, key: str):

        if key == "hidden_states":
            return self.hidden_states
        if key == "attentions":
            return self.attentions

        return super().__getitem__(key)
