import torch
import torch.nn as nn

from . import register_task
from ..models import TransformerForLM


@register_task('GluePARusModel')
class PARusTaskModel(TransformerForLM):

    def __init__(self, configs):
        super().__init__(configs.model)

        self.proba_fn = nn.Softmax(dim=1)
        self.metrics = ['accuracy']

    def training_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, nsp_labels = batch[:4]

        outputs = self.forward(input_ids, attention_mask, token_type_ids, next_sentence_label=nsp_labels)

        return {'loss': outputs.loss}

    def validation_step(self, batch, batch_nb):
        return self.calculate_metrics(batch)

    def test_step(self, batch, batch_nb):
        return self.calculate_metrics(batch, stage='test')

    def calculate_metrics(self, batch, stage='val'):
        input_ids, attention_mask, token_type_ids, nsp_labels = batch[:4]

        outputs = self.forward(input_ids, attention_mask, token_type_ids, next_sentence_label=nsp_labels)

        nsp_probas = self.proba_fn(outputs.seq_relationship_logits)
        nsp_labels_pred = torch.tensor([int(probas[0] > probas[1]) for probas in nsp_probas])

        accuracy = sum(nsp_labels == nsp_labels_pred) / len(nsp_labels_pred)

        return {stage + '_loss': outputs.loss, stage + '_accuracy': accuracy}
