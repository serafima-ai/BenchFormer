import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPreTrainingHeads, BertOnlyNSPHead, BertOnlyMLMHead
from benchformer.models import Model, ModelForLMOutput


class TransformerForLM(Model):

    def __init__(self, configs):
        super().__init__(configs)

        self.nsp_head = None
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

        if labels and mlm_scores:
            mlm_loss = self.loss(mlm_scores.view(-1, self.vocab_size), labels.view(-1))
            loss = mlm_loss

        if next_sentence_label and nsp_score:
            nsp_loss = self.loss(nsp_score.view(-1, 2), next_sentence_label.view(-1))
            loss = loss + nsp_loss if loss else nsp_loss

        return TransformerForLMOutput(
            loss=loss,
            hidden_states=sequence_output,
            prediction_logits=mlm_scores,
            seq_relationship_logits=nsp_score
        )

    def training_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, mlm_labels, nsp_labels = batch[:5]

        outputs = self.forward(input_ids, attention_mask, token_type_ids, labels=mlm_labels,
                               next_sentence_label=nsp_labels)

        return {'loss': outputs.loss}

    def validation_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, mlm_labels, nsp_labels = batch[:5]

        outputs = self.forward(input_ids, attention_mask, token_type_ids, labels=mlm_labels,
                               next_sentence_label=nsp_labels)

        return {'loss': outputs.loss}

    def test_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, mlm_labels, nsp_labels = batch[:5]

        outputs = self.forward(input_ids, attention_mask, token_type_ids, labels=mlm_labels,
                               next_sentence_label=nsp_labels)

        return {'loss': outputs.loss}


class TransformerForLMOutput(ModelForLMOutput):

    def __init__(self,
                 loss=None,
                 hidden_states=None,
                 attentions=None,
                 prediction_logits=None,
                 seq_relationship_logits=None):

        super().__init__(loss, prediction_logits, seq_relationship_logits)

        self.hidden_states = hidden_states
        self.attentions = attentions

    def __getitem__(self, key):

        if key == "hidden_states":
            return self.hidden_states
        if key == "attentions":
            return self.attentions

        return super().__getitem__(key)
