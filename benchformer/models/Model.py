import torch
import pytorch_lightning as pl
from . import ModelBuilder


class Model(pl.LightningModule):

    def __init__(self, configs):
        super().__init__()

        self.configs = configs

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.net = ModelBuilder.build(self.configs.model)

    def forward(self, **params):

        device_params = {k: v.to(self.net.device) if v is not None else v for k, v in params.items()}

        outputs = self.net(**device_params)

        return outputs

    def training_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, labels = batch[:4]

        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               labels=labels)

        return {'loss': outputs.loss}

    def validation_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, labels = batch[:4]

        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               labels=labels)

        return {'val_loss': outputs.loss}

    def validation_end(self, outputs):
        avg_loss, avg_metrics = self.aggregate_metrics(outputs, stage='val')

        tensorboard_logs = {**{'val_loss': avg_loss}, **avg_metrics}

        return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, labels = batch[:4]

        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               labels=labels)

        return {'test_loss': outputs.loss}

    def test_end(self, outputs):
        avg_loss, avg_metrics = self.aggregate_metrics(outputs, stage='test')

        tensorboard_logs = {**{'test_loss': avg_loss}, **avg_metrics}

        return {'avg_test_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.fnet.parameters() if p.requires_grad], lr=self.configs.learning_rate,
                                eps=self.configs.epsilon)

    @pl.data_loader
    def train_dataloader(self):
        return self.train_dataloader

    @pl.data_loader
    def val_dataloader(self):
        return self.val_dataloader

    @pl.data_loader
    def test_dataloader(self):
        return self.test_dataloader

    def aggregate_metrics(self, outputs, stage='val'):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
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

    def __init__(self, loss=None, hidden_states=None, attentions=None):
        self.loss = loss
        self.hidden_states = hidden_states
        self.attentions = attentions

    def __getitem__(self, key):
        if key == "loss":
            return self.loss
        if key == "hidden_states":
            return self.hidden_states
        if key == "attentions":
            return self.attentions

        return None


class ModelForLMOutput(ModelOutput):

    def __init__(self,
                 loss=None,
                 hidden_states=None,
                 attentions=None,
                 prediction_logits=None,
                 seq_relationship_logits=None):

        super().__init__(loss, hidden_states, attentions)

        self.prediction_logits = prediction_logits
        self.seq_relationship_logits = seq_relationship_logits

    def __getitem__(self, key):

        if key == "prediction_logits":
            return self.prediction_logits
        if key == "seq_relationship_logits":
            return self.seq_relationship_logits

        return super().__getitem__(key)
