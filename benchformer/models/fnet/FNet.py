import json
import logging
import torch
import torch.nn as nn
import torch.fft
from dotmap import DotMap
from functools import partial
from transformers.models.bert.modeling_bert import (
    BertEmbeddings, BertAttention, BertIntermediate, BertOutput, BertPooler
)

from benchformer.models import register_model

try:
    from scipy import linalg

    _scipy_available = True
except ImportError:
    _scipy_available = False


class FNetBlock(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.fft = FNetBasicFourierTransform(configs)

    def forward(self, x, attention_mask=None):
        # x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        x = torch.fft.fft2(x).real
        # x = self.fft(x)

        return x, attention_mask


class FNetFeedForward(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.linear1 = nn.Linear(configs.input_size, configs.intermediate_size)
        self.activation = nn.GELU() if configs.hidden_act == 'gelu' else nn.ReLU()
        # self.dropout1 = nn.Dropout(configs.intermediate_dropout_prob)
        self.linear2 = nn.Linear(configs.intermediate_size, configs.input_size)
        self.dropout2 = nn.Dropout(configs.intermediate_dropout_prob)

        self.configs = configs

        # self.ff = nn.Sequential(
        # nn.Linear(configs.input_size, configs.hidden_size),
        # nn.GELU(),
        # nn.Dropout(configs.feed_forward_dropout_prob),
        # nn.Linear(configs.hidden_size, configs.input_size),
        # nn.Dropout(configs.feed_forward_dropout_prob)
        # )

    def forward(self, x):
        hidden_states = self.linear1(x)
        hidden_states = self.activation(hidden_states)
        output_states = self.linear2(hidden_states)
        # drop_output = self.dropout1(output)

        # return self.dropout2(self.linear2(drop_output))
        return self.dropout2(output_states)

    def init_weights(self, init_range):
        self.linear1.weight.data.uniform_(-init_range, init_range)
        self.linear2.weight.data.uniform_(-init_range, init_range)


class FNetLayer(nn.Module):

    def __init__(self, configs, bert_self_attn=False):
        super().__init__()

        self.attn = None
        self.intermediate = None
        self.output = None

        if bert_self_attn:
            self.attn = BertAttention(configs)
            self.intermediate = BertIntermediate(configs)
            self.output = BertOutput(configs)
        else:
            self.attn = FNetBlock(configs)

        self.ff = FNetFeedForward(configs)
        self.attn_norm = nn.LayerNorm(configs.input_size, eps=configs.layer_norm_eps)
        self.ff_norm = nn.LayerNorm(configs.input_size, eps=configs.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        # x = self.attn(self.attn_norm(x)) + x
        # return self.ff(self.ff_norm(x)) + x

        attn_outputs = self.attn(hidden_states, attention_mask)
        attn_output = attn_outputs[0]

        # print(type(self.attn))

        # if isinstance(self.attn, BertAttention):
        #     print(hidden_states[0].tolist())

        # attention_mask, x = attn_output[:2]

        if isinstance(self.attn, BertAttention):
            intermediate_output = self.intermediate(attn_output)
            return self.output(intermediate_output, attn_output), attention_mask

        hidden_states = self.attn_norm(attn_output + hidden_states)

        return self.ff_norm(self.ff(hidden_states) + hidden_states), attention_mask

    def init_weights(self, init_range):
        # initrange = self.config.initializer_range #0.1
        # self.attn.weight.data.uniform_(-init_range, init_range)

        self.ff.init_weights(init_range)

        if isinstance(self.attn, BertAttention):
            self.attn.init_weights(init_range)


class FNetEncoder(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.layers = nn.ModuleList([])

        for i in range(configs.num_layers):
            self.layers.append(FNetLayer(configs, bert_self_attn=i in configs.spec_attention_layers))

    def forward(self, x, attn=None):
        for layer in self.layers:
            x, attn = layer(x, attn)

        return x

    def init_weights(self, init_range):
        for layer in self.layers:
            layer.init_weights(init_range)


@register_model('FNet')
class FNetModel(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.config = configs

        self.embeddings = BertEmbeddings(configs)
        self.encoder = FNetEncoder(configs)
        self.pooler = BertPooler(configs)

    def init_weights(self):
        self.encoder.init_weights(self.config.initializer_range)
        self.pooler.init_weights(self.config.initializer_range)

    def forward(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds):
        embeddings = self.embeddings(input_ids, token_type_ids, position_ids, inputs_embeds)
        sequence_output = self.encoder(embeddings)
        pooler_output = self.pooler(sequence_output)

        return sequence_output, pooler_output

    @classmethod
    def load_configs_from_file(cls, configs_path: str) -> DotMap:
        with open(configs_path, "r") as f:
            content = f.read()
            return DotMap(json.loads(content))


# # Adapted from https://github.com/google-research/google-research/blob/master/f_net/fourier.py

def _two_dim_matmul(x, matrix_dim_one, matrix_dim_two):
    """Applies 2D matrix multiplication to 3D input arrays."""
    seq_length = x.shape[1]
    matrix_dim_one = matrix_dim_one[:seq_length, :seq_length]
    x = x.type(torch.complex64)
    return torch.einsum("bij,jk,ni->bnk", x, matrix_dim_two, matrix_dim_one)


def two_dim_matmul(x, matrix_dim_one, matrix_dim_two):
    return _two_dim_matmul(x, matrix_dim_one, matrix_dim_two)


def fftn(x):
    """
    Applies n-dimensional Fast Fourier Transform (FFT) to input array.
    Args:
        x: Input n-dimensional array.
    Returns:
        n-dimensional Fourier transform of input n-dimensional array.
    """
    out = x
    for axis in reversed(range(x.ndim)[1:]):  # We don't need to apply FFT to last axis
        out = torch.fft.fft(out, axis=axis)
    return out


class FNetBasicFourierTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._init_fourier_transform(config)

    def _init_fourier_transform(self, config):
        if not config.use_tpu_fourier_optimizations:
            self.fourier_transform = partial(torch.fft.fftn, dim=(1, 2))
        elif config.max_position_embeddings <= 4096:
            if _scipy_available:
                self.register_buffer(
                    "dft_mat_hidden", torch.tensor(linalg.dft(config.hidden_size), dtype=torch.complex64)
                )
                self.register_buffer(
                    "dft_mat_seq", torch.tensor(linalg.dft(config.tpu_short_sequence_length), dtype=torch.complex64)
                )
                self.fourier_transform = partial(
                    two_dim_matmul, matrix_dim_one=self.dft_mat_seq, matrix_dim_two=self.dft_mat_hidden
                )
            else:
                logging.warning(
                    "SciPy is need for DFT matrix calculation and is not found. Using TPU optimized fast fourier transform instead."
                )
                self.fourier_transform = fftn
        else:
            self.fourier_transform = fftn

    def forward(self, hidden_states):

        # NOTE: We do not use torch.vmap as it is not integrated into PyTorch stable versions.
        # Interested users can modify the code to use vmap from the nightly versions, getting the vmap from here:
        # https://pytorch.org/docs/master/generated/torch.vmap.html. Note that fourier transform methods will need
        # change accordingly.

        # outputs = self.fourier_transform(hidden_states).real
        return two_dim_matmul(hidden_states, self.dft_mat_seq, self.dft_mat_hidden).real  # outputs
