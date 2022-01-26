from .Model import (
    Model,
    ModelForLMOutput,
)

from .TransformerModel import (
    TransformerForLM,
)

from .fnet import (
    FNetModel,
    FNetForLM,
)

models_implementations_dict = {
    'FNet': FNetModel,
    'FNetForLM': FNetForLM
}


class ModelBuilder(object):
    models_dict = models_implementations_dict

    @classmethod
    def build(cls, configs):
        try:
            return cls.models_dict[configs.model_name](configs)
        except KeyError:
            raise "{} architecture not implemented!".format(configs.model_name)
