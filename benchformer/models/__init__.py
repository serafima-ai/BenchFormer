from .fnet import (
    FNetModel,
    FNetForLM,
)

from .Model import Model, ModelForLMOutput

models_implementations_dict = {
    'fnet': fnet.FNetModel
}


class ModelBuilder(object):
    models_dict = models_implementations_dict

    @classmethod
    def build(cls, configs):
        try:
            return cls.models_dict[configs.model_name](configs)
        except KeyError:
            raise "{} architecture not implemented!".format(configs.model_name)
