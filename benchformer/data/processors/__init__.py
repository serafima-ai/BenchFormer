from .Processor import (
    DataProcessor,
    FeaturesProcessor,
)

from .TransformerProcessor import (
    TransformerDataProcessor,
)

from .GlueTaskProcessor import (
    PARusDataProcessor,
    PARusFeaturesProcessor,
)

data_processors_implementations_dict = {
    'TransformerProcessor': TransformerDataProcessor,
    'GluePARusProcessor': PARusDataProcessor
}

features_processors_implementations_dict = {
    'GluePARusFeaturesProcessor': PARusFeaturesProcessor
}


class DataProcessorBuilder(object):
    data_processors_dict = data_processors_implementations_dict

    @classmethod
    def build(cls, configs):
        try:
            return cls.data_processors_dict[configs.data_processor_name](configs)
        except KeyError:
            raise "{} data processor not implemented!".format(configs.data_processor_name)


class FeaturesProcessorBuilder(object):
    features_processors_dict = features_processors_implementations_dict

    @classmethod
    def build(cls, configs):
        try:
            return cls.features_processors_dict[configs.features_processor_name](configs)
        except KeyError:
            raise "{} features processor not implemented!".format(configs.features_processor_name)
