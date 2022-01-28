data_processors_implementations_dict = {}
features_processors_implementations_dict = {}


def register_processor(name: str = None) -> type:
    def decorate(cls: type, processor_name: str = None) -> type:

        if not processor_name:
            processor_name = cls.__module__ + '.' + cls.__name__

        if 'features' in processor_name.lower():
            global features_processors_implementations_dict

            if processor_name in features_processors_implementations_dict:
                print(
                    "Features processor class {} is already registered and will be overwritten!".format(processor_name)
                )

            features_processors_implementations_dict[processor_name] = cls
        else:
            global data_processors_implementations_dict

            if processor_name in data_processors_implementations_dict:
                print("Data processor class {} is already registered and will be overwritten!".format(processor_name))

            data_processors_implementations_dict[processor_name] = cls

        return cls

    return lambda cls: decorate(cls, name)


class DataProcessorBuilder(object):
    data_processors_dict = data_processors_implementations_dict

    @classmethod
    def build(cls, configs):
        try:
            return cls.data_processors_dict[configs.data_processor](configs)
        except KeyError:
            raise "{} data processor not implemented!".format(configs.data_processor)


class FeaturesProcessorBuilder(object):
    features_processors_dict = features_processors_implementations_dict

    @classmethod
    def build(cls, configs):
        try:
            return cls.features_processors_dict[configs.features_processor](configs)
        except KeyError:
            raise "{} features processor not implemented!".format(configs.features_processor)
