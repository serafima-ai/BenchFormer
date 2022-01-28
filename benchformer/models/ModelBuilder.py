models_implementations_dict = {}


def register_model(name: str = None) -> type:
    def decorate(cls: type, model_name: str = None) -> type:
        global models_implementations_dict

        if not model_name:
            model_name = cls.__module__ + '.' + cls.__name__

        if model_name in models_implementations_dict:
            print("Model class {} is already registered and will be overwritten!".format(model_name))

        models_implementations_dict[model_name] = cls

        return cls

    return lambda cls: decorate(cls, name)


class ModelBuilder(object):
    models_dict = models_implementations_dict

    @classmethod
    def build(cls, configs):
        print(configs.model_type)
        try:
            return cls.models_dict[configs.model_type](configs)
        except KeyError:
            raise "{} architecture not implemented!".format(configs.model_type)
