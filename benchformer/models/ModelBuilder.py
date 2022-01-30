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

from dotmap import DotMap

models_implementations_dict = {}


def register_model(name: str = None) -> type:
    """
    This decorator function is used to register (store in a dictionary) classes of implemented model architectures.

    Args:
        name (str): The key of the model class in a dictionary. Can be derived from the class name if no name is
            specified.

    Returns:
        type: Type of the registered class.
    """

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
    """
    This class is a factory for models creation.

    Attributes:
        models_dict (dict): Dictionary with model name as a key and class as a value.
    """
    models_dict = models_implementations_dict

    @classmethod
    def build(cls, configs: DotMap):
        """
        Creates model object using model type specified in configs.

        Args:
            configs (DotMap): Object contains parameters required for model usage.

        Raises:
            KeyError: No architecture found for the specified model type.

        Returns:
            Model: Created object of model
        """
        try:
            return cls.models_dict[configs.model_type](configs)
        except KeyError:
            raise "{} architecture not implemented!".format(configs.model_type)
