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

data_processors_implementations_dict = {}
features_processors_implementations_dict = {}


def register_processor(name: str = None) -> type:
    """
    This decorator function is used to register (store in a dictionary) classes of data and features processors.

    Args:
        name (str): The key of the processor class in a dictionary. Can be derived from the class name if no name is
            specified.

    Returns:
        type: Type of the registered class.
    """

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
    """
    This class is a factory for data processors creation.

    Attributes:
        data_processors_dict (dict): Dictionary with data processor name as a key and class as a value.
    """
    data_processors_dict = data_processors_implementations_dict

    @classmethod
    def build(cls, configs: DotMap):
        """
        Creates data processor object using data processor type specified in configs.

        Args:
            configs (DotMap): Object contains parameters required for data processor usage.

        Raises:
            KeyError: No data processor found for the specified data processor type.

        Returns:
            Model: Created object of data processor
        """
        try:
            return cls.data_processors_dict[configs.data_processor](configs)
        except KeyError:
            raise "{} data processor not implemented!".format(configs.data_processor)


class FeaturesProcessorBuilder(object):
    """
    This class is a factory for features processors creation.

    Attributes:
        features_processors_dict (dict): Dictionary with features processor name as a key and class as a value.
    """
    features_processors_dict = features_processors_implementations_dict

    @classmethod
    def build(cls, configs: DotMap):
        """
        Creates features processor object using features processor type specified in configs.

        Args:
            configs (DotMap): Object contains parameters required for features processor usage.

        Raises:
            KeyError: No features processor found for the specified features processor type.

        Returns:
            Model: Created object of features processor
        """
        try:
            return cls.features_processors_dict[configs.features_processor](configs)
        except KeyError:
            raise "{} features processor not implemented!".format(configs.features_processor)
