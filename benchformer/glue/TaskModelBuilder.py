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

tasks_implementations_dict = {}


def register_task(name: str = None) -> type:
    """
    This decorator function is used to register (store in a dictionary) classes of task models.

    Args:
        name (str): The key of the task model class in a dictionary. Can be derived from the class name if no name is
            specified.

    Returns:
        type: Type of the registered class.
    """

    def decorate(cls: type, task_name: str = None) -> type:
        global tasks_implementations_dict

        if not task_name:
            task_name = cls.__module__ + '.' + cls.__name__

        if task_name in tasks_implementations_dict:
            print("Task model class {} is already registered and will be overwritten!".format(task_name))

        tasks_implementations_dict[task_name] = cls
        return cls

    return lambda cls: decorate(cls, name)


class TaskModelBuilder(object):
    """
    This class is a factory for task models creation.

    Attributes:
        tasks_dict (dict): Dictionary with task name as a key and class as a value.
    """
    tasks_dict = tasks_implementations_dict

    @classmethod
    def build(cls, configs: DotMap):
        """
        Creates task model object using task name specified in configs.

        Args:
            configs (DotMap): Object contains parameters required for task model usage.

        Raises:
            KeyError: No task model found for the specified task name.

        Returns:
            Model: Created object of task model.
        """
        try:
            return cls.tasks_dict[configs.task_name](configs)
        except KeyError:
            raise "{} task not implemented!".format(configs.task_name)
