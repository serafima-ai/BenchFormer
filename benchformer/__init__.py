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

import copy
from pathlib import Path
from dotmap import DotMap

from . import (
    data,
    models,
    glue,
)

from .configs import process_json_config


def build(config: DotMap = None, config_path: str = ''):
    """
    Uses DotMap object or loads configuration from file to build specified object (model/task model).

    Args:
        config (DotMap): Configuration object.
        config_path (str): Path to the file with JSON-configuration.

    Returns:
        object: The object specified in the configuration.
    """
    if not config and not config_path:
        raise Exception("No configs source specified!")

    if config_path:
        config = process_json_config(json_file=config_path)

    if config.task:
        return build_tasks(config.task)


def build_tasks(config: DotMap) -> list:
    """
    Creates a task from the configuration. If there are several models specified in the task config, then the output
    will contain a task for each model.

    Args:
        config (DotMap): Task configuration object.

    Returns:
        tasks (list): The list of tasks.
    """
    tasks = []

    for model in config.models:

        if isinstance(model.net_configs, str):

            net_config_path = Path(model.net_configs)
            if net_config_path.is_file():
                model.net_config = process_json_config(json_file=net_config_path)
            else:
                tasks.append(None)
                continue
        cfg = copy.deepcopy(config)
        cfg.models = None
        cfg.model = model
        cfg.model.data = cfg.data
        tasks.append(glue.TaskModelBuilder.build(cfg))

    return tasks
