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

import json
from dotmap import DotMap


def process_json_config(content_json: str = '', json_file: str = '') -> DotMap:
    """
    Reads JSON data from string or file and converts it to the DotMap object.

    Args:
        content_json (str): Serialized JSON-document.
        json_file (str): Path to the file with JSON-content.

    Returns:
        type: Type of the registered class.
    """
    content = DotMap()

    if content_json:
        content = parse_json(content_json)
    elif json_file:
        content = parse_json(parse_configs_file(json_file))
    else:
        return content

    content = DotMap(content)

    return content


def parse_configs_file(filepath) -> str:
    with open(filepath, 'r') as f:
        content = f.read()

    return content


def parse_json(content: str) -> dict:
    return json.loads(content)
