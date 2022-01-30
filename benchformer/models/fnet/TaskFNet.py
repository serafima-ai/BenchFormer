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

from benchformer.models import TransformerForLM
from benchformer.models.ModelBuilder import register_model


@register_model('FNetForLM')
class FNetForLM(TransformerForLM):
    """
    FNet architecture model for language modeling task.

    Args:
        configs (DotMap): Object contains parameters required for model initialization and usage.
    """

    def __init__(self, configs: DotMap):
        super().__init__(configs)
