from dotmap import DotMap

from benchformer.models import TransformerForLM
from benchformer.models.ModelBuilder import register_model


@register_model('FNetForLM')
class FNetForLM(TransformerForLM):

    def __init__(self, configs: DotMap):
        super().__init__(configs)
