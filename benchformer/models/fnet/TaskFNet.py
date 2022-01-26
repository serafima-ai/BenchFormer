from benchformer.models import TransformerForLM


class FNetForLM(TransformerForLM):

    def __init__(self, configs):
        super().__init__(configs)
