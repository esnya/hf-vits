from transformers import FeatureExtractionMixin


class VitsFeatureExtractor(FeatureExtractionMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("VitsFeatureExtractor is not implemented yet.")
