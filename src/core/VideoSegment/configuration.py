from transformers import PretrainedConfig


class VideoMAEConfig(PretrainedConfig):
    model_type = "VideoMAEForFrameClassification"

    def __init__(self,
                 num_classes=3,
                 backbone="MCG-NJU/videomae-base-finetuned-kinetics",
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.backbone = backbone
        self.id2label = {
            0: "negative",
            1: "positive",
            2: "neutral",
        }
        self.label2id = {
            "negative": 0,
            "positive": 1,
            "neutral": 2,
        }