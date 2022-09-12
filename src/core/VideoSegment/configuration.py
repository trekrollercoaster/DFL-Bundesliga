from transformers import PretrainedConfig


class VideoSegmentConfig(PretrainedConfig):
    model_type = "VideoMAEForFrameClassification"

    def __init__(self,
                 backbone_name="MCG-NJU/videomae-base-finetuned-kinetics",
                 seq_length=16,
                 label2id=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
        self.seq_length = seq_length
        self.label2id = {"O": 0} if label2id is None else label2id
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_classes = len(self.id2label.keys())
