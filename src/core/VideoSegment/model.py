from src.core.VideoSegment.configuration import VideoMAEConfig
from transformers import PreTrainedModel, PretrainedConfig
from src.core.VideoMAE import VideoMAEModel
from abc import ABC
import numpy as np
import torch


class VideoSegmentModel(PreTrainedModel, ABC):
    config_class = VideoMAEConfig()

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super(PreTrainedModel).__init__(config, *inputs, **kwargs)
        self.num_classes = config.num_classes
        self.backbone = VideoMAEModel.from_pretrained(config.backbone_name)
