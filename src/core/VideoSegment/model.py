from src.core.VideoSegment.configuration import VideoSegmentConfig
from transformers.modeling_outputs import TokenClassifierOutput
from src.core.VideoMAE import VideoMAEModel
from transformers import PreTrainedModel
import torch.nn.functional as F
from torchcrf import CRF
import torch.nn as nn
from abc import ABC
import torch


class VideoSegmentModel(PreTrainedModel, ABC):
    config_class = VideoSegmentConfig

    def __init__(self, config: VideoSegmentConfig, *inputs, **kwargs):
        super(VideoSegmentModel, self).__init__(config, *inputs, **kwargs)
        self.num_classes = config.num_classes
        self.seq_length = config.seq_length
        self.backbone = VideoMAEModel.from_pretrained(config.backbone_name)
        self.num_frames = self.backbone.config.num_frames
        hidden_state_dim = (self.backbone.config.image_size / self.backbone.config.patch_size) ** 2
        hidden_state_dim = hidden_state_dim / self.backbone.config.tubelet_size * self.backbone.config.hidden_size
        self.hidden_state_dim = int(hidden_state_dim)
        self.mlp = nn.Linear(self.hidden_state_dim, self.num_classes)
        self.crf = CRF(self.num_classes, batch_first=True)

    def forward(self, pixel_values, labels=None):
        backbone_output = self.backbone(pixel_values)
        last_hidden_state = backbone_output["last_hidden_state"]
        hidden_shape = last_hidden_state.shape
        reshape_hidden_state = torch.reshape(last_hidden_state, (hidden_shape[0], self.num_frames, -1))
        assert self.hidden_state_dim == reshape_hidden_state.shape[-1]
        frame_hidden_state = self.mlp(reshape_hidden_state)
        predict = self.crf.decode(frame_hidden_state)
        loss = None
        if labels is not None:
            loss = -self.crf(F.log_softmax(frame_hidden_state, 2), labels, reduction='mean')
        out = TokenClassifierOutput(
            loss=loss,
            logits=predict,
            hidden_states=frame_hidden_state,
            attentions=backbone_output.attentions,
        )
        return out
