from src.core.VideoSegment.configuration import VideoSegmentConfig
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import PreTrainedModel, PretrainedConfig
from src.core.VideoMAE import VideoMAEModel
import torch.nn.functional as F
from torchcrf import CRF
from abc import ABC
import torch


class VideoSegmentModel(PreTrainedModel, ABC):
    config_class = VideoSegmentConfig()

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super(PreTrainedModel).__init__(config, *inputs, **kwargs)
        self.num_classes = config.num_classes
        self.seq_length = config.seq_length
        self.backbone = VideoMAEModel.from_pretrained(config.backbone_name)
        self.crf = CRF(self.num_classes, batch_first=True)

    def forward(self, pixel_values, labels=None):
        backbone_output = self.backbone(pixel_values)
        last_hidden_state = backbone_output["last_hidden_state"]
        predict = self.crf.decode(last_hidden_state)
        for j in range(len(predict)):
            while len(predict[j]) < self.seq_length:
                predict[j].append(self.num_classes - 1)
        predict = torch.tensor(predict).contiguous().view(-1)
        loss = None
        if labels is not None:
            loss = -self.crf(F.log_softmax(last_hidden_state, 2), labels, reduction='mean')
        out = TokenClassifierOutput(
            loss=loss,
            logits=predict,
            hidden_states=backbone_output.hidden_states,
            attentions=backbone_output.attentions,
        )
        return out
