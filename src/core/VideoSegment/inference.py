from src.core.VideoSegment.configuration import VideoSegmentConfig
from src.core.VideoSegment.model import VideoSegmentModel
from src import CONFIG, DEVICE
import numpy as np
import torch


class Inference:
    def __init__(self):
        self.args = CONFIG
        self._init_model()

    def _init_model(self):
        self.config = VideoSegmentConfig.from_pretrained(self.args["output_dir"])
        self.id2label = self.config.id2label
        self.model = VideoSegmentModel.from_pretrained(self.args["output_dir"]).to(DEVICE)
        print("Loaded VideoSegmentModel")

    def predict(self, pixel_values: torch.Tensor):
        with torch.no_grad():
            output = self.model(pixel_values)
        predict = output["logits"]
        logits = output["hidden_states"].cpu().numpy()
        return predict, logits
