from transformers import Trainer, HfArgumentParser, TrainingArguments, set_seed
from src.core.VideoSegment.configuration import VideoSegmentConfig
from src.core.VideoSegment.load_dataset import LoadDataset
from src.core.VideoSegment.model import VideoSegmentModel
from src import BASE_PATH, CONFIG, ARG_PATH
import os


class TrainModel:
    def __init__(self):
        self.args = CONFIG
        self.data_path = os.path.join(BASE_PATH, "data/train_source")
        self._init_dataset()

    def _init_dataset(self):
        dataset_util = LoadDataset()
        self.dataset = dataset_util.load_dataset()

    def train(self):
        pass
