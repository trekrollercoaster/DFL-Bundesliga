from transformers import Trainer, HfArgumentParser, TrainingArguments, set_seed
from src.core.VideoSegment.load_dataset import LoadDataset, VideoDataCollator
from src.core.VideoSegment.configuration import VideoSegmentConfig
from src.core.VideoSegment.model import VideoSegmentModel
from src import BASE_PATH, CONFIG, ARG_PATH
import logging
import os

logger = logging.getLogger(__name__)


class TrainModel:
    def __init__(self):
        self.args = CONFIG
        self.data_path = os.path.join(BASE_PATH, "data/train_source")
        self._init_dataset()
        self._init_model()

    def _init_dataset(self):
        dataset_util = LoadDataset()
        dataset = dataset_util.load_dataset()
        self.label2id = dataset_util.label2id
        self.train_data = dataset["train"]
        self.valid_data = dataset["valid"]

    def _init_model(self):
        VideoSegmentConfig.register_for_auto_class()
        VideoSegmentModel.register_for_auto_class()
        self.config = VideoSegmentConfig(
            backbone_name=self.args["backbone"],
            seq_length=self.args["seq_length"],
            label2id=self.label2id
        )
        self.model = VideoSegmentModel(self.config)
        print("Loaded VideoSegmentModel")

    def train(self):
        parser = HfArgumentParser(TrainingArguments)
        args = parser.parse_json_file(json_file=ARG_PATH)
        training_args: TrainingArguments = args[0]

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        )
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            bool(training_args.local_rank != -1),
            training_args.fp16,
        )

        # Set seed
        set_seed(training_args.seed)

        # Initialize our Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,
            eval_dataset=self.valid_data,
            data_collator=VideoDataCollator(self.label2id)
        )

        # Training
        if training_args.do_train:
            trainer.train(
                model_path=self.args["model_name_or_path"] if os.path.isdir(self.args["model_name_or_path"]) else None
            )
            trainer.save_model()
