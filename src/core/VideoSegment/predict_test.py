from src.core.VideoSegment.load_dataset import LoadTestDataset
from src.core.VideoSegment.inference import Inference
from tqdm.autonotebook import trange
from src import BASE_PATH
import pandas as pd
import numpy as np
import os


class PredictTest:
    def __init__(self, batch_size=1):
        self.inference = Inference()
        self.id2label = self.inference.id2label
        self.batch_size = batch_size
        self._init_dataset()
        self.softmax = lambda x: (np.exp(x) / np.exp(x).sum())

    def _init_dataset(self):
        self.dataset_util = LoadTestDataset()
        dataset = self.dataset_util.load_dataset()
        self.label2id = self.dataset_util.label2id
        self.test_data = dataset["test"]

    def predict(self):
        result = []
        for index in trange(0, self.test_data.num_rows, self.batch_size, desc="Predicting"):
            batch_data = self.test_data.select(range(index, index + self.batch_size))
            batch_data = self.dataset_util.convert_to_features(batch_data)
            pixel_values = batch_data["pixel_values"]
            times = batch_data["labels"]
            names = batch_data["names"]
            predicts, logits = self.inference.predict(pixel_values)
            for name, time, predict, logit in zip(names, times, predicts, logits):
                predict_label = [self.id2label[i] for i in predict]
                logit = self.softmax(logit)
                score = [logit[i][x] for i, x in enumerate(predict)]
                video_id = str(name).split("--")[0]
                for t, e, s in zip(time, predict_label, score):
                    result.append({
                        "video_id": video_id,
                        "time": t,
                        "event": str(e).split("-")[0],
                        "score": round(s, 2)
                    })
        result_df = pd.json_normalize(result)
        result_df = result_df.sort_values(["video_id", "time"])
        result_df.to_csv(os.path.join(BASE_PATH, "data/sample_submission.csv"), encoding="utf-8", index=False)
