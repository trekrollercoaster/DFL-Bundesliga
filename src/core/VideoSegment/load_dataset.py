from transformers.modeling_outputs import ModelOutput
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple
from dataclasses import dataclass
from datasets import load_dataset
from typing import Dict, List
from src import BASE_PATH
import pandas as pd
import numpy as np
import torch
import json
import os


class LoadDataset:
    def __init__(self, test_size=0.2):
        self.test_size = test_size
        self.features_path = os.path.join(BASE_PATH, "data/train_source/features")
        self.labels_path = os.path.join(BASE_PATH, "data/train_source/labels")
        self.train_data_save_path = os.path.join(BASE_PATH, "data/train_source/train.csv")
        self.valid_data_save_path = os.path.join(BASE_PATH, "data/train_source/valid.csv")
        with open(os.path.join(BASE_PATH, "data/label2id.json"), "r", encoding="utf-8") as f:
            self.label2id = json.load(f)

    @staticmethod
    def _data_collection(path, fix):
        data = {}
        for root, dir_list, file_list in os.walk(path):
            for file_name in file_list:
                if fix in file_name:
                    data.setdefault(str(file_name).replace(fix, ""), os.path.join(root, file_name))
        return data

    def _build_dataset(self):
        feature_data = self._data_collection(self.features_path, ".npz")
        label_data = self._data_collection(self.labels_path, ".json")
        assert len(set(list(feature_data.keys())) & set(list(label_data.keys()))) == len(feature_data.keys()) == len(
            label_data.keys())
        data_index = list(feature_data.keys())
        train_index, valid_index = train_test_split(data_index, test_size=self.test_size)
        train_data, valid_data = [], []
        for data_name in feature_data.keys():
            feature = feature_data[data_name]
            label = label_data[data_name]
            if data_name in train_index:
                train_data.append({
                    "feature_path": feature,
                    "label_path": label
                })
            else:
                valid_data.append({
                    "feature_path": feature,
                    "label_path": label
                })
        pd.json_normalize(train_data).to_csv(self.train_data_save_path, index=False, encoding="utf-8")
        pd.json_normalize(valid_data).to_csv(self.valid_data_save_path, index=False, encoding="utf-8")

    def load_dataset(self):
        if not os.path.exists(self.train_data_save_path) or not os.path.exists(self.valid_data_save_path):
            self._build_dataset()
        dataset = load_dataset("csv", data_files={
            "train": self.train_data_save_path,
            "valid": self.valid_data_save_path
        })
        return dataset


@dataclass
class VideoDataCollator:

    def __init__(self, label2id):
        self.label2id = label2id

    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        pixel_values, labels = [], []
        for example in batch:
            try:
                feature = torch.from_numpy(np.load(example["feature_path"], allow_pickle=True)["arr_0"])
                pixel_values.append(feature)
                with open(example["label_path"], "r", encoding="utf-8") as f:
                    label_data = json.load(f)
                label = torch.Tensor([self.label2id[x] for x in label_data["labels"]]).type(torch.long)
                labels.append(label)
            except Exception as e:
                os.remove(example["feature_path"])
                os.remove(example["label_path"])
                pixel_values.append(torch.from_numpy(np.zeros((16, 3, 244, 244))).type(torch.float))
                labels.append(torch.from_numpy(np.zeros((16,))).type(torch.long))
                print(e)

        pixel_values = torch.stack(pixel_values)
        labels = torch.stack(labels)
        return {
            'pixel_values': pixel_values,
            'labels': labels
        }


@dataclass
class FrameClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: List = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
