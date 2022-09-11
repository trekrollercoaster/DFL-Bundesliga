from src import CONFIG, BASE_PATH
from tqdm import tqdm
import pandas as pd
import json
import os


def norm_event_attributes(event, event_attributes):
    event_attributes = None if isinstance(event_attributes, float) else eval(event_attributes)
    normed_label = "-".join([event] + event_attributes if event_attributes else [event])
    return normed_label


def analysis_ground_truth():
    label2id = {"O": 0}
    train_data_path = os.path.join(CONFIG["data_path"], "train.csv")
    train_data = pd.read_csv(train_data_path, encoding="utf-8")
    video_info = {video_id: [] for video_id in list(set(train_data[["video_id"]].values.flatten().tolist()))}
    train_data_array = train_data.to_dict(orient="records")
    time_interval = []
    index = 0
    for row in tqdm(train_data_array, desc="parse data"):
        event = row["event"]
        label = norm_event_attributes(event, row["event_attributes"])
        if label == "start" and row["video_id"] == train_data_array[index + 1]["video_id"]:
            next_event_attributes = norm_event_attributes(train_data_array[index + 1]["event"],
                                                          train_data_array[index + 1]["event_attributes"])
            label = "-".join([event, next_event_attributes])
            time_interval.append({
                "start2action": train_data_array[index + 1]["time"] - row["time"],
                "action2end": train_data_array[index + 2]["time"] - train_data_array[index + 1]["time"],
                "start2end": train_data_array[index + 2]["time"] - row["time"]
            })
        elif label == "end" and row["video_id"] == train_data_array[index - 1]["video_id"]:
            next_event_attributes = norm_event_attributes(train_data_array[index - 1]["event"],
                                                          train_data_array[index - 1]["event_attributes"])
            label = "-".join([event, next_event_attributes])
        if label not in label2id.keys():
            label2id.setdefault(label, len(label2id))
        video_info[row["video_id"]].append([row["time"], label])
        index += 1
    time_interval_df = pd.json_normalize(time_interval)
    print(time_interval_df.describe())
    time_interval_df.to_csv(os.path.join(BASE_PATH, "data/time_interval.csv"), encoding="utf-8", index=False)
    with open(os.path.join(BASE_PATH, "data/video_info.json"), "w", encoding="utf-8") as f:
        json.dump(video_info, f, indent=4, ensure_ascii=False)
    with open(os.path.join(BASE_PATH, "data/label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=4, ensure_ascii=False)
