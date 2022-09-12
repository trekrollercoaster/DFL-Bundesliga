from src.core.VideoMAE.feature_extraction_videomae import VideoMAEFeatureExtractor
from tqdm.autonotebook import trange
from src import BASE_PATH, CONFIG
from decord import VideoReader
from decord import cpu
import numpy as np
import json
import os

pu_func = cpu


class ClipFrame:
    def __init__(self, batch_size=16, sample_num_per_sec=4):
        self.batch_size = batch_size
        self.sample_num_per_sec = sample_num_per_sec
        self.features_save_path = os.path.join(BASE_PATH, "data/train_source/features")
        self.labels_save_path = os.path.join(BASE_PATH, "data/train_source/labels")
        with open(os.path.join(BASE_PATH, "data/video_info.json"), "r", encoding="utf-8") as f:
            self.video_info = json.load(f)
        self.video_path = os.path.join(CONFIG["data_path"], "train")
        self.feature_extractor = VideoMAEFeatureExtractor.from_pretrained(CONFIG["backbone"])

    @staticmethod
    def _tag_time(video_info, time_sequence):
        tag_data = []
        all_info_time = [x[0] for x in video_info]
        for time in time_sequence:
            found = None
            if min(all_info_time) <= time <= max(all_info_time):
                for index in range(len(video_info) - 1):
                    info = video_info[index]
                    next_info = video_info[index + 1]
                    if info[0] <= time <= next_info[0]:
                        anchor = np.argmin([abs(time - info[0]), abs(time - next_info[0])])
                        found = [info, next_info][anchor][1]
                        break
            tag = found if found else "O"
            tag_data.append(tag)
        return tag_data

    def _gen_frame_batch(self, video_name, video_info, vr):
        fps = vr.get_avg_fps()
        print('Frame rates:', fps)
        all_event_time = [x[0] for x in video_info]
        min_time2frame = int((int(all_event_time[0]) - 1) * fps)
        max_time2frame = int((int(all_event_time[-1]) + 3) * fps)
        sample_num = int(self.batch_size / self.sample_num_per_sec * fps)
        i = 0
        for index in trange(min_time2frame, max_time2frame, sample_num, desc="Sampling"):  # sample 4 frame each seconds
            feature_save_path = os.path.join(self.features_save_path, f"{video_name}--{i}.npy")
            label_save_path = os.path.join(self.labels_save_path, f"{video_name}--{i}.json")
            if os.path.exists(feature_save_path) and os.path.exists(label_save_path):
                continue
            batch_indies = [x for x in range(index, index + sample_num, int(fps / self.sample_num_per_sec) + 1)]
            batch_indies.extend(list(range(index, index + sample_num))[-max((0, self.batch_size - len(batch_indies))):])
            times = [x / fps for x in batch_indies]
            frames = vr.get_batch(batch_indies).asnumpy()
            video = [frames[i] for i in range(frames.shape[0])]
            frames_feature = self.feature_extractor(video, return_tensors="np")
            labels = self._tag_time(video_info, times)
            frame_label = {"times": times, "labels": labels}
            np.save(feature_save_path, frames_feature["pixel_values"][0], allow_pickle=True)
            with open(label_save_path, "w", encoding="utf-8") as f:
                json.dump(frame_label, f, indent=4, ensure_ascii=False)
            i += 1

    def clip(self):
        video_path = {}
        for root, dir_list, file_list in os.walk(self.video_path):
            for file_name in file_list:
                if ".mp4" in file_name:
                    video_path.setdefault(str(file_name).replace(".mp4", ""),
                                          os.path.join(root, file_name))
        i = 0
        for video_name, path in video_path.items():
            print(f"\nStart process {video_name}")
            sample_video_info = self.video_info[video_name]
            vr = VideoReader(path, ctx=pu_func(0))
            print('video frames:', len(vr))
            self._gen_frame_batch(video_name, sample_video_info, vr)
            print(f"Finish process {video_name}, {len(video_path) - i - 1} left\n")
            i += 1
