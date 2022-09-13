from src.core.VideoMAE.feature_extraction_videomae import VideoMAEFeatureExtractor
from src.core.PreProcess.clip_frame import ClipFrame
from tqdm.autonotebook import trange
from src import BASE_PATH, CONFIG
from decord import VideoReader
from decord import cpu
import numpy as np
import json
import os

pu_func = cpu


class ClipTestFrame(ClipFrame):
    def __init__(self, batch_size=16, sample_num_per_sec=4):
        super(ClipTestFrame, self).__init__(batch_size, sample_num_per_sec)
        self.features_save_path = os.path.join(BASE_PATH, "data/test_source/features")
        self.labels_save_path = os.path.join(BASE_PATH, "data/test_source/labels")
        self.video_path = os.path.join(CONFIG["data_path"], "test")

    def _gen_test_frame_batch(self, video_name, vr):
        fps = vr.get_avg_fps()
        print('Frame rates:', fps)
        min_time2frame = 0
        max_time2frame = len(vr)
        sample_num = int(self.batch_size / self.sample_num_per_sec * fps)
        i = 0
        for index in trange(min_time2frame, max_time2frame, sample_num, desc="Sampling"):  # sample 4 frame each seconds
            feature_save_path = os.path.join(self.features_save_path, f"{video_name}--{i}.npz")
            label_save_path = os.path.join(self.labels_save_path, f"{video_name}--{i}.json")
            if os.path.exists(feature_save_path) and os.path.exists(label_save_path):
                i += 1
                continue
            if index + sample_num > max_time2frame:
                index_range = (max_time2frame - sample_num, max_time2frame)
            else:
                index_range = (index, index + sample_num)
            batch_indies = [x for x in range(index_range[0], index_range[1], int(fps / self.sample_num_per_sec) + 1)]
            batch_indies.extend(list(range(
                index_range[0], index_range[1]
            ))[-max((0, self.batch_size - len(batch_indies))):])
            times = [x / fps for x in batch_indies]
            frames = vr.get_batch(batch_indies).asnumpy()
            video = [frames[i] for i in range(frames.shape[0])]
            frames_feature = self.feature_extractor(video, return_tensors="np")
            frame_label = {"times": times}
            np.savez_compressed(feature_save_path, frames_feature["pixel_values"][0], allow_pickle=True)
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
            vr = VideoReader(path, ctx=pu_func(0))
            print('video frames:', len(vr))
            self._gen_test_frame_batch(video_name, vr)
            print(f"Finish process {video_name}, {len(video_path) - i - 1} left\n")
            i += 1
