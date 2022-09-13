from src.core.PreProcess.data_process import analysis_ground_truth
from src.core.PreProcess.clip_test_frame import ClipTestFrame
from src.core.PreProcess.clip_frame import ClipFrame
import argparse


def main():
    parser = argparse.ArgumentParser(description='VideSegment Preprocess')
    parser.add_argument('--data_part', type=str, default='test',
                        help='data_part: train or test')

    args = parser.parse_args()

    if args.data_part == "train":
        analysis_ground_truth()
        ClipFrame().clip()
    else:
        ClipTestFrame().clip()


if __name__ == '__main__':
    main()
