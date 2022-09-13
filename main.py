import argparse


def main():
    parser = argparse.ArgumentParser(description='VideSegment Preprocess')
    parser.add_argument('--do_part', type=str, default='test',
                        help='data_part: train or test')

    args = parser.parse_args()

    if args.do_part == "train":
        from src.core.VideoSegment.train import TrainModel
        TrainModel().train()
    else:
        from src.core.VideoSegment.predict_test import PredictTest
        PredictTest().predict()


if __name__ == '__main__':
    main()
