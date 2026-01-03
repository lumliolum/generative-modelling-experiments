import argparse

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # data params
    parser.add_argument(
        "--num-categories", type=int, default=16, help="number of categories for 1d data"
    )

    # model params
    parser.add_argument(
        "--hidden-dim", type=int, default=100, help="hidden dimension of the model"
    )

    # training params
    parser.add_argument(
        "--batch-size", type=int, default=32, help="batch size to be used for training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate of adam optimizer"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to run"
    )

    # inference params
    parser.add_argument(
        "--num-samples-to-generate", type=int, default=1000, help="Number of samples to generate using MH algorithm"
    )
    parser.add_argument(
        "--mh-iters", type=int, default=100, help="Number of MH iterations to run to generate"
    )

    # misc params
    parser.add_argument(
        "--seed", type=int, default=42, help="seed value for the run. If not provided default value of 42 will be used."
    )
    parser.add_argument(
        "--device", type=str, help="device to use for training. If you want to specific device, can give cuda:<device-index>."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="path to save the outputs."
    )

    return parser
