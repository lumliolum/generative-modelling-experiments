import argparse


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--timesteps", type=int, default=1000, help="Number of timesteps to train for diffusion model. If not provided default value of 1000 steps will be used."
    )
    parser.add_argument(
        "--beta-start", type=float, required=True, help="noise schedule beta0"
    )
    parser.add_argument(
        "--beta-end", type=float, required=True, help="noise schedule betaT"
    )
    parser.add_argument(
        "--variance-type", type=str, default="fixed_small", help="type of variance to be used for sampling. Can be either fixed_small or fixed_large."
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch size to be used"
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=256, help="evaluation batch size to be used"
    )
    parser.add_argument(
        "--lr", type=float, required=True, help="learning rate of adam optimizer"
    )
    parser.add_argument(
        "--epochs", type=int, required=True, help="number of epochs to run"
    )
    parser.add_argument(
        "--eval-epochs", type=int, required=True, help="Run evaluation after every eval-epochs"
    )
    parser.add_argument(
        "--do-grad-clip", action="store_true", help="whether to do gradient clipping or not."
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="norm of gradient that will be used for clipping."
    )
    parser.add_argument(
        "--do-ema", action="store_true", help="whether to do EMA or not."
    )
    parser.add_argument(
        "--ema-decay", type=float, default=0.999, help="decay to be used for EMA."
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="seed value for the run. If not provided default value of 42 will be used."
    )
    parser.add_argument(
        "--device", type=str, default="cuda: 0", help="device to use for training. If you want to specific device, can give cuda:<device-index>."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="path to save the outputs."
    )

    return parser
