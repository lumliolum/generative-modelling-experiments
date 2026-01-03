import argparse


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # dataset params
    parser.add_argument(
        "--dataset-name", type=str, default="mnist", help="name of the dataset to use. Currently only 'mnist' and 'cifar10' is supported."
    )

    # training params
    parser.add_argument(
        "--training-objective", type=str, default="denoiser", help="Specifies the training objective - either 'denoiser' (model learns to denoise the input) or 'score' (model learns to predict the score function)."
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.2, help="fraction of training data to be used for validation"
    )
    parser.add_argument(
        "--noise-var", type=float, default=1.0, help="variance of noise to be added"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch size to be used"
    )
    parser.add_argument(
        "--lr", type=float, required=True, help="learning rate of adamw optimizer"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-2, help="weight decay to be used in adamw optimizer"
    )
    parser.add_argument(
        "--epochs", type=int, required=True, help="number of epochs to run"
    )
    parser.add_argument(
        "--do-grad-clip", action="store_true", help="whether to do gradient clipping or not."
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="norm of gradient that will be used for clipping."
    )
    parser.add_argument(
        "--do-ema", action="store_true", help="whether to do gradient clipping or not."
    )
    parser.add_argument(
        "--ema-decay", type=float, default=0.999, help="decay to be used for EMA."
    )

    # model params
    parser.add_argument(
        "--model-dim", type=int, default=128, help="base dimension to be used in UNet model."
    )
    parser.add_argument(
        "--norm-groups", type=int, default=32, help="norm groups to be used in group normalization."
    )
    parser.add_argument(
        "--dropout-prob", type=float, default=0.1, help="dropout probability to be used in the model."
    )

    # langevin sampling params
    parser.add_argument(
        "--langevin-steps", type=int, default=10000, help="number of langevin sampling steps to run"
    )
    parser.add_argument(
        "--langevin-save-every", type=int, default=100, help="save samples every n steps during langevin sampling"
    )

    # misc params
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
