import os
import time
from copy import deepcopy
from functools import reduce

import torch


def load_checkpoint(checkpoint_path, device):
    _, ext = os.path.splitext(os.path.basename(checkpoint_path))
    assert ext in (
        ".pth",
        ".tar",
    ), "Only support ext and tar extensions of l1 checkpoint."
    model_checkpoint = torch.load(
        os.path.abspath(os.path.expanduser(checkpoint_path)),
        map_location=device,
    )

    if ext == ".pth":
        print(f"Loading {checkpoint_path}.")
        return model_checkpoint
    else:  # load tar
        print(
            f"Loading {checkpoint_path}, epoch = {model_checkpoint['epoch']}."
        )
        return model_checkpoint["l1"]


def prepare_empty_dir(dirs, resume=False):
    """
    if resume the experiment, assert the dirs exist. If not the resume experiment, set up new dirs.

    Args:
        dirs (list): directors list
        resume (bool): whether to resume experiment, default is False
    """
    for dir_path in dirs:
        if resume:
            assert os.path.exists(
                dir_path
            ), "In resume mode, you must be have an old experiment dir."
        else:
            os.makedirs(dir_path, exist_ok=True)


def check_nan(tensor, key=""):
    if torch.sum(torch.isnan(tensor)) > 0:
        print(f"Found NaN in {key}")


class ExecutionTime:
    """
    Count execution time.

    Examples:
        timer = ExecutionTime()
        ...
        print(f"Finished in {timer.duration()} seconds.")
    """

    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return int(time.time() - self.start_time)


def print_tensor_info(tensor, flag="Tensor"):
    def floor_tensor(float_tensor):
        return int(float(float_tensor) * 1000) / 1000

    print(
        f"{flag}\n"
        f"\t"
        f"max: {floor_tensor(torch.max(tensor))}, min: {float(torch.min(tensor))}, "
        f"mean: {floor_tensor(torch.mean(tensor))}, std: {floor_tensor(torch.std(tensor))}"
    )


def set_requires_grad(nets, requires_grad=False):
    """
    Args:
        nets: list of networks
        requires_grad
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def merge_config(*config_dicts):
    """
    Deep merge configuration dicts.

    Args:
        *config_dicts: any number of configuration dicts.

    Notes:
        1. The values of item in the later configuration dict(s) will update the ones in the former dict(s).
        2. The key in the later dict must be exist in the former dict. It means that the first dict must consists of all keys.

    Examples:
        a = [
            "a": 1,
            "b": 2,
            "c": {
                "d": 1
            }
        ]
        b = [
            "a": 2,
            "b": 2,
            "c": {
                "e": 1
            }
        ]
        c = merge_config(a, b)
        c = [
            "a": 2,
            "b": 2,
            "c": {
                "d": 1,
                "e": 1
            }
        ]

    Returns:
        New deep-copied configuration dict.
    """

    def merge(older_dict, newer_dict):
        for new_key in newer_dict:
            if new_key not in older_dict:
                # Checks items in custom config must be within common config
                raise KeyError(
                    f"Key {new_key} is not exist in the common config."
                )

            if isinstance(older_dict[new_key], dict):
                older_dict[new_key] = merge(
                    older_dict[new_key], newer_dict[new_key]
                )
            else:
                older_dict[new_key] = deepcopy(newer_dict[new_key])

        return older_dict

    return reduce(merge, config_dicts[1:], deepcopy(config_dicts[0]))


def prepare_device(n_gpu: int, keep_reproducibility=False):
    """
    Choose to use CPU or GPU depend on the value of "n_gpu".

    Args:
        n_gpu(int): the number of GPUs used in the experiment. if n_gpu == 0, use CPU; if n_gpu >= 1, use GPU.
        keep_reproducibility (bool): if we need to consider the repeatability of experiment, set keep_reproducibility to True.

    See Also
        Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
    """
    if n_gpu == 0:
        print("Using CPU in the experiment.")
        device = torch.device("cpu")
    else:
        # possibly at the cost of reduced performance
        if keep_reproducibility:
            print("Using CuDNN deterministic mode in the experiment.")
            torch.backends.cudnn.benchmark = False  # ensures that CUDA selects the same convolution algorithm each time
            torch.set_deterministic(
                True
            )  # configures PyTorch only to use deterministic implementation
        else:
            # causes cuDNN to benchmark multiple convolution algorithms and select the fastest
            torch.backends.cudnn.benchmark = True

        device = torch.device("cuda:0")

    return device


def expand_path(path):
    return os.path.abspath(os.path.expanduser(path))
