import argparse
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf as omg
from src.inferencer.VAD_Inferencer import Inferencer
from src.dataset.inference_dataset import get_data_loader

cudnn.benchmark = True


def main(config, checkpoint_path, output_dir, device):

    dataloder = get_data_loader(path=config.data.test_files_path)

    inferencer = Inferencer(
        config, dataloder, device, checkpoint_path, output_dir
    )
    inferencer()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Inference")

    CFG_PATH = "./config/config.yaml"
    cfg = omg.load(CFG_PATH)

    parser.add_argument(
        "-dir",
        "--output_dir",
        type=str,
        required=True,
        help="The path to save the enhanced speech.",
    )

    parser.add_argument(
        "-g", "--gpu", type=int, default=1, help="GPU number",
    )

    parser.add_argument(
        "-chk", "--model_path", type=str, help="model_checkpoint_path",
    )

    args = parser.parse_args()
    main(cfg, args.model_path, args.output_dir, args.gpu)
