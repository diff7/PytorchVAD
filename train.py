import random
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

# from src.dataset.interaction_dataset import from_path

from omegaconf import OmegaConf as omg
from src.model.vad_model import CrnVad
from src.dataset.dataset_main import VADSet, collate_fn
from src.trainer.vad_trainer import Trainer
from src.model.loss import BCELossMaskOutput


def main(cfg, resume, device):
    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)
    random.seed(cfg.env.seed)

    loaders = []

    for t in [True, False]:
        if t:
            bs = cfg.training.batch_size
        else:
            bs = 1

        loaders.append(
            DataLoader(
                VADSet(
                    clean_dataset=cfg.data.clean_files_path,
                    noise_dataset=cfg.data.clean_files_path,
                    snr_range=cfg.data.processing.snr_range,
                    silence_length=cfg.data.processing.silence_length,
                    target_dB_FS=cfg.data.processing.target_dB_FS,
                    target_dB_FS_floating_value=cfg.data.processing.target_dB_FS_floating_value,
                    sr=cfg.data.sr,
                    num_workers=cfg.env.workers,
                    vad_mode=cfg.data.processing.vad_mode,
                    data_bit=cfg.data.processing.data_bit,
                    noise_proportion=cfg.data.processing.noise_proportion,
                    validation=t,
                ),
                collate_fn=collate_fn,
                batch_size=bs,
            )
        )

    train_loader, val_loader = loaders

    model = CrnVad(
        rnn_layers=cfg.model.rnn_layers,
        rnn_units=cfg.model.rnn_units,
        kernel_num=cfg.model.kernel_num,
        fc_hidden_dim=cfg.model.fc_hidden_dim,
        fft_len=cfg.model.fft_len,
        look_ahead=cfg.model.look_ahead,
        use_offline_norm=cfg.model.use_offline_norm,
        spec_size=cfg.model.spec_size,
    )

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    loss_function = BCELossMaskOutput(
        reduction=cfg.training.loss.reduction,
        silent_weight=cfg.training.loss.silent_weight,
    )

    trainer = Trainer(
        device=device,
        config=cfg,
        resume=resume,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        train_dataloader=train_loader,
        validation_dataloader=val_loader,
    )

    trainer.train()


if __name__ == "__main__":
    CFG_PATH = "./config/config.yaml"
    cfg = omg.load(CFG_PATH)

    parser = argparse.ArgumentParser(description="VAD")

    parser.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="Resume experiment from latest checkpoint.",
    )

    parser.add_argument(
        "-g", "--gpu", type=int, default=1, help="GPU number",
    )
    args = parser.parse_args()

    main(cfg, args.resume, args.gpu)
