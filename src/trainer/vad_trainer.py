import warnings

import torch
from tqdm import tqdm
from torchaudio.transforms import MelSpectrogram

from src.common.trainer import BaseTrainer
from src.silero.silero_inferencer import SileroInferencer


class Trainer(BaseTrainer):
    def __init__(
        self,
        device,
        config,
        resume: bool,
        model,
        use_mel: bool,
        loss_function,
        optimizer,
        scheduler,
        train_dataloader,
        validation_dataloader,
        use_silero=True,
    ):
        super(Trainer, self).__init__(
            device, config, resume, model, loss_function, optimizer, scheduler,
        )

        n_fft = self.acoustic_config["n_fft"]
        hop_length = self.acoustic_config["hop_length"]
        win_length = self.acoustic_config["win_length"]
        center = self.acoustic_config["center"]
        n_mel = self.acoustic_config["n_mel"]
        self.sr = config.data.sr
        self.hop_length = hop_length
        self.use_silero = use_silero

        self.train_dataloader = train_dataloader
        self.valid_dataloader = validation_dataloader

        self.use_mel = use_mel
        if use_mel:
            self.mel_spectrogram = MelSpectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                center=center,
                n_mels=n_mel,
            )

    def _train_epoch(self, epoch):
        loss_total = 0.0

        i = 0
        desc = f"Training {self.device}"
        with tqdm(
            self.train_dataloader, desc=desc, total=len(self.train_dataloader)
        ) as pgbr:
            for noisy, labels, mask in pgbr:
                self.optimizer.zero_grad()
                noisy = noisy.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)

            if self.use_mel:
                self.mel_spectrogram = self.mel_spectrogram.to(self.device)
                noisy = self.mel_spectrogram(noisy).unsqueeze(1)

            pred_scores = self.model(noisy)
            pred_scores = pred_scores[:, : labels.size(-1)]
            loss = self.loss_function(pred_scores, labels, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad_norm_value
            )
            self.optimizer.step()

            loss_total += loss.item()
            pgbr.desc = desc + " loss = {:5.3f}".format(loss.item())

            self.writer.add_scalar(
                f"Loss/Train", loss_total / len(self.train_dataloader), epoch,
            )

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        visualization_metrics = self.metrics

        loss_total = 0.0
        pred_scores_list = []
        labels_list = []
        preds_silero_list = []
        silero_labels_list = []

        if self.use_silero:
            self.silero = SileroInferencer(
                self.sr,
                silero_window_size=512,
                actual_window_size=self.hop_length,
                device=self.device,
            )

        for i, (noisy, labels, mask) in tqdm(
            enumerate(self.valid_dataloader),
            total=len(self.valid_dataloader),
            desc="Validation",
        ):
            assert (
                noisy.shape[0] == 1
            ), "The batch size of validation stage must be one."

            if self.use_silero:
                preds_silero = self.silero(noisy.squeeze(0).to(self.device))
            noisy = noisy.to(self.device)
            labels = labels.to(self.device)

            if self.use_mel:
                self.mel_spectrogram = self.mel_spectrogram.to(self.device)
                noisy = self.mel_spectrogram(noisy).unsqueeze(1)

            pred_scores = self.model(noisy)
            pred_scores = pred_scores[:, : labels.size(-1)]

            loss = self.loss_function(pred_scores, labels)

            loss_total += loss

            if self.use_silero:
                # use only half of the sample due to accumalting error since window hops are different
                silero_half_size = labels.shape[1] // 2
                preds_silero_list.append(
                    preds_silero[:, :silero_half_size].cpu()
                )
                silero_labels_list.append(
                    labels.to("cpu")[:, :silero_half_size]
                )

            pred_scores_list.append(pred_scores.to("cpu"))
            labels_list.append(labels.to("cpu"))

        self.writer.add_scalar(
            f"Loss/Validation_Total",
            loss_total / len(self.valid_dataloader),
            epoch,
        )

        # print("silero preds", preds_silero[:, :silero_half_size].cpu().shape)
        # print("orig preds", pred_scores[:, :silero_half_size].cpu().shape)
        # print("silero labels", torch.cat(silero_labels_list, 1).shape)
        # print("silero preds", torch.cat(preds_silero_list, 1).shape)
        # print("orig preds", torch.cat(pred_scores_list, 1).shape)
        # print("orig labels", torch.cat(labels_list, 1).shape)

        if self.use_silero:
            self.metrics_visualization(
                torch.cat(silero_labels_list, 1),
                torch.cat(preds_silero_list, 1),
                visualization_metrics,
                epoch,
                prefix="Silero",
            )

        validation_score = self.metrics_visualization(
            torch.cat(labels_list, 1),
            torch.cat(pred_scores_list, 1),
            visualization_metrics,
            epoch,
        )

        self.scheduler.step(validation_score)

        del pred_scores_list
        del labels_list
        del self.silero

        return validation_score
