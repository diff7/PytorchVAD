import os
import time
from functools import partial
from pathlib import Path


import torch
import colorful
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import DetCurveDisplay

import src.util.metrics as metrics
from src.util.acoustic_utils import stft, istft
from src.util.utils import prepare_empty_dir, ExecutionTime

plt.switch_backend("agg")

from torch.utils.tensorboard import SummaryWriter


class BaseTrainer:
    def __init__(
        self,
        device,
        config,
        resume: bool,
        model,
        loss_function,
        optimizer,
        scheduler,
    ):

        self.color_tool = colorful
        self.color_tool.use_style("solarized")

        self.model = model
        self.model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.device = device

        # Supported STFT
        self.acoustic_config = config.acoustic
        n_fft = config.acoustic["n_fft"]
        hop_length = config.acoustic["hop_length"]
        win_length = config.acoustic["win_length"]
        center = config.acoustic["center"]

        self.torch_stft = partial(
            stft,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            device=self.device,
            center=center,
        )
        self.istft = partial(
            istft,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            device=self.device,
            center=center,
        )

        # Trainer.train in config
        self.epochs = config.training["epochs"]
        self.save_checkpoint_interval = config.training[
            "save_checkpoint_interval"
        ]
        assert self.save_checkpoint_interval >= 1
        self.clip_grad_norm_value = config.training["clip_grad_norm_value"]

        # Trainer.validation in config
        self.validation_interval = config.training["validation_interval"]
        self.save_max_metric_score = config.training["save_max_metric_score"]
        assert self.validation_interval >= 1

        # Trainer.visualization in config
        self.metrics = config["metrics"]

        # In the 'train.py' file, if the 'resume' item is True, we will update the following args:
        self.start_epoch = 1
        self.best_score = -np.inf if self.save_max_metric_score else np.inf
        self.save_dir = config.env.save_dir
        self.writer = SummaryWriter(
            log_dir=config.env.save_dir, max_queue=5, flush_secs=30
        )
        self.checkpoints_dir = os.path.join(self.save_dir, "checkpoints")
        self.logs_dir = os.path.join(self.save_dir, "logs")

        self.thresholds = {"eer": 0, "fpr_1": 0, "fnr_1": 0}

        if resume:
            self._resume_checkpoint()

        prepare_empty_dir([self.checkpoints_dir, self.logs_dir], resume=resume)

        print(self.color_tool.cyan("The configurations are as follows: "))
        print(self.color_tool.cyan("=" * 40))
        print(self.color_tool.cyan("=" * 40))

        self._print_networks([self.model])

    def _preload_model(self, model_path):
        """
        Preload model parameters (in "*.tar" format) at the start of experiment.

        Args:
            model_path (Path): The file path of the *.tar file
        """
        model_path = model_path.expanduser().absolute()
        assert (
            model_path.exists()
        ), f"The file {model_path} is not exist. please check path."

        map_location = {"cuda:%d" % 0: "cuda:%d" % self.device}
        model_checkpoint = torch.load(model_path, map_location=map_location)
        self.model.load_state_dict(model_checkpoint["model"], strict=False)

    def _resume_checkpoint(self):
        """
        Resume experiment from the latest checkpoint.
        """
        latest_model_path = self.checkpoints_dir / "latest_model.tar"
        assert (
            latest_model_path.exists()
        ), f"{latest_model_path} does not exist, can not load latest checkpoint."

        map_location = {"cuda:%d" % 0: "cuda:%d" % self.device}
        checkpoint = torch.load(latest_model_path, map_location=map_location)

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_score = checkpoint["best_score"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.model.load_state_dict(checkpoint["model"])
        self.thresholds = checkpoint["thresholds"]

        print(
            f"Model checkpoint loaded. Training will begin at {self.start_epoch} epoch."
        )

    def _save_checkpoint(self, epoch, is_best_epoch=False):
        """
        Save checkpoint to "<save_dir>/<config name>/checkpoints" directory, which consists of:
            - the epoch number
            - the best metric score in history
            - the optimizer parameters
            - the model parameters

        Args:
            is_best_epoch (bool): In current epoch, if the model get a best metric score (is_best_epoch=True),
                                the checkpoint of model will be saved as "<save_dir>/checkpoints/best_model.tar".
        """
        print(f"\t Saving {epoch} epoch model checkpoint...")

        state_dict = {
            "epoch": epoch,
            "best_score": self.best_score,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "model": self.model.state_dict(),
            "thresholds": self.thresholds,
        }

        # "latest_model.tar"
        # Contains all checkpoint information, including the optimizer parameters, the model parameters, etc.
        # New checkpoint will overwrite the older one.
        torch.save(state_dict, f"{self.checkpoints_dir}/latest_model.tar")

        # "model_{epoch_number}.tar"
        # Contains all checkpoint information, like "latest_model.tar". However, the newer information will no overwrite the older one.
        torch.save(
            state_dict,
            f"{self.checkpoints_dir}/model_{str(epoch).zfill(4)}.tar",
        )

        # If the model get a best metric score (is_best_epoch=True) in the current epoch,
        # the model checkpoint will be saved as "best_model.tar."
        # The newer best-scored checkpoint will overwrite the older one.
        if is_best_epoch:
            print(
                self.color_tool.red(
                    f"\t Found a best score in the {epoch} epoch, saving..."
                )
            )
            torch.save(state_dict, f"{self.checkpoints_dir}/best_model.tar")

    def _is_best_epoch(self, score, save_max_metric_score=True):
        """
        Check if the current model got the best metric score
        """
        if save_max_metric_score and score >= self.best_score:
            self.best_score = score
            return True
        elif not save_max_metric_score and score <= self.best_score:
            self.best_score = score
            return True
        else:
            return False

    @staticmethod
    def _print_networks(models: list):
        print(
            f"This project contains {len(models)} models, the number of the parameters is: "
        )

        params_of_all_networks = 0
        for idx, model in enumerate(models, start=1):
            params_of_network = 0
            for param in model.parameters():
                params_of_network += param.numel()

            print(f"\tNetwork {idx}: {params_of_network / 1e6} million.")
            params_of_all_networks += params_of_network

        print(
            f"The amount of parameters in the project is {params_of_all_networks / 1e6} million."
        )

    def _set_models_to_train_mode(self):
        self.model.train()

    def _set_models_to_eval_mode(self):
        self.model.eval()

    @staticmethod
    def get_thresholds(labels, scores):
        (
            eer_t,
            eer,
            fpr_1_t,
            fpr_1_fnr,
            fnr_1_t,
            fnr_1_fpr,
        ) = metrics.compute_thresholds(labels, scores)
        return eer_t, fpr_1_t, fnr_1_t, eer, fpr_1_fnr, fnr_1_fpr

    def metrics_visualization(
        self, labels, predicted, metrics_list, epoch, prefix="OUR"
    ):
        """
        Get metrics on validation dataset by paralleling.
        """
        assert "ROC_AUC" in metrics_list

        # Check if the metric is registered in "util.metrics" file.
        for i in metrics_list:
            assert (
                i in metrics.REGISTERED_METRICS.keys()
            ), f"{i} is not registered, please check 'util.metrics' file."

        fpr, tpr, thresholds = metrics.roc_curve(
            labels.reshape(-1), predicted.reshape(-1)
        )

        roc_auc_mean = 0
        for metric_name in metrics_list:
            mean_score = metrics.REGISTERED_METRICS[metric_name](fpr, tpr)

            # Add the mean value of the metric to tensorboard
            self.writer.add_scalars(
                f"Validation/{metric_name}", {prefix: mean_score}, epoch
            )

            if metric_name == "ROC_AUC":
                roc_auc_mean = mean_score

        fig, axes = plt.subplots(1, 1, figsize=(6, 6))
        display = DetCurveDisplay(
            fpr=fpr, fnr=1 - tpr, estimator_name=f"ROC_AUC = {roc_auc_mean}"
        )
        display.plot(axes)
        self.writer.add_figure(f"DetCurve", fig, epoch)

        eer_t, fpr_1_t, fnr_1_t, _, _, _ = self.get_thresholds(
            labels.reshape(-1), predicted.reshape(-1)
        )

        f1, _, _, precision, recall = metrics.get_f1(
            (predicted.reshape(-1) > eer_t).int(), labels.reshape(-1)
        )

        self.writer.add_scalars(f"Validation/F1", {prefix: f1}, epoch)
        self.writer.add_scalars(
            f"Validation/Precision", {prefix: precision}, epoch
        )
        self.writer.add_scalars(f"Validation/recall", {prefix: recall}, epoch)

        self.thresholds = {"eer": eer_t, "fpr_1": fpr_1_t, "fnr_1": fnr_1_t}

        return roc_auc_mean

    def train(self):
        print("Test validation")
        self._validation_epoch(0)
        for epoch in range(self.start_epoch, self.epochs + 1):
            print(
                self.color_tool.yellow(f"{'=' * 15} {epoch} epoch {'=' * 15}")
            )
            print("[0 seconds] Begin training...")

            timer = ExecutionTime()
            self._set_models_to_train_mode()
            self._train_epoch(epoch)

            # Only use the first GPU (process) to the validation.
            if epoch % self.validation_interval == 0:
                print(
                    f"[{timer.duration()} seconds] Training has finished, validation is in progress..."
                )

                if self.save_checkpoint_interval != 0 and (
                    epoch % self.save_checkpoint_interval == 0
                ):
                    self._save_checkpoint(epoch)

                self._set_models_to_eval_mode()
                metric_score = self._validation_epoch(epoch)

                if self.save_checkpoint_interval != 0 and (
                    epoch % self.save_checkpoint_interval == 0
                ):
                    self._save_checkpoint(epoch)

                if self._is_best_epoch(
                    metric_score,
                    save_max_metric_score=self.save_max_metric_score,
                ):
                    self._save_checkpoint(epoch, is_best_epoch=True)

                print(f"[{timer.duration()} seconds] This epoch has finished.")

