import os
import torch
from tqdm import tqdm
from torchaudio.transforms import MelSpectrogram
from src.common.inferencer import BaseInferencer


class Inferencer(BaseInferencer):
    def __init__(self, config, dataloader, device, checkpoint_path, output_dir):
        super().__init__(config, device, checkpoint_path, output_dir)

        n_fft = config["acoustic"]["n_fft"]
        hop_length = config["acoustic"]["hop_length"]
        win_length = config["acoustic"]["win_length"]
        center = config["acoustic"]["center"]
        n_mel = config["acoustic"]["n_mel"]
        self.sr = config.data.sr

        self.dataloader = dataloader
        self.mel_spectrogram = MelSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=center,
            n_mels=n_mel,
        )
        self.mel_spectrogram.to(self.device)

    @torch.no_grad()
    def mel(self, noisy):

        out_len = int(noisy.shape[-1] // (self.sr / 100))

        noisy_spec = self.mel_spectrogram(noisy)

        pred_scores = self.model(noisy_spec.unsqueeze(1))[0, :out_len]

        return pred_scores.cpu().numpy()

    @torch.no_grad()
    def __call__(self):
        # inference_type = self.inference_config["type"]
        # assert inference_type in dir(
        #     self
        # ), f"Not implemented Inferencer type: {inference_type}"

        # inference_args = self.inference_config["args"]

        with open(os.path.join(self.scores_dir, "eer.txt"), "w") as f:
            for noisy, name in tqdm(self.dataloader, desc="Inference"):
                assert (
                    len(name) == 1
                ), "The batch size of inference stage must be 1."
                name = name[0]
                scores = self.mel(noisy.to(self.device))
                line = name + ", [" + ",".join([str(s) for s in scores]) + "]\n"
                f.write(line)

        with open(os.path.join(self.scores_dir, "thresholds.txt"), "w") as f:
            for key, val in self.thresholds.items():
                f.write(f"{key} = {val}\n")

