import random
import struct

import torch
import webrtcvad
import numpy as np
from glob import glob

from src.common.dataset import BaseDataset
from src.util.acoustic_utils import (
    norm_amplitude,
    tailor_dB_FS,
    is_clipped,
    load_wav,
    read_wave,
    frame_generator,
)

from torch.nn.utils.rnn import pad_sequence


def collate_fn(samples, batch_first=True):
    audio, labels = zip(*samples)
    audio_pad = pad_sequence(audio, batch_first=batch_first)
    labels_pad = pad_sequence(labels, batch_first=batch_first)
    mask = [
        [int(j < labels[i].shape[-1]) for j in range(len(labels_pad[i]))]
        for i in range(len(samples))
    ]
    return audio_pad, labels_pad, torch.tensor(mask).int()


class VADSet(BaseDataset):
    def __init__(
        self,
        clean_dataset,
        noise_dataset,
        snr_range,
        silence_length,
        target_dB_FS,
        target_dB_FS_floating_value,
        sr,
        num_workers,
        vad_mode,
        data_bit,
        noise_proportion,
        validation=False,
    ):
        """
        Dynamic mixing for training
        """
        super().__init__()
        # acoustic args
        self.sr = sr

        # parallel args
        self.num_workers = num_workers

        clean_dataset_list = glob(f"{clean_dataset}/**/*.wav", recursive=True)
        val_size = int(len(clean_dataset_list) * 0.2)

        if validation:
            clean_dataset_list = clean_dataset_list[val_size:]
        else:
            clean_dataset_list = clean_dataset_list[:val_size]

        noise_dataset_list = glob(f"{noise_dataset}/**/*.wav", recursive=True)

        self.clean_dataset_list = clean_dataset_list
        self.noise_dataset_list = noise_dataset_list

        snr_list = self._parse_snr_range(snr_range)
        self.snr_list = snr_list
        self.silence_length = silence_length
        self.target_dB_FS = target_dB_FS
        self.target_dB_FS_floating_value = target_dB_FS_floating_value

        self.vad_mode = vad_mode
        self.vad = webrtcvad.Vad(vad_mode)
        self.scale = 2 ** (data_bit - 1)

        self.noise_proportion = noise_proportion

        self.length = len(self.clean_dataset_list)

    def __len__(self):
        return self.length

    @staticmethod
    def _random_select_from(dataset_list):
        return random.choice(dataset_list)

    def get_vad_lables(self, clean_path):
        audio_b, sr, n_frames = read_wave(clean_path)
        lables = np.array(
            [
                self.vad.is_speech(frame.bytes, sr)
                for frame in frame_generator(10, audio_b, sr)
            ]
        )
        audio = np.array(struct.unpack("{n}h".format(n=n_frames), audio_b))
        audio, _ = norm_amplitude(audio, self.scale)
        return audio, lables

    def _select_noise_y(self, target_length):
        noise_y = np.zeros(0, dtype=np.float32)
        silence = np.zeros(int(self.sr * self.silence_length), dtype=np.float32)
        remaining_length = target_length

        while remaining_length > 0:
            noise_file = self._random_select_from(self.noise_dataset_list)
            noise_new_added = load_wav(noise_file, sr=self.sr)
            noise_y = np.append(noise_y, noise_new_added)
            remaining_length -= len(noise_new_added)

            if remaining_length > 0:
                silence_len = min(remaining_length, len(silence))
                noise_y = np.append(noise_y, silence[:silence_len])
                remaining_length -= silence_len

        if len(noise_y) > target_length:
            idx_start = np.random.randint(len(noise_y) - target_length)
            noise_y = noise_y[idx_start : idx_start + target_length]

        return noise_y

    @staticmethod
    def snr_mix(
        clean_y,
        noise_y,
        snr,
        target_dB_FS,
        target_dB_FS_floating_value,
        eps=1e-6,
    ):
        """
        mix clean signal, noise and add reverberation with given SNR

        Args:
            clean_y: clean audio
            noise_y: noise audio
            snr (int): dB
            target_dB_FS (int):
            target_dB_FS_floating_value (int):
            eps: eps

        Returns:
            (noisy_y，clean_y)
        """

        clean_y, _ = norm_amplitude(clean_y)
        clean_y, _, _ = tailor_dB_FS(clean_y, target_dB_FS)
        clean_rms = (clean_y ** 2).mean() ** 0.5

        noise_y, _ = norm_amplitude(noise_y)
        noise_y, _, _ = tailor_dB_FS(noise_y, target_dB_FS)
        noise_rms = (noise_y ** 2).mean() ** 0.5

        snr_scalar = clean_rms / (10 ** (snr / 20)) / (noise_rms + eps)
        noise_y *= snr_scalar
        noisy_y = clean_y + noise_y

        # Randomly select RMS value of dBFS between -15 dBFS and -35 dBFS and normalize noisy speech with that value
        noisy_target_dB_FS = np.random.randint(
            target_dB_FS - target_dB_FS_floating_value,
            target_dB_FS + target_dB_FS_floating_value,
        )

        noisy_y, _, noisy_scalar = tailor_dB_FS(noisy_y, noisy_target_dB_FS)

        if is_clipped(noisy_y):
            noisy_y_scalar = np.max(np.abs(noisy_y)) / (0.99 - eps)
            noisy_y = noisy_y / noisy_y_scalar

        return noisy_y

    def __getitem__(self, item):
        clean_file = self.clean_dataset_list[item]
        clean_y, labels = self.get_vad_lables(clean_file)
        if bool(np.random.random(1) < self.noise_proportion):
            noise_y = self._select_noise_y(target_length=len(clean_y))
            assert len(clean_y) == len(
                noise_y
            ), f"Inequality: {len(clean_y)} {len(noise_y)}"

            snr = self._random_select_from(self.snr_list)

            noisy_y = self.snr_mix(
                clean_y=clean_y,
                noise_y=noise_y,
                snr=snr,
                target_dB_FS=self.target_dB_FS,
                target_dB_FS_floating_value=self.target_dB_FS_floating_value,
            )
        else:
            noisy_y = clean_y

        #print(noisy_y.shape, len(labels))
        noisy_y = torch.tensor(noisy_y)
        labels = torch.tensor(labels)

        return noisy_y.float(), labels.float()
