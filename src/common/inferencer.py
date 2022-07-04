from functools import partial
from pathlib import Path

import librosa
import torch
from src.util.acoustic_utils import stft, istft
from src.util.utils import prepare_empty_dir
from src.model.vad_model import CrnVad


class BaseInferencer:
    def __init__(self, config, device, checkpoint_path, output_dir):
        checkpoint_path = Path(checkpoint_path).expanduser().absolute()
        root_dir = Path(output_dir).expanduser().absolute()
        self.device = device
        # self.device = torch.device("cpu")

        self.model, epoch, thresholds = self._load_model(
            config["model"], checkpoint_path, self.device
        )
        self.thresholds = thresholds

        self.scores_dir = f"{root_dir}/vad_{str(epoch).zfill(4)}"

        prepare_empty_dir([self.scores_dir])

        self.acoustic_config = config["acoustic"]
        n_fft = self.acoustic_config["n_fft"]
        hop_length = self.acoustic_config["hop_length"]
        win_length = self.acoustic_config["win_length"]

        self.stft = partial(
            stft,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            device=self.device,
        )
        self.istft = partial(
            istft,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            device=self.device,
        )
        self.librosa_stft = partial(
            librosa.stft,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
        self.librosa_istft = partial(
            librosa.istft, hop_length=hop_length, win_length=win_length
        )

    @staticmethod
    def _load_model(model_config, checkpoint_path, device):

        model = CrnVad(
            rnn_layers=model_config.rnn_layers,
            rnn_units=model_config.rnn_units,
            kernel_num=model_config.kernel_num,
            fc_hidden_dim=model_config.fc_hidden_dim,
            fft_len=model_config.fft_len,
            look_ahead=model_config.look_ahead,
            use_offline_norm=model_config.use_offline_norm,
            spec_size=model_config.spec_size,
        )

        model_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_static_dict = model_checkpoint["model"]
        epoch = model_checkpoint["epoch"]
        print(
            f"The model breakpoint in tar format is currently being processed, and its epoch is：{epoch}."
        )

        # new_state_dict = OrderedDict()
        # for k, v in model_static_dict.items():
        #     name = k[7:]  # remove `module.`
        #     new_state_dict[name] = v

        # load params
        model.load_state_dict(model_static_dict)
        model.to(device)
        model.eval()
        return model, model_checkpoint["epoch"], model_checkpoint["thresholds"]

    def inference(self):
        raise NotImplementedError
