import torch
import torch.nn as nn
import torch.nn.functional as functional

from src.model.frequency_layer import FreqFilter, get_min_seq_len
from src.model.module.causal_conv import CausalConvBlock


class FrModel(nn.Module):
    def __init__(
        self,
        window_hop=160,
        periods=[3, 11, 25, 51, 101, 201, 251],
        rnn_layers=2,
        fr_features_size=32,
        rnn_units=128,
        fc_hidden_dim=64,
    ):
        super(FrModel, self).__init__()

        self.fc_hidden_dim = fc_hidden_dim

        self.rnn_units = rnn_units

        self.frequencies = nn.ModuleList(
            [FreqFilter(p, window_hop, fr_features_size,) for p in periods]
        )

        self.window_hop = window_hop
        self.periods = periods
        hidden_dim = len(periods) * fr_features_size

        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=self.rnn_units,
            num_layers=rnn_layers,
            dropout=0.0,
            batch_first=True,
        )

        self.fc = nn.Linear(self.rnn_units, self.fc_hidden_dim)
        self.activation = nn.GELU()
        self.classification = nn.Linear(self.fc_hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, signal):
        assert signal.dim() == 2
        seq_len = signal.shape[1]
        min_sequence_len = get_min_seq_len(
            seq_len, self.periods, self.window_hop
        )

        frequencies = []
        signal = signal.unsqueeze(1)
        for fr in self.frequencies:
            frequencies.append(fr(signal, seq_len, min_sequence_len))
        features = torch.cat(frequencies, 1).transpose(1, 2)

        # print("features", features.shape)
        # # Pad look ahead

        # x = features.transpose(1, 2)
        x, (h, c) = self.rnn(features)
        x = self.fc(x)
        x = self.activation(x)
        x = self.classification(x)
        x = self.sigmoid(x)

        return x
