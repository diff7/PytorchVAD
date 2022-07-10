import torch
import torch.nn as nn
import torch.nn.functional as functional

from src.model.module.causal_conv import CausalConvBlock


class CrnVad(nn.Module):
    def __init__(
        self,
        rnn_layers=2,
        rnn_units=128,
        kernel_num=[1, 16, 32, 64, 128, 256],
        fc_hidden_dim=64,
        fft_len=160,
        look_ahead=2,
        spec_size=128,
    ):
        super(CrnVad, self).__init__()

        self.fc_hidden_dim = fc_hidden_dim

        self.kernel_num = kernel_num
        self.rnn_units = rnn_units
        self.fft_len = fft_len
        self.look_ahead = look_ahead

        self.encoder = nn.ModuleList()
        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                CausalConvBlock(self.kernel_num[idx], self.kernel_num[idx + 1])
            )

        hidden_dim = spec_size // (2 ** (len(self.kernel_num) - 1))

        self.rnn = nn.LSTM(
            input_size=hidden_dim * self.kernel_num[-1],
            hidden_size=self.rnn_units,
            num_layers=rnn_layers,
            dropout=0.0,
            batch_first=True,
        )

        self.fc = nn.Linear(self.rnn_units, self.fc_hidden_dim)
        self.activation = nn.GELU()
        self.classification = nn.Linear(self.fc_hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
            Args:
                x: [B, 1, F, T]

            Returns:
                [B, T]
            """
        assert x.dim() == 4
        # # Pad look ahead
        #  print(f"Input {x.shape}")
        x = functional.pad(x, [0, self.look_ahead])
        #   print(f"after pad {x.shape}")
        batch_size, n_channels, n_freqs, n_frames = x.size()

        x_mu = torch.mean(x, dim=(1, 2, 3)).reshape(batch_size, 1, 1, 1)
        x = x / (x_mu + 1e-10)

        for block in self.encoder:
            x = block(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, n_frames, -1).contiguous()

        #    print(f"Before RNN {x.shape}")

        x, (h, c) = self.rnn(x)
        x = self.fc(x)
        x = self.activation(x)
        x = self.classification(x)
        x = self.sigmoid(x)

        # print(f"After RNN {x.shape}")
        # print(f"After RNN look ahead {x[:, self.look_ahead :, 0]}")

        return x[:, self.look_ahead :, 0]


if __name__ == "__main__":

    inp = torch.rand((16, 1, 160, 128), device="cuda:0")

    model = CrnVad()
    model.to(0)

    o = model(inp)

    print(o)


# acoustic:
#   n_fft: 320
#   win_length: 320
#   hop_length: 160
#   center: true
#   n_mel: 80

# Input before MEL torch.Size([32, 153360])
# Input torch.Size([32, 1, 80, 959])
# after pad torch.Size([32, 1, 80, 961])
# Before RNN torch.Size([32, 961, 1280])
# After RNN torch.Size([32, 961, 1])
# After RNN look ahead torch.Size([32, 959])


# Input before MEL torch.Size([32, 214400])
# Input torch.Size([32, 1, 80, 1341])
# after pad torch.Size([32, 1, 80, 1343])
# Before RNN torch.Size([32, 1343, 1280])
# After RNN torch.Size([32, 1343, 1])
# After RNN look ahead torch.Size([32, 1341])


# Input before MEL torch.Size([32, 290720])
# Input torch.Size([32, 1, 80, 1818])
# after pad torch.Size([32, 1, 80, 1820])
# Before RNN torch.Size([32, 1820, 1280])
# After RNN torch.Size([32, 1820, 1])
# After RNN look ahead torch.Size([32, 1818])
