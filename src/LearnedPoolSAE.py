"""
    Some parts of the code are adapted from the original implementation of the paper: https://github.com/MEPP-team/SAE.
"""

import torch
import torch.nn as nn


class LearnedPoolSAE(torch.nn.Module):
    def __init__(self,
                 k: int,
                 size_latent: int = 16,
                 initial_feature_size: int = 8,
                 depth: int = 4,
                 activation_func: str = 'ELU',
                 device: str = 'cpu'
                 ) -> None:
        super(LearnedPoolSAE, self).__init__()
        self.k = k
        self.size_latent = size_latent
        self.activation_func = activation_func
        self.device = device

        encoder_features = [3] + [initial_feature_size * (2 ** i) for i in range(depth)]

        sizes_downsample = [self.k] + encoder_features[1:][::-1]
        sizes_upsample = sizes_downsample[::-1]

        sizes_convs_encode = [3] * depth
        sizes_convs_decode = sizes_convs_encode[::-1]

        encoder_linear = [encoder_features[-1] * sizes_downsample[-1], self.size_latent]

        decoder_linear = [self.size_latent, encoder_features[-1] * sizes_downsample[-1]]

        decoder_features = encoder_features[::-1]
        decoder_features[-1] = decoder_features[-2]

        self.latent_space = encoder_linear[-1]

        if self.activation_func == "ReLU":
            self.activation = nn.ReLU
        elif self.activation_func == "Tanh":
            self.activation = nn.Tanh
        elif self.activation_func == "Sigmoid":
            self.activation = nn.Sigmoid
        elif self.activation_func == "LeakyReLU":
            self.activation = nn.LeakyReLU
        elif self.activation_func == "ELU":
            self.activation = nn.ELU
        else:
            raise ValueError('Activation function not supported.')

        # Encoder
        self.encoder_features = torch.nn.ModuleList()

        for i in range(len(encoder_features) - 1):
            self.encoder_features.append(
                torch.nn.Conv1d(
                    encoder_features[i], encoder_features[i + 1], sizes_convs_encode[i],
                    padding=sizes_convs_encode[i] // 2
                )
            )
            self.encoder_features.append(self.activation())

        self.encoder_linear = torch.nn.ModuleList()

        for i in range(len(encoder_linear) - 1):
            self.encoder_linear.append(torch.nn.Linear(encoder_linear[i], encoder_linear[i + 1]))

        # Decoder
        self.decoder_linear = torch.nn.ModuleList()

        for i in range(len(decoder_linear) - 1):
            self.decoder_linear.append(torch.nn.Linear(decoder_linear[i], decoder_linear[i + 1]))

        self.decoder_features = torch.nn.ModuleList()

        for i in range(len(decoder_features) - 1):
            self.decoder_features.append(
                torch.nn.Conv1d(
                    decoder_features[i], decoder_features[i + 1], sizes_convs_decode[i],
                    padding=sizes_convs_decode[i] // 2
                )
            )
            self.decoder_features.append(self.activation())

        self.last_conv = torch.nn.Conv1d(
            decoder_features[-1], 3, sizes_convs_decode[-1],
            padding=sizes_convs_decode[-1] // 2
        )

        # Downsampling mats
        self.downsampling_mats = torch.nn.ParameterList()

        k = 0

        for i, layer in enumerate(self.encoder_features):
            if isinstance(layer, self.activation):
                self.downsampling_mats.append(
                    torch.nn.Parameter(
                        torch.zeros(sizes_downsample[k], sizes_downsample[k + 1]).to(self.device),
                        requires_grad=True
                    )
                )

                k += 1

        self.upsampling_mats = torch.nn.ParameterList()

        k = 0

        for i, layer in enumerate(self.decoder_features):
            if isinstance(layer, torch.nn.Conv1d):
                self.upsampling_mats.append(
                    torch.nn.Parameter(
                        torch.zeros(sizes_upsample[k], sizes_upsample[k + 1]).to(self.device),
                        requires_grad=True
                    )
                )

                k += 1

    def encoder(self, x) -> torch.Tensor:
        x = x.permute(0, 2, 1)

        k = 0

        for i, layer in enumerate(self.encoder_features):
            x = layer(x)

            if isinstance(layer, self.activation):
                x = torch.matmul(x, self.downsampling_mats[k])
                k += 1

        x = torch.flatten(x, start_dim=1, end_dim=2)

        for i, layer in enumerate(self.encoder_linear):
            x = layer(x)

        return x

    def decoder(self, x) -> torch.Tensor:
        for i, layer in enumerate(self.decoder_linear):
            x = layer(x)

        x = x.view(x.shape[0], -1, self.upsampling_mats[0].shape[0])

        k = 0

        for i, layer in enumerate(self.decoder_features):
            if isinstance(layer, torch.nn.Conv1d):
                x = torch.matmul(x, self.upsampling_mats[k])
                k += 1

            x = layer(x)

        x = self.last_conv(x)

        x = x.permute(0, 2, 1)

        return x

    def forward(self, x) -> torch.Tensor:
        latent = self.encoder(x)
        output = self.decoder(latent)

        return output

    def get_latent_code(self, x) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x) -> torch.Tensor:
        x = x.unsqueeze(0)
        return self.decoder(x)
