"""Невронни мрежи за разпознаване на аномалии върху хистопатологични патчове.

Три семейства, отговарящи на стъпки 3-5 от заданието:

* :class:`SimpleCNN`        -- базова конволюционна мрежа (стъпка 3).
* :class:`VGG`              -- конфигурируем VGG (VGG11 / VGG16, стъпка 4).
* :class:`UNetAutoencoder`  -- автоенкодер тип U-Net (стъпка 5).

Всички класификатори приемат вход (N, 3, H, W) и връщат логити (N, num_classes).
"""

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """Базова CNN: няколко conv-блока + класификационна глава.

    Всеки блок е Conv2d -> BatchNorm -> ReLU -> MaxPool, който намалява
    пространствените размери наполовина.
    """

    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 channels: tuple[int, ...] = (32, 64, 128)) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        prev = in_channels
        for ch in channels:
            blocks += [
                nn.Conv2d(prev, ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            prev = ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# Конфигурации на VGG: число = брой канали на conv слой, 'M' = max pooling.
VGG_CONFIGS: dict[str, list[int | str]] = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16':
    [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    """Конфигурируема VGG архитектура (стъпка 4).

    Използва adaptive pooling, за да работи и с малки патчове (напр. 96x96),
    при които класическите пет max-pool слоя биха стопили картата до 0.
    """

    def __init__(self,
                 config: str = 'VGG16',
                 in_channels: int = 3,
                 num_classes: int = 2,
                 batch_norm: bool = True) -> None:
        super().__init__()
        if config not in VGG_CONFIGS:
            raise ValueError(f'Непознат VGG конфиг: {config!r}. '
                             f'Избери от {list(VGG_CONFIGS)}.')
        self.config = config
        self.features = self._make_layers(VGG_CONFIGS[config], in_channels, batch_norm)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _make_layers(config: list[int | str], in_channels: int, batch_norm: bool) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = in_channels
        for v in config:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(prev, int(v), kernel_size=3, padding=1))
                if batch_norm:
                    layers.append(nn.BatchNorm2d(int(v)))
                layers.append(nn.ReLU(inplace=True))
                prev = int(v)
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class _DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x2 -- базовият блок на U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetAutoencoder(nn.Module):
    """U-Net тип автоенкодер (стъпка 5).

    Енкодер-декодер със skip връзки. Изходът има същата форма като входа.

    Два режима на употреба за откриване на аномалии:

    * ``out_channels == in_channels``: реконструкция. Обучава се само върху
      нормални патчове; аномалията се измерва чрез грешката на реконструкция.
    * ``out_channels == 1``: сегментация на туморна маска (с BCEWithLogitsLoss).
    """

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 features: tuple[int, ...] = (64, 128, 256, 512)) -> None:
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        prev = in_channels
        for f in features:
            self.downs.append(_DoubleConv(prev, f))
            prev = f

        self.bottleneck = _DoubleConv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(_DoubleConv(f * 2, f))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections: list[torch.Tensor] = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:])
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


class ConvAutoencoder(nn.Module):
    """Класически конволюционен автоенкодер **без** skip връзки.

    За разлика от :class:`UNetAutoencoder`, информацията минава през тясно гърло
    (bottleneck) -- няма пряк път вход->изход. Това принуждава модела да научи
    разпределението на нормалните патчове и затова е по-подходящ за
    reconstruction-based anomaly detection: невижданите (tumor) патчове не пасват
    в наученото нормално -> по-висока грешка на реконструкция.

    Вход/изход (N, 3, 96, 96); изходът минава през Sigmoid -> стойности в [0, 1],
    за да съвпада с входа (``ToTensor`` дава [0, 1]).
    """

    def __init__(self, in_channels: int = 3, latent_channels: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 96 -> 48
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 48 -> 24
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 24 -> 12
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_channels, kernel_size=1),  # тясно гърло (без skip)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 12 -> 24
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 24 -> 48
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),  # 48 -> 96
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
