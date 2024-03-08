import torch
import torch.nn as nn
from torch.nn import functional as F
from canconv.layers.canconv import CANConv


class CANResBlock(nn.Module):
    def __init__(self, channels, cluster_num, filter_threshold, cluster_source="channel", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = CANConv(channels, channels, cluster_num=cluster_num,
                            cluster_source=cluster_source, filter_threshold=filter_threshold)
        self.conv2 = CANConv(channels, channels, cluster_num=cluster_num)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x, cache_indice=None, cluster_override=None):
        res, idx = self.conv1(x, cache_indice, cluster_override)
        res = F.leaky_relu(res)
        res, _ = self.conv2(res, cache_indice, idx)
        x = x + res
        return x, idx


class ConvDown(nn.Module):
    def __init__(self, in_channels, dsconv=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if dsconv:
            self.conv = nn.Sequential(
                # nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_channels, in_channels, 2, 2, 0),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1,
                          groups=in_channels, bias=False),
                nn.Conv2d(in_channels, in_channels*2, 1, 1, 0)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=2, padding=1),
                # nn.Conv2d(in_channels, in_channels, 2, 2, 0),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels*2, 3, 1, 1)
            )

    def forward(self, x):
        return self.conv(x)


class ConvUp(nn.Module):
    def __init__(self, in_channels, dsconv=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # self.conv1 = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels//2, 2, 2, 0)
        if dsconv:
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels//2, in_channels//2, 3, 1,
                          1, groups=in_channels//2, bias=False),
                nn.Conv2d(in_channels//2, in_channels//2, 1, 1, 0)
            )
        else:
            self.conv2 = nn.Conv2d(in_channels//2, in_channels//2, 3, 1, 1)

    def forward(self, x, y):
        x = F.leaky_relu(self.conv1(x))
        x = x + y
        x = F.leaky_relu(self.conv2(x))
        return x


class CANNet(nn.Module):
    def __init__(self, spectral_num=8, channels=32, cluster_num=32, filter_threshold=0.005, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.head_conv = nn.Conv2d(spectral_num+1, channels, 3, 1, 1)
        self.rb1 = CANResBlock(channels, cluster_num, filter_threshold)
        self.down1 = ConvDown(channels)
        self.rb2 = CANResBlock(channels*2, cluster_num, filter_threshold)
        self.down2 = ConvDown(channels*2)
        self.rb3 = CANResBlock(channels*4, cluster_num, filter_threshold)
        self.up1 = ConvUp(channels*4)
        self.rb4 = CANResBlock(channels*2, cluster_num, filter_threshold)
        self.up2 = ConvUp(channels*2)
        self.rb5 = CANResBlock(channels, cluster_num, filter_threshold)
        self.tail_conv = nn.Conv2d(channels, spectral_num, 3, 1, 1)

    def forward(self, pan, lms, cache_indice=None):
        x1 = torch.cat([pan, lms], dim=1)
        x1 = self.head_conv(x1)
        x1, idx1 = self.rb1(x1, cache_indice)
        x2 = self.down1(x1)
        x2, idx2 = self.rb2(x2, cache_indice)
        x3 = self.down2(x2)
        x3, _ = self.rb3(x3, cache_indice)
        x4 = self.up1(x3, x2)
        del x2
        x4, _ = self.rb4(x4, cache_indice, idx2)
        del idx2
        x5 = self.up2(x4, x1)
        del x1
        x5, _ = self.rb5(x5, cache_indice, idx1)
        del idx1
        x5 = self.tail_conv(x5)
        return lms + x5
