import torch

from torch import nn
from torchvision.models.resnet import Bottleneck


class MyBottleNeck(Bottleneck):
    expansion: int = 1


class ResUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_block1 = self.make_block(3, 32)
        self.encoder_block2 = self.make_block(32, 64)
        self.encoder_block3 = self.make_block(64, 128)
        self.encoder_block4 = self.make_block(128, 256)

        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.decoder_block1 = self.make_block(256, 128, stride=1)
        self.decoder_block2 = self.make_block(128, 64, stride=1)
        self.decoder_block3 = self.make_block(64, 32, stride=1)
        self.decoder_block4 = self.make_block(32, 3, stride=1)

        self.cov_out = nn.Sequential(
            nn.Conv2d(3, 3, 1, 1, 0),
            nn.Softmax(dim=1)
        )

    @staticmethod
    def make_block(in_channel, out_channel, stride=2):
        skip_connection = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        block = nn.Sequential(
            MyBottleNeck(in_channel, out_channel, stride, skip_connection),
            MyBottleNeck(out_channel, out_channel),
            MyBottleNeck(out_channel, out_channel),
            MyBottleNeck(out_channel, out_channel)
        )

        return block

    def forward(self, x):
        enc1 = self.encoder_block1(x)
        enc2 = self.encoder_block2(enc1)
        enc3 = self.encoder_block3(enc2)
        enc4 = self.encoder_block4(enc3)

        dec1 = self.up_sample(enc4)
        dec1 = self.decoder_block1(dec1)

        dec2 = self.up_sample(dec1 + enc3)
        dec2 = self.decoder_block2(dec2)

        dec3 = self.up_sample(dec2 + enc2)
        dec3 = self.decoder_block3(dec3)

        dec4 = self.up_sample(dec3 + enc1)
        dec4 = self.decoder_block4(dec4)

        out = self.cov_out(dec4)
        return out
