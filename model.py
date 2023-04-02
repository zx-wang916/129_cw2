import torch

from torch import nn


class Block(nn.Module):
    # this is the implementation of the Bottleneck module from ResNet
    def __init__(self, in_channel, out_channel, stride=2):
        super().__init__()

        width = out_channel // 4
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, width, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),

            nn.Conv2d(width, width, 3, stride, 1, 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),

            nn.Conv2d(width, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

        self.downsample = None
        if stride != 1 or in_channel != out_channel:
            self.downsample = nn.Conv2d(in_channel, out_channel, 1, stride, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.downsample is None:
            identity = x
        else:
            identity = self.downsample(x)

        out = self.conv(x)
        out = out + identity
        out = self.relu(out)
        return out


class ResUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_block1 = self.make_block(3, 64)
        self.encoder_block2 = self.make_block(64, 128)
        self.encoder_block3 = self.make_block(128, 256)
        self.encoder_block4 = self.make_block(256, 512)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.decoder_block1 = self.make_block(512, 256, 1, 1)
        self.decoder_block2 = self.make_block(256, 128, 1, 1)
        self.decoder_block3 = self.make_block(128, 64, 1, 1)
        self.decoder_block4 = self.make_block(64, 32, 1, 1)

        self.seg_out = nn.Sequential(
            nn.Conv2d(32, 3, 1, 1, bias=True),
            nn.Softmax(dim=1)
        )

        self.cla_out = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 37),
            nn.Softmax(dim=1)
        )

    @staticmethod
    def make_block(in_channel, out_channel, stride=2, num_block=2):
        block = nn.Sequential(Block(in_channel, out_channel, stride))

        for _ in range(num_block - 1):
            block.append(Block(out_channel, out_channel, 1))

        return block

    def backbone_forward(self, x):
        # encoder part
        enc1 = self.encoder_block1(x)
        enc2 = self.encoder_block2(enc1)
        enc3 = self.encoder_block3(enc2)
        enc4 = self.encoder_block4(enc3)

        # decoder part
        dec1 = self.upsample(enc4)
        dec1 = self.decoder_block1(dec1)

        dec2 = self.upsample(dec1 + enc3)
        dec2 = self.decoder_block2(dec2)

        dec3 = self.upsample(dec2 + enc2)
        dec3 = self.decoder_block3(dec3)

        dec4 = self.upsample(dec3 + enc1)
        dec4 = self.decoder_block4(dec4)

        return dec4, enc4

    def forward(self, x):
        out, _ = self.backbone_forward(x)
        return self.seg_out(out)

    def noisy_forward(self, x):
        # add noise to the input
        noise = torch.clamp(torch.randn_like(x) * 0.1, -0.2, 0.2).to(x.device)
        return self.forward(x + noise)

    def seg_cla_forward(self, x):
        dec_out, enc_out = self.backbone_forward(x)
        return self.seg_out(dec_out), self.cla_out(enc_out)
