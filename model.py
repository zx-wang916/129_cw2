import torch

from torch import nn
from torchvision.models.resnet import resnet34


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, encoder=True, stride=2):
        super().__init__()

        width = out_channel // 4
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, width, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU()
        )

        self.downsample = None

        if stride == 1:
            self.conv.append(nn.Conv2d(width, width, 3, stride, 1, 1))
        else:
            if encoder:
                self.conv.append(nn.Conv2d(width, width, 3, stride, 1, 1))
                self.downsample = nn.Conv2d(in_channel, out_channel, 1, 2, bias=False)

            else:
                self.conv.append(nn.ConvTranspose2d(width, width, 3, stride, 1, 1))
                self.downsample = nn.ConvTranspose2d(in_channel, out_channel, 1, 2, 0, 1, bias=False)

        self.conv.extend([
            nn.BatchNorm2d(width),
            nn.ReLU(),

            nn.Conv2d(width, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        ])

        self.relu = nn.ReLU()

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

        self.decoder_block1 = self.make_block(512, 256, False, 2)
        self.decoder_block2 = self.make_block(256, 128, False, 2)
        self.decoder_block3 = self.make_block(128, 64, False, 2)
        self.decoder_block4 = self.make_block(64, 32, False, 2)

        self.seg_out = nn.Sequential(
            nn.Conv2d(32, 3, 1, 1, bias=True),
            nn.Softmax(dim=1)
        )

        self.cla_out = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 37),
            nn.Softmax(dim=1)
        )

    @staticmethod
    def make_block(in_channel, out_channel, encoder=True, num_block=4):
        block = nn.Sequential(Block(in_channel, out_channel, encoder))

        for _ in range(num_block - 1):
            block.append(Block(out_channel, out_channel, encoder, 1))
        return block

    def backbone_forward(self, x):
        # encoder part
        enc1 = self.encoder_block1(x)
        enc2 = self.encoder_block2(enc1)
        enc3 = self.encoder_block3(enc2)
        enc4 = self.encoder_block4(enc3)

        # decoder part
        dec1 = self.decoder_block1(enc4)
        dec2 = self.decoder_block2(dec1 + enc3)
        dec3 = self.decoder_block3(dec2 + enc2)
        dec4 = self.decoder_block4(dec3 + enc1)

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
