import torch

from torch import nn
from torchvision.models.resnet import resnet34


class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=2):
        super().__init__()

        width = out_channel // 4
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, width, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),

            nn.ConvTranspose2d(width, width, 3, stride, 1, 1),
            nn.BatchNorm2d(width),
            nn.ReLU(),

            nn.Conv2d(width, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.relu = nn.ReLU()
        self.downsample = nn.ConvTranspose2d(in_channel, out_channel, 3, 2, 1, 1, bias=False)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResUNet(nn.Module):
    def __init__(self):
        super().__init__()
        encoder = resnet34()
        # encoder.load_state_dict(torch.load('./model/encoder_pretrained.pth'))

        self.head = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool,
        )
        self.encoder_block1 = encoder.layer1
        self.encoder_block2 = encoder.layer2
        self.encoder_block3 = encoder.layer3
        self.encoder_block4 = encoder.layer4

        self.decoder_block1 = DecoderBlock(512, 256)
        self.decoder_block2 = DecoderBlock(256, 128)
        self.decoder_block3 = DecoderBlock(128, 64)
        self.decoder_block4 = DecoderBlock(64, 32)

        self.seg_out = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, 2, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Softmax(dim=1)
        )

        self.cla_out = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 37),
            nn.Softmax(dim=1)
        )

    def backbone_forward(self, x):
        # encoder part
        head = self.head(x)
        enc1 = self.encoder_block1(head)
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
