import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNetUNet(nn.Module):
    """
    U-Net with a ResNet-18 encoder backbone for 84x84x3 inputs.
    decoder upsamples and concatenates skip features from encoder stages.
    """
    def __init__(self, out_channels=60, pretrained=False):
        super(ResNetUNet, self).__init__()
        # Load ResNet-18 backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        # Encoder layers
        self.layer0 = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu
        )  # output: [B,64,42,42]
        self.pool0 = self.backbone.maxpool  # -> [B,64,21,21]
        self.layer1 = self.backbone.layer1  # -> [B,64,21,21]
        self.layer2 = self.backbone.layer2  # -> [B,128,11,11]
        self.layer3 = self.backbone.layer3  # -> [B,256,6,6]
        self.layer4 = self.backbone.layer4  # -> [B,512,3,3]

        # Decoder blocks
        self.conv4 = self._double_conv(512 + 256, 256)
        self.conv3 = self._double_conv(256 + 128, 128)
        self.conv2 = self._double_conv(128 + 64, 64)
        self.conv1 = self._double_conv(64 + 64, 64)
        self.conv0 = self._double_conv(64, 64)

        # Final 1x1 conv
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder forward
        x0 = self.layer0(x)      # [B,64,42,42]
        x1 = self.layer1(self.pool0(x0))  # [B,64,21,21]
        x2 = self.layer2(x1)     # [B,128,11,11]
        x3 = self.layer3(x2)     # [B,256,6,6]
        x4 = self.layer4(x3)     # [B,512,3,3]

        # Decoder forward
        d4 = F.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, x3], dim=1)  # [B,512+256,6,6]
        d4 = self.conv4(d4)               # [B,256,6,6]

        d3 = F.interpolate(d4, size=x2.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, x2], dim=1)  # [B,256+128,11,11]
        d3 = self.conv3(d3)               # [B,128,11,11]

        d2 = F.interpolate(d3, size=x1.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, x1], dim=1)  # [B,128+64,21,21]
        d2 = self.conv2(d2)               # [B,64,21,21]

        d1 = F.interpolate(d2, size=x0.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, x0], dim=1)  # [B,64+64,42,42]
        d1 = self.conv1(d1)               # [B,64,42,42]

        d0 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)  # -> [B,64,84,84]
        d0 = self.conv0(d0)            # [B,32,84,84]

        out = self.final_conv(d0)      # [B,out_channels,84,84]
        return out

    def _double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

if __name__ == '__main__':
    # Example usage:
    model = ResNetUNet(out_channels=61, pretrained=False)
    x = torch.randn(2, 3, 84, 84)
    y = model(x)
    print(y.shape)  # torch.Size([2, 60, 84, 84])
    logits = y[:,-1, :,:].reshape(2, -1)
    one_hot = F.gumbel_softmax(logits, hard=True, dim=1)
    breakpoint()