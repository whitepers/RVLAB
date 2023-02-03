import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        se = self.squeeze(x).view(b, c)
        se = self.excitation(se).view(b, c, 1, 1)
        return x * se.expand_as(x)

def mobile_block(in_dim, out_dim, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=stride, padding=1, groups=in_dim),
        nn.BatchNorm2d(in_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(inplace=True),
        SEBlock(c=out_dim, r=16),
    )

class SEMobileNet(nn.Module):
    def __init__(self, width_multi=1, resolution_multi=1, num_classes=1000):
        super(SEMobileNet, self).__init__()
        base_width = int(32 * width_multi)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=base_width, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            mobile_block(base_width, base_width*2),
            mobile_block(base_width*2, base_width*4, 2),
            mobile_block(base_width*4, base_width*4),
            mobile_block(base_width*4, base_width*8, 2),
            mobile_block(base_width*8, base_width*8),
            mobile_block(base_width*8, base_width*16, 2), # 800x800 -> 50x50
            *[mobile_block(base_width*16, base_width*16) for _ in range(5)], # 512 channel
            mobile_block(base_width*16, base_width*32, 2),
            mobile_block(base_width*32, base_width*32),

            nn.AvgPool2d(7),
        )
        self.classifier = nn.Linear(base_width*32, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_bb_clf():
    model = SEMobileNet()
    backbone = model.conv[:-3]
    for p in backbone[:2].parameters():
        p.requires_grad = False
    return backbone, nn.Sequential(nn.Linear(512*7*7, 4096), nn.ReLU(inplace=True))