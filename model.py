import torch
import torch.nn as nn


class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        # stage 1 (3 -> 64 channels, downsample by 2)
        self.stage1 = nn.Sequential(
            Conv2dLayer(3, 32, kernel_size=3, stride=2, padding=0),
            Conv2dLayer(32, 32, kernel_size=3, stride=1, padding=0),
            Conv2dLayer(32, 64, kernel_size=3, stride=1, padding=1)
        )

        # stage 2 (parallel downsampling branch, downsample by 2)
        self.stage2_branch_a = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.stage2_branch_b = Conv2dLayer(64, 96, kernel_size=3, stride=2, padding=0)

        # stage 3 (filter concatenation branch)
        self.stage3_branch_a = nn.Sequential(
            Conv2dLayer(160, 64, kernel_size=1, stride=1, padding=0),
            Conv2dLayer(64, 96, kernel_size=3, stride=1, padding=0)
        )
        self.stage3_branch_b = nn.Sequential(
            Conv2dLayer(160, 64, kernel_size=1, stride=1, padding=0),
            Conv2dLayer(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            Conv2dLayer(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            Conv2dLayer(64, 96, kernel_size=3, stride=1, padding=0)
        )

        # stage 4 (parallel downsampling branch, downsample by 2)
        self.stage4_branch_a = Conv2dLayer(192, 192, kernel_size=3, stride=2, padding=0)
        self.stage4_branch_b = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = self.stage1(x)
        x = torch.cat([self.stage2_branch_a(x), self.stage2_branch_b(x)], dim=1)
        x = torch.cat([self.stage3_branch_a(x), self.stage3_branch_b(x)], dim=1)
        x = torch.cat([self.stage4_branch_a(x), self.stage4_branch_b(x)], dim=1)
        return x


class AdaptedStem(nn.Module):
    def __init__(self):
        super().__init__()
        # stride=1 on first conv preserves 2x more spatial fidelity on small 128px inputs
        self.stage1 = nn.Sequential(
            Conv2dLayer(3, 32, kernel_size=3, stride=1, padding=1),
            Conv2dLayer(32, 32, kernel_size=3, stride=1, padding=0),
            Conv2dLayer(32, 64, kernel_size=3, stride=1, padding=1)
        )

        self.stage2_branch_a = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.stage2_branch_b = Conv2dLayer(64, 96, kernel_size=3, stride=2, padding=0)

        self.stage3_branch_a = nn.Sequential(
            Conv2dLayer(160, 64, kernel_size=1, stride=1, padding=0),
            Conv2dLayer(64, 96, kernel_size=3, stride=1, padding=0)
        )
        self.stage3_branch_b = nn.Sequential(
            Conv2dLayer(160, 64, kernel_size=1, stride=1, padding=0),
            Conv2dLayer(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            Conv2dLayer(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            Conv2dLayer(64, 96, kernel_size=3, stride=1, padding=0)
        )

        self.stage4_branch_a = Conv2dLayer(192, 192, kernel_size=3, stride=2, padding=0)
        self.stage4_branch_b = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = self.stage1(x)
        x = torch.cat([self.stage2_branch_a(x), self.stage2_branch_b(x)], dim=1)
        x = torch.cat([self.stage3_branch_a(x), self.stage3_branch_b(x)], dim=1)
        x = torch.cat([self.stage4_branch_a(x), self.stage4_branch_b(x)], dim=1)
        return x


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        bottleneck = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class InceptionA(nn.Module):
    def __init__(self, scale=0.17, use_se=False):
        super().__init__()
        self.branch_a = Conv2dLayer(384, 32, kernel_size=1, stride=1, padding=0)
        self.branch_b = nn.Sequential(
            Conv2dLayer(384, 32, kernel_size=1, stride=1, padding=0),
            Conv2dLayer(32, 32, kernel_size=3, stride=1, padding=1)
        )
        self.branch_c = nn.Sequential(
            Conv2dLayer(384, 32, kernel_size=1, stride=1, padding=0),
            Conv2dLayer(32, 48, kernel_size=3, stride=1, padding=1),
            Conv2dLayer(48, 64, kernel_size=3, stride=1, padding=1)
        )
        self.stage_2 = nn.Conv2d(128, 384, kernel_size=1, stride=1, padding=0)
        # identity avoids extra runtime overhead when se is disabled
        self.se = SEBlock(384) if use_se else nn.Identity()
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = torch.cat([self.branch_a(x), self.branch_b(x), self.branch_c(x)], dim=1)
        res = self.stage_2(x1) * self.scale
        return self.relu(self.se(res) + x)


class ReductionA(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_a = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.branch_b = Conv2dLayer(384, 384, kernel_size=3, stride=2, padding=0)
        self.branch_c = nn.Sequential(
            Conv2dLayer(384, 256, kernel_size=1, stride=1, padding=0),
            Conv2dLayer(256, 256, kernel_size=3, stride=1, padding=1),
            Conv2dLayer(256, 384, kernel_size=3, stride=2, padding=0)
        )

    def forward(self, x):
        return torch.cat([self.branch_a(x), self.branch_b(x), self.branch_c(x)], dim=1)


class InceptionB(nn.Module):
    def __init__(self, scale=0.10, use_se=False):
        super().__init__()
        self.branch_a = Conv2dLayer(1152, 192, kernel_size=1, stride=1, padding=0)
        self.branch_b = nn.Sequential(
            Conv2dLayer(1152, 128, kernel_size=1, stride=1, padding=0),
            Conv2dLayer(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            Conv2dLayer(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )
        self.stage_2 = nn.Conv2d(384, 1152, kernel_size=1, stride=1, padding=0)
        self.se = SEBlock(1152) if use_se else nn.Identity()
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.stage_2(torch.cat([self.branch_a(x), self.branch_b(x)], dim=1)) * self.scale
        return self.relu(self.se(res) + x)


class ReductionB(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_a = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.branch_b = nn.Sequential(
            Conv2dLayer(1152, 256, kernel_size=1, stride=1, padding=0),
            Conv2dLayer(256, 384, kernel_size=3, stride=2, padding=0)
        )
        self.branch_c = nn.Sequential(
            Conv2dLayer(1152, 256, kernel_size=1, stride=1, padding=0),
            Conv2dLayer(256, 288, kernel_size=3, stride=2, padding=0)
        )
        self.branch_d = nn.Sequential(
            Conv2dLayer(1152, 256, kernel_size=1, stride=1, padding=0),
            Conv2dLayer(256, 288, kernel_size=3, stride=1, padding=1),
            Conv2dLayer(288, 320, kernel_size=3, stride=2, padding=0)
        )

    def forward(self, x):
        return torch.cat([self.branch_a(x), self.branch_b(x), self.branch_c(x), self.branch_d(x)], dim=1)


class InceptionC(nn.Module):
    def __init__(self, scale=0.20, use_se=False):
        super().__init__()
        self.branch_a = Conv2dLayer(2144, 192, kernel_size=1, stride=1, padding=0)
        self.branch_b = nn.Sequential(
            Conv2dLayer(2144, 192, kernel_size=1, stride=1, padding=0),
            Conv2dLayer(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            Conv2dLayer(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )
        self.conv_up = nn.Conv2d(448, 2144, kernel_size=1, stride=1, padding=0)
        self.se = SEBlock(2144) if use_se else nn.Identity()
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.conv_up(torch.cat([self.branch_a(x), self.branch_b(x)], dim=1)) * self.scale
        return self.relu(self.se(res) + x)


class InceptionResnetV2(nn.Module):
    def __init__(
        self,
        num_classes=10,
        num_blocks=(10, 20, 10),
        dropout=0.2,
        stem_type="standard",
        use_se=False
    ):
        super().__init__()
        if stem_type == "adapted":
            self.stem = AdaptedStem()
        elif stem_type == "standard":
            self.stem = Stem()
        elif isinstance(stem_type, nn.Module):
            self.stem = stem_type
        else:
            raise ValueError(f"unsupported stem_type: {stem_type}. Expected 'standard', 'adapted', or an nn.Module.")

        self.stage_a = nn.Sequential(*[InceptionA(scale=0.17, use_se=use_se) for _ in range(num_blocks[0])])
        self.red_a = ReductionA()

        self.stage_b = nn.Sequential(*[InceptionB(scale=0.10, use_se=use_se) for _ in range(num_blocks[1])])
        self.red_b = ReductionB()

        self.stage_c = nn.Sequential(*[InceptionC(scale=0.20, use_se=use_se) for _ in range(num_blocks[2])])

        self.conv_head = Conv2dLayer(2144, 1536, kernel_size=1, stride=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage_a(x)
        x = self.red_a(x)
        x = self.stage_b(x)
        x = self.red_b(x)
        x = self.stage_c(x)
        x = self.conv_head(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


# convenience subclasses for explicit ablation variants
class InceptionResnetV2_mod(InceptionResnetV2):
    def __init__(self, **kwargs):
        kwargs["stem_type"] = "adapted"
        super().__init__(**kwargs)


class SEInceptionResnetV2(InceptionResnetV2):
    def __init__(self, **kwargs):
        kwargs["use_se"] = True
        super().__init__(**kwargs)