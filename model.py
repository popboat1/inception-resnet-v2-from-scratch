import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------- LAYERS

class Conv2dLayer(nn.Module):
  """
  Conv2d + BatchNorm2d + ReLU
  """
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
    super().__init__()
    # conv
    self.conv = nn.Conv2d(in_channels,
                          out_channels, 
                          kernel_size, 
                          stride, 
                          padding, 
                          bias=False)
    # batchnorm
    self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    # relu
    self.relu = nn.ReLU(inplace=True)
  
  def forward(self, x):
    return self.relu(self.bn(self.conv(x)))

class Stem(nn.Module):
  """
  acts as a funnel at the start of the network
  """
  def __init__(self):
    super().__init__()

    # stage 1, 3 -> 64 channels
    self.stage1 = nn.Sequential(
        Conv2dLayer(3, 32, kernel_size=3, stride=2, padding=0), # (b, 3, 128, 128) -> (b, 32, 63, 63)
        Conv2dLayer(32, 32, kernel_size=3, stride=1, padding=0), # (b, 32, 63, 63) -> (b, 32, 61, 61)
        Conv2dLayer(32, 64, kernel_size=3, stride=1, padding=1) # (b, 32, 61, 61) -> (b, 64, 61, 61)
    )

    # stage 2, first split branch
    # parallel downsampling 
    self.stage2_branch_a = nn.MaxPool2d(kernel_size=3, stride=2, padding=0) 
    self.stage2_branch_b = Conv2dLayer(64, 96, kernel_size=3, stride=2, padding=0)

    # stage 3, second branch
    self.stage3_branch_a = nn.Sequential(
        Conv2dLayer(160, 64, kernel_size=1, stride=1, padding=0),
        Conv2dLayer(64, 96, kernel_size=3, stride=1, padding=0)
    )
    self.stage3_branch_b = nn.Sequential(
        Conv2dLayer(160, 64, kernel_size=1, stride=1, padding=0),
        Conv2dLayer(64, 64, kernel_size=(7,1), stride=1, padding=(3,0)),
        Conv2dLayer(64, 64, kernel_size=(1,7), stride=1, padding=(0,3)),
        Conv2dLayer(64, 96, kernel_size=3, stride=1, padding=0)
    )

    # stage 4 final splits
    self.stage4_branch_a = Conv2dLayer(192, 192, kernel_size=3, stride=2, padding=0)
    self.stage4_branch_b = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

  def forward(self, x):
    # stage 1 
    # (b, 3, 128, 128) -> (b, 64, 61, 61)
    x = self.stage1(x)

    # stage 2 
    # (b, 64, 61, 61) -> (b, 160, 30, 30)
    x = torch.cat([self.stage2_branch_a(x), self.stage2_branch_b(x)], dim=1)

    # stage 3
    # (b, 160, 30, 30) -> (b, 192, 28, 28)
    x = torch.cat([self.stage3_branch_a(x), self.stage3_branch_b(x)], dim=1)

    # stage 4
    # (b, 192, 28, 28) -> (b, 384, 13, 13)
    x = torch.cat([self.stage4_branch_a(x), self.stage4_branch_b(x)], dim=1)

    return x

class AdaptedStem(nn.Module):
  """
  acts as a funnel at the start of the network
  """
  def __init__(self):
        super().__init__()
        # stage 1: changed stride=2 to stride=1, padding=1
        self.stage1 = nn.Sequential(
            Conv2dLayer(3, 32, kernel_size=3, stride=1, padding=1), # (b, 3, 128, 128) -> (b, 32, 128, 128)
            Conv2dLayer(32, 32, kernel_size=3, stride=1, padding=0), # (b, 32, 128, 128) -> (b, 32, 126, 126)
            Conv2dLayer(32, 64, kernel_size=3, stride=1, padding=1)  # (b, 32, 126, 126) -> (b, 64, 126, 126)
        )
        # stage 2 (126 -> 62)
        self.stage2_branch_a = nn.MaxPool2d(kernel_size=3, stride=2, padding=0) 
        self.stage2_branch_b = Conv2dLayer(64, 96, kernel_size=3, stride=2, padding=0)

        # stage 3 (62 -> 60)
        self.stage3_branch_a = nn.Sequential(
            Conv2dLayer(160, 64, kernel_size=1, stride=1, padding=0),
            Conv2dLayer(64, 96, kernel_size=3, stride=1, padding=0)
        )
        self.stage3_branch_b = nn.Sequential(
            Conv2dLayer(160, 64, kernel_size=1, stride=1, padding=0),
            Conv2dLayer(64, 64, kernel_size=(7,1), stride=1, padding=(3,0)),
            Conv2dLayer(64, 64, kernel_size=(1,7), stride=1, padding=(0,3)),
            Conv2dLayer(64, 96, kernel_size=3, stride=1, padding=0)
        )

        # stage 4 (60 -> 29)
        self.stage4_branch_a = Conv2dLayer(192, 192, kernel_size=3, stride=2, padding=0)
        self.stage4_branch_b = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
  def forward(self, x):
        x = self.stage1(x)
        x = torch.cat([self.stage2_branch_a(x), self.stage2_branch_b(x)], dim=1)
        x = torch.cat([self.stage3_branch_a(x), self.stage3_branch_b(x)], dim=1)
        x = torch.cat([self.stage4_branch_a(x), self.stage4_branch_b(x)], dim=1)
        return x

class InceptionA(nn.Module):
  def __init__(self, scale=0.17):
    super().__init__()

    # stage 1
    # parallel multi-scale branches, all same-padded keeping (b, c, 13, 13)
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

    # stage 2
    self.stage_2 = nn.Conv2d(128, 384, kernel_size=1, stride=1, padding=0)

    self.scale = scale
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    # concat along channel dim: 32 + 32 + 64 = 128 -> (b, 128, 13, 13)
    x1 = torch.cat([self.branch_a(x), self.branch_b(x), self.branch_c(x)], dim=1)
    # linear 1x1 projection to 384, scale by alpha, add identity shortcut, then relu -> (b, 384, 13, 13)
    return self.relu((self.stage_2(x1) * self.scale) + x)

class ReductionA(nn.Module):
  def __init__(self, scale=0.10):
    super().__init__()

    self.branch_a = nn.MaxPool2d(kernel_size=3, stride=2, padding=0) # (b, 384, 13, 13) -> (b, 384, 6, 6)

    self.branch_b = Conv2dLayer(384, 384, kernel_size=3, stride=2, padding=0) # (b, 384, 13, 13) -> (b, 384, 6, 6)

    self.branch_c = nn.Sequential(
        Conv2dLayer(384, 256, kernel_size=1, stride=1, padding=0),
        Conv2dLayer(256, 256, kernel_size=3, stride=1, padding=1),
        Conv2dLayer(256, 384, kernel_size=3, stride=2, padding=0)
    ) # (b, 384, 13, 13) -> (b, 384, 6, 6)

  def forward(self, x):
    return torch.cat([self.branch_a(x), self.branch_b(x), self.branch_c(x)], dim=1) # (b, 1152, 6, 6)

class InceptionB(nn.Module):
  def __init__(self, scale=0.10):
    super().__init__()

    # stage 1
    self.branch_a = Conv2dLayer(1152, 192, kernel_size=1, stride=1, padding=0) # (b, 1152, 6, 6) -> (b, 192, 6, 6)
    self.branch_b = nn.Sequential(
        Conv2dLayer(1152, 128, kernel_size=1, stride=1, padding=0),
        Conv2dLayer(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)),
        Conv2dLayer(160, 192, kernel_size=(7,1), stride=1, padding=(3,0))
    ) # (b, 1152, 6, 6) -> (b, 192, 6, 6)

    # stage 2
    self.stage_2 = nn.Conv2d(384, 1152, kernel_size=1, stride=1, padding=0)

    self.scale = scale
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    return self.relu(self.stage_2(torch.cat([self.branch_a(x), self.branch_b(x)], dim=1)) * self.scale + x)

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
    # (b, 1152, 6, 6) -> (b, 2144, 2, 2)
    
class InceptionC(nn.Module):
  def __init__(self, scale=0.20):
    super().__init__()

    # stage 1
    self.branch_a = Conv2dLayer(2144, 192, kernel_size=1, stride=1, padding=0) # (b, 2144, 2, 2) -> (b, 192, 2, 2)
    self.branch_b = nn.Sequential(
        Conv2dLayer(2144, 192, kernel_size=1, stride=1, padding=0),
        Conv2dLayer(192, 224, kernel_size=(1, 3), stride=1, padding=(0,1)),
        Conv2dLayer(224, 256, kernel_size=(3, 1), stride=1, padding=(1,0))
    ) # (b, 2144, 2, 2) -> (b, 256, 2, 2)

    # stage 2
    self.conv_up = nn.Conv2d(448, 2144, kernel_size=1, stride=1, padding=0)

    self.scale = scale
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    return self.relu(self.conv_up(torch.cat([self.branch_a(x), self.branch_b(x)], dim=1)) * self.scale + x)







# --------------- MODEL DEFINITION

class InceptionResnetV2(nn.Module):
  def __init__(self, num_classes=10, num_blocks=(10, 20, 10), dropout=0.2):
    super().__init__()
    
    self.stem = Stem()

    self.stage_a = nn.Sequential(*[InceptionA(scale=0.17) for _ in range(num_blocks[0])])
    self.red_a = ReductionA()

    self.stage_b = nn.Sequential(*[InceptionB(scale=0.10) for _ in range(num_blocks[1])])
    self.red_b = ReductionB()

    self.stage_c = nn.Sequential(*[InceptionC(scale=0.20) for _ in range(num_blocks[2])])

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

    # flatten (b, 1536, 1, 1) -> (b, 1536)
    x = torch.flatten(x, 1)
    x = self.dropout(x)
    return self.fc(x)


class InceptionResnetV2_mod(nn.Module):
  def __init__(self, num_classes=10, num_blocks=(10, 20, 10), dropout=0.2):
    super().__init__()
    
    self.stem = AdaptedStem()

    self.stage_a = nn.Sequential(*[InceptionA(scale=0.17) for _ in range(num_blocks[0])])
    self.red_a = ReductionA()

    self.stage_b = nn.Sequential(*[InceptionB(scale=0.10) for _ in range(num_blocks[1])])
    self.red_b = ReductionB()

    self.stage_c = nn.Sequential(*[InceptionC(scale=0.20) for _ in range(num_blocks[2])])

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

    # flatten (b, 1536, 1, 1) -> (b, 1536)
    x = torch.flatten(x, 1)
    x = self.dropout(x)
    return self.fc(x)