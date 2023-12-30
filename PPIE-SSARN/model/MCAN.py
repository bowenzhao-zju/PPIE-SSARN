import torch.nn as nn
import numpy as np
import math
import torch

def get_WB_filter(size):
    """make a 2D weight bilinear kernel suitable for WB_Conv"""
    ligne = []
    colonne = []
    for i in range(size):
        if (i + 1) <= np.floor(math.sqrt(16)):
            ligne.append(i + 1)
            colonne.append(i + 1)
        else:
            ligne.append(ligne[i - 1] - 1.0)
            colonne.append(colonne[i - 1] - 1.0)
    BilinearFilter = np.zeros(size * size)
    for i in range(size):
        for j in range(size):
            BilinearFilter[(j + i * size)] = (ligne[i] * colonne[j] / 16)
    filter0 = np.reshape(BilinearFilter, (7, 7))
    return torch.from_numpy(filter0).float()

class _Conv_Block(nn.Module):
    def __init__(self):
        super(_Conv_Block, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.cov_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, x):
        residual = x
        output = self.cov_block(x)
        output += residual
        output = self.relu(output)
        return output

class Shuffle_d(nn.Module):
    def __init__(self, scale=2):
        super(Shuffle_d, self).__init__()
        self.scale = scale

    def forward(self, x):
        def _space_to_channel(x, scale):
            b, C, h, w = x.size()
            Cout = C * scale ** 2
            hout = h // scale
            wout = w // scale
            x = x.contiguous().view(b, C, hout, scale, wout, scale)
            x = x.contiguous().permute(0, 1, 3, 5, 2, 4)
            x = x.contiguous().view(b, Cout, hout, wout)
            return x
        return _space_to_channel(x, self.scale)

class MALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(MALayer, self).__init__()
        self.shuffledown = Shuffle_d(4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel*16, channel*16 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel*16 // reduction, channel*16, bias=False),
            nn.Sigmoid()
        )
        self.shuffleup = nn.PixelShuffle(4)

    def forward(self, x):
        ex_x = self.shuffledown(x)
        b, c, _, _ = ex_x.size()
        y = self.avg_pool(ex_x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        ex_x = ex_x * y.expand_as(ex_x)
        x = self.shuffleup(ex_x)
        # buff_error = buff_x - x
        # buff_error = buff_x - x
        return x

class _Conv_attention_Block(nn.Module):
    def __init__(self):
        super(_Conv_attention_Block, self).__init__()
        self.ma = MALayer(64, 4)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.cov_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, x):
        residual = x
        output = self.cov_block(x)
        output = self.ma(output)
        output += residual
        output = self.relu(output)
        return output

class branch_block_front(nn.Module):
    def __init__(self):
        super(branch_block_front, self).__init__()
        # self.relu = nn.LeakyReLU(0.2, inplace=False)
        self.se = MALayer(16, 4)
        # self.se = SELayer(16, 4)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # self.front_conv_input = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = self.se(x)
        x = self.relu(x)
        # x = self.front_conv_input(x)
        return x

class branch_block_back(nn.Module):
    def __init__(self):
        super(branch_block_back, self).__init__()
        # self.relu = nn.LeakyReLU(0.2, inplace=True)
        # self.se = SELayer(64, 16)
        self.cov_block = nn.Sequential(
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, x):
        # x = self.relu(x)
        output = self.cov_block(x)
        return output

class Pos2Weight(nn.Module):
    def __init__(self, outC=16, kernel_size=5, inC=1):
        super(Pos2Weight, self).__init__()
        self.inC = inC
        self.kernel_size = kernel_size
        self.outC = outC
        self.meta_block = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.kernel_size * self.kernel_size * self.inC * self.outC)
        )

    def forward(self, x):
        output = self.meta_block(x)
        return output

class MCAN(nn.Module):
    def __init__(self):
        super(MCAN, self).__init__()
        self.scale = 1
        self.outC = 16
        self.WB_Conv = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=3, bias=False, groups=16)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.front_conv_input = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1,
                                          bias=True)
        self.convt_br1_front = self.make_layer(branch_block_front)
        self.convt_F1 = self.make_layer(_Conv_attention_Block)
        self.convt_F2 = self.make_layer(_Conv_attention_Block)
        self.convt_br1_back = self.make_layer(branch_block_back)
        self.P2W = Pos2Weight(outC=self.outC)
        # self.mosaic_conv = ConvMosaic(in_channels=1, out_channels=16, kernel_size=5, msfa_size=4,stride=1, padding=2,bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.kernel_size[0] == 7:
                    c1, c2, h, w = m.weight.data.size()
                    WB = get_WB_filter(h)
                    for i in m.parameters():
                        i.requires_grad = False
                    m.weight.data = WB.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.front_conv_input(x)
        out = self.convt_F1(x)
        out = self.convt_F2(out)
        return out

    def repeat_y(self, y):
        scale_int = math.ceil(self.scale)
        N, C, H, W = y.size()
        y = y.view(N, C, H, 1, W, 1)

        y = torch.cat([y] * scale_int, 3)
        y = torch.cat([y] * scale_int, 5).permute(0, 3, 5, 1, 2, 4)

        return y.contiguous().view(-1, C, H, W)

    def forward(self, raw, hard, pos_mat):
        x = hard
        y = raw
        WB_norelu = self.WB_Conv(x)
        # _, HW, _ =  pos_mat.size()
        # H = int(HW ** 0.5)
        # pos_mat = pos_mat.view(1, H, H, 2)
        # pos_mat = pos_mat[:, 0:4, 0:4, :]
        # pos_mat = pos_mat.contiguous().view(1, 16, 2)
        # local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        # local_weight = local_weight.view(4, 4, self.outC*5*5)
        # local_weight = local_weight.repeat(int(H/4), int(H/4), 1)
        # local_weight = local_weight.view(H*H, 400) # local_weight size :[(128*128), (5*5*16)]= [16384, 400]
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1), -1))
        up_y = self.repeat_y(y)
        cols = nn.functional.unfold(up_y, 5, padding=2)
        scale_int = math.ceil(self.scale)
        cols = cols.contiguous().view(cols.size(0) // (scale_int ** 2), scale_int ** 2, cols.size(1), cols.size(2),
                                      1).permute(0, 1, 3, 4, 2).contiguous()
        local_weight = local_weight.contiguous().view(y.size(2), scale_int, y.size(3), scale_int, -1,
                                                      self.outC).permute(1, 3, 0, 2, 4, 5).contiguous()
        local_weight = local_weight.contiguous().view(scale_int ** 2, y.size(2) * y.size(3), -1, self.outC)
        Raw_conv = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)
        Raw_conv = Raw_conv.contiguous().view(y.size(0), scale_int, scale_int, self.outC, y.size(2), y.size(3)).permute(
            0, 3, 4, 1, 5, 2)
        Raw_conv = Raw_conv.contiguous().view(y.size(0), self.outC, scale_int * y.size(2), scale_int * y.size(3))

        Raw_conv = self.relu(Raw_conv)
        convt_br1_front = self.convt_br1_front(Raw_conv)
        convt_br1_temp = self.forward_once(convt_br1_front)
        convt_br1_back = self.convt_br1_back(convt_br1_temp)
        HR_4x = convt_br1_back
        return  torch.add(HR_4x, WB_norelu)
