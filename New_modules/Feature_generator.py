import torch
import torch.nn as nn


class Feature_Generator(nn.Module):
    def __init__(self, blocknum = 3, in_channels = 512):
        super(Feature_Generator, self).__init__()

        resblock_list = []
        for i in range(blocknum):
            resblock_list.append(ResBlock(in_channels = in_channels))

        self.resblock = nn.Sequential(*resblock_list)

    def forward(self, x, sub):
        input = torch.cat((x,sub),dim=1)

        out = self.resblock(input)
        refined_feature = torch.split(out,[1,1],dim=1)

        return refined_feature


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        self.layer1 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels, kernel_size=3, padding=1)
        self.layer2 = nn.Conv2d(in_channels = in_channels, out_channels=in_channels, kernel_size=3, padding=1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out += x

        return out


