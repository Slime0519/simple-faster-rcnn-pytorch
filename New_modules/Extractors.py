from __future__ import absolute_import
import torch
import torch.nn as nn
import torchvision.models.vgg as refvgg
import torch.hub as hub
from torchvision.models import vgg16
from torchvision.ops import RoIPool

from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from utils import array_tool as at
from utils.config import opt

model_urls = refvgg.model_urls
cfgs= refvgg.cfgs

def construct_backbone(pretrained = True, progress = False):
    return vggbackbone('vgg16_bn', 'D', True, pretrained, progress)

class vggbackbone(nn.Module):
    def __init__(self, pre_layers, post_layers, init_weights=True):
        super(vggbackbone, self).__init__()
        self.pre_layers =pre_layers
        self.post_layers= post_layers
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        subfeatures = self.pre_layers(x)
        postfeatures = self.post_layers(subfeatures)

        return subfeatures, postfeatures

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class TargetExtractor(vggbackbone):
    def __init__(self):
        super(TargetExtractor, self).__init__()

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.post_layers(x)
        return x


def makeextractor(cfg, batch_norm,progress,  pretrained = True, arch = 'vgg16_bn',**kwargs ):
    if pretrained:
        kwargs['init_weights'] = False
    layersetup = cfgs[cfg]
    cfg_presub =[]

    checksum =0
    while not checksum:
        if layersetup[0] =='M':
            checksum =True
        cfg_presub.append(cfg.pop(0))
    cfg_postsub = cfg
    layerlist_presub = refvgg.make_layers(cfg_presub)
    layerlist_postsub = refvgg.make_layers(cfg_postsub)

    model = vggbackbone(refvgg.make_layers(layerlist_presub,layerlist_postsub, batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = hub.load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict= False)
    return model


