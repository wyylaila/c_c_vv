import torch
import torch.nn as nn
import torch.nn.functional as F

from .pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .decoders import EMCAD
from .RMT import VisRetNet



class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels=9, kernel_size=3, upsampling=2):  # 修改 upsampling 参数为2
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling)  # 使用双线性插值上采样
        super().__init__(conv2d, upsampling)


class RMEUNet(nn.Module):
    def __init__(self, num_classes=9, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True, add=True, lgag_ks=3,
                 activation='relu', encoder='resnet50', pretrain=True):
        super(RMEUNet, self).__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        # backbone network initialization with pretrained weight
        if encoder == 'pvt_v2_b0':
            self.backbone = pvt_v2_b0()
            path = './pretrained_pth/pvt/pvt_v2_b0.pth'
            channels = [256, 160, 64, 32]
        elif encoder == 'pvt_v2_b1':
            self.backbone = pvt_v2_b1()
            path = './pretrained_pth/pvt/pvt_v2_b1.pth'
            channels = [512, 320, 128, 64]
        elif encoder == 'pvt_v2_b2':
            self.backbone = pvt_v2_b2()
            path = './pretrained_pth/pvt/pvt_v2_b2.pth'
            channels = [512, 320, 128, 64]
        elif encoder == 'pvt_v2_b3':
            self.backbone = pvt_v2_b3()
            path = './pretrained_pth/pvt/pvt_v2_b3.pth'
            channels = [512, 320, 128, 64]
        elif encoder == 'pvt_v2_b4':
            self.backbone = pvt_v2_b4()
            path = './pretrained_pth/pvt/pvt_v2_b4.pth'
            channels = [512, 320, 128, 64]
        elif encoder == 'pvt_v2_b5':
            self.backbone = pvt_v2_b5()
            path = './pretrained_pth/pvt/pvt_v2_b5.pth'
            channels = [512, 320, 128, 64]
        elif encoder == 'resnet18':
            self.backbone = resnet18(pretrained=pretrain)
            channels = [512, 256, 128, 64]
        elif encoder == 'resnet34':
            self.backbone = resnet34(pretrained=pretrain)
            channels = [512, 256, 128, 64]
        elif encoder == 'resnet50':
            self.backbone = resnet50(pretrained=pretrain, pretrained_path='./REUNet/lib/resnet50-19c8e357.pth')
            channels = [2048, 2048, 1024, 512, 256]
        elif encoder == 'resnet101':
            self.backbone = resnet101(pretrained=pretrain)
            channels = [2048, 1024, 512, 256]
        elif encoder == 'resnet152':
            self.backbone = resnet152(pretrained=pretrain)
            channels = [2048, 1024, 512, 256]
        else:
            print('Encoder not implemented! Continuing with default encoder pvt_v2_b2.')
            self.backbone = pvt_v2_b2()
            path = './pretrained_pth/pvt/pvt_v2_b2.pth'
            channels = [512, 320, 128, 64]

        if pretrain == True and 'pvt_v2' in encoder:
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)


        #   decoder initialization
        self.decoder = EMCAD(channels=channels, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                             dw_parallel=dw_parallel, add=add, lgag_ks=lgag_ks, activation=activation)


        self.out_head1 = nn.Conv2d(channels[0], num_classes, 1)
        self.out_head2 = nn.Conv2d(channels[1], num_classes, 1)
        self.out_head3 = nn.Conv2d(channels[2], num_classes, 1)
        self.out_head4 = nn.Conv2d(channels[3], num_classes, 1)
        self.out_head5 = nn.Conv2d(channels[4], num_classes, 1)
        self.rmt = VisRetNet()







        self.segmentation_head = SegmentationHead(
            in_channels=channels[4],
            out_channels=9,
            kernel_size=3,
        )

    def forward(self, x, mode='test'):

        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)

        # encoder
        x1, x2, x3, x4 = self.backbone(x)

        x5 = self.rmt(x4)


        dec_outs = self.decoder(x5, [x4, x3, x2, x1])

        p = dec_outs

        p = F.interpolate(p, scale_factor=2, mode='bilinear')



        logits = self.segmentation_head(p)

        if mode == 'test':
            return logits

        return logits

