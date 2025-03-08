import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from ldm.modules.diffusionmodules.util import conv_nd, batch_norm_nd


class UnetConv(nn.Module):
    def __init__(self, in_size, out_size, is_separate_batchnorm,
                 kernel_size=3, padding_size=1, init_stride=1, dims=3):
        super(UnetConv, self).__init__()

        if is_separate_batchnorm:
            self.conv1 = nn.Sequential(OrderedDict([
                ('conv', conv_nd(dims, in_size, out_size, kernel_size, init_stride, padding_size)),
                ('bn', batch_norm_nd(dims, out_size)),
                ('nl', nn.ReLU(inplace=True)),
            ]))
            self.conv2 = nn.Sequential(OrderedDict([
                ('conv', conv_nd(dims, out_size, out_size, kernel_size, init_stride, padding_size)),
                ('bn', batch_norm_nd(dims, out_size)),
                ('nl', nn.ReLU(inplace=True)),
            ]))
        else:
            self.conv1 = nn.Sequential(OrderedDict([
                ('conv', conv_nd(dims, in_size, out_size, kernel_size, init_stride, padding_size)),
                ('nl', nn.ReLU(inplace=True)),
            ]))
            self.conv2 = nn.Sequential(OrderedDict([
                ('conv', conv_nd(dims, out_size, out_size, kernel_size, init_stride, padding_size)),
                ('nl', nn.ReLU(inplace=True)),
            ]))

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetUpConcat(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True, dims=3):
        super(UnetUpConcat, self).__init__()
        self.conv = UnetConv(in_size + out_size, out_size, is_batchnorm, dims=dims)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        out1 = self.conv(torch.cat([outputs1, outputs2], 1))
        return out1
    

class UNetModel(nn.Module):
    def __init__(self, 
                 input_channels=16,
                 dims=3,):
        super(UNetModel, self).__init__()
        self.in_channels = input_channels
        self.is_batchnorm = True
        self.dims = dims

        filters = [16 * 2 ** x for x in range(5)]
        max_pool_nd = getattr(nn, f"MaxPool{self.dims}d", nn.Identity)

        # downsampling
        self.conv1 = UnetConv(self.in_channels, filters[0], self.is_batchnorm, dims=dims)
        self.maxpool1 = max_pool_nd(kernel_size=2)
        
        self.conv2 = UnetConv(filters[0], filters[1], self.is_batchnorm, dims=dims)
        self.maxpool2 = max_pool_nd(kernel_size=2)
        
        self.conv3 = UnetConv(filters[1], filters[2], self.is_batchnorm, dims=dims)
        self.maxpool3 = max_pool_nd(kernel_size=2)
        
        self.conv4 = UnetConv(filters[2], filters[3], self.is_batchnorm, dims=dims)
        self.maxpool4 = max_pool_nd(kernel_size=2)

        self.center = UnetConv(filters[3], filters[4], self.is_batchnorm, dims=dims)

        # upsampling
        self.up_concat4 = UnetUpConcat(filters[4], filters[3], self.is_batchnorm, dims=dims)
        self.up_concat3 = UnetUpConcat(filters[3], filters[2], self.is_batchnorm, dims=dims)
        self.up_concat2 = UnetUpConcat(filters[2], filters[1], self.is_batchnorm, dims=dims)
        self.up_concat1 = UnetUpConcat(filters[1], filters[0], self.is_batchnorm, dims=dims)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        
        self.out_ch = filters[0]

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
       
        center = self.center(maxpool4)
        center = self.dropout1(center)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        
        out = self.dropout2(up1)
        return out