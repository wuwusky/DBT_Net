import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import math
import os

class GroupConv(nn.Module):
    def __init__(self, in_channels, out_channels, width, num_group):
        super(GroupConv, self).__init__()
        self.num_group = num_group
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.matrix_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

        nn.init.constant_(self.matrix_conv.weight, 1.0)
        nn.init.constant_(self.matrix_conv.bias, 0.1)
        self.loss = 0
    def forward(self, x):
        channels = self.out_channels
        # matrix_act = super(GroupConv, self).forward(x) # 分组映射矩阵，核尺寸为1的卷积层
        matrix_act = self.matrix_conv(x)
        matrix_act = self.bn(matrix_act)
        matrix_act = self.relu(matrix_act)

        tmp = matrix_act + 0.001
        b, c, w, h = tmp.shape
        width = w
        tmp = tmp.view(int((b*c*w*h)/(width*width)), width*width)
        tmp = F.normalize(tmp, p=2)
        tmp = tmp.view(b, channels, width*width)
        tmp = tmp.permute(1, 0, 2)
        tmp = tmp.reshape(channels, b*w*h)

        tmp_T = tmp.transpose(1,0)
        co = tmp.mm(tmp_T)
        co = co.view(1, channels*channels)
        co = co / 128

        gt = torch.ones((self.num_group))
        gt = gt.diag()
        gt = gt.reshape((1, 1, self.num_group, self.num_group))
        gt = gt.repeat((1, int((channels/self.num_group)*(channels/self.num_group)), 1, 1))
        gt = F.pixel_shuffle(gt, upscale_factor=int(channels/self.num_group))
        gt = gt.reshape((1, channels*channels))

        loss_single = torch.sum((co-gt)*(co-gt)*0.001, dim=1)
        loss = loss_single.repeat(b)
        loss = loss / ((channels/512.0)*(channels/512.0))

        self.loss = loss
        return matrix_act

class GroupBillinear(nn.Module):
    def __init__(self, num_group, width, channels):
        super(GroupBillinear, self).__init__()
        self.num_group = num_group
        self.num_per_group = int(channels/num_group)
        self.channels = channels
        self.fc = nn.Linear(channels, channels, bias=True)
        self.bn = nn.BatchNorm2d(channels)
        # self.BL = nn.Bilinear(self.num_group, self.num_group, channels)
    def forward(self, x):
        b, c, w, h = x.shape
        width = w
        num_dim = b*c*w*h
        tmp = x.permute(0, 2, 3, 1)

        tmp = tmp.reshape(num_dim//self.channels, self.channels)
        my_tmp = self.fc(tmp)
        tmp = tmp + my_tmp

        tmp = tmp.reshape(((num_dim//self.channels), self.num_group, self.num_per_group))
        tmp_T = tmp.permute((0,2,1))


        # tmp = self.BL(tmp_T, tmp_T)
        # tmp = tmp.reshape((b, self.width, self.width, c))
        # tmp = tmp.permute((0,3,1,2))


        tmp = torch.tanh(torch.bmm(tmp_T, tmp)/32)
        tmp = tmp.reshape((b, width, width, self.num_per_group*self.num_per_group))
        # tmp = F.upsample_bilinear(tmp, (width, c))
        tmp = F.interpolate(tmp, (width, c))
        tmp = tmp.permute((0,3,1,2))


        out = x + self.bn(tmp)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.act = nn.PReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.inplaces = inplanes
        self.planes = planes
        self.conv_ch = conv1x1(inplanes, planes, stride=1)

    def forward(self, x):
        if self.inplaces != self.planes:
            identity = self.conv_ch(x)
            identity = self.bn1(identity)
            identity = self.relu(identity)
        else:
            identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_SG_GB=False, featuremap_size=0):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # self.act = nn.PReLU()
        self.downsample = downsample
        self.stride = stride
        self.use_SG_GB = use_SG_GB
        if self.use_SG_GB:
            self.SG = GroupConv(inplanes, planes, featuremap_size, 16)
            self.GB = GroupBillinear(16, featuremap_size, planes)
            self.conv1 = conv3x3(planes, planes)
        else:
            self.conv1 = conv1x1(inplanes, planes)

    def forward(self, x):
        identity = x

        if self.use_SG_GB:
            out = self.SG(x)
            out = self.GB(out)
            out = self.conv1(out)
        else:
            out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=4, zero_init_residual=True):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_sim = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.act = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.out = nn.Sequential(
            nn.Linear(512 * block.expansion, num_classes),
            nn.Sigmoid(),
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = F.dropout2d(x, p=0.25, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.out(x)

        return x

class ResNet_SG_GB(nn.Module):

    def __init__(self, block, layers, num_classes=4, zero_init_residual=True, down_1=False):
        super(ResNet_SG_GB, self).__init__()
        self.inplanes = 64
        self.featuremap_size = 224
        self.down_1 = down_1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.featuremap_size = int(self.featuremap_size * 0.5)
        # self.conv1_sim = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.act = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.featuremap_size = int(self.featuremap_size * 0.5)
        self.all_gconvs = []

        if down_1:
            self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        else:
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.featuremap_size = int(self.featuremap_size * 0.5)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.featuremap_size = int(self.featuremap_size * 0.5)
        self.layer3 = self._make_layer_SG_GB(block, 256, layers[2], stride=2)
        self.featuremap_size = int(self.featuremap_size * 0.5)
        self.layer4 = self._make_layer_SG_GB(block, 512, layers[3], stride=2)
        self.featuremap_size = int(self.featuremap_size * 0.5)

        self.SG_end = GroupConv(512 * block.expansion, 512 * block.expansion, self.featuremap_size, 32)
        self.all_gconvs.append(self.SG_end)
        self.GB_end = GroupBillinear(32, self.featuremap_size, 512 * block.expansion)
        self.bn_end = nn.BatchNorm2d(512*block.expansion)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out = nn.Sequential(
            nn.Linear(512 * block.expansion, num_classes),
            nn.Sigmoid(),
        )


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def _make_layer_SG_GB(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        my_block = block(self.inplanes, planes, stride, downsample, True, self.featuremap_size)
        layers.append(my_block)
        self.all_gconvs.append(my_block.SG)
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            my_block = block(self.inplanes, planes, 1, None, True, self.featuremap_size)
            layers.append(my_block)
            self.all_gconvs.append(my_block.SG)
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.SG_end(x)
        x = self.GB_end(x)
        x = self.bn_end(x)

        x = self.avgpool(x)
        # x = F.dropout2d(x, p=0.25, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.out(x)

        cnt = 0
        for sg in self.all_gconvs:
            if cnt == 0:
                loss = sg.loss
            else:
                loss = loss + sg.loss
            cnt = cnt + 1
        loss_sg = loss/cnt

        return x, loss_sg


# temp_model = ResNet_SG_GB(Bottleneck, (3,4,6,3), 5)

# print(temp_model.parameters)

# temp_input = torch.randn(5, 3, 224, 224)
# temp_label = torch.empty(5, dtype=torch.long).random_(3)

# loss_fun = nn.CrossEntropyLoss()
# loss_fun_2 = nn.MSELoss(reduction=False)

# temp_model.zero_grad()
# temp_out, temp_loss = temp_model(temp_input)


# loss = loss_fun(temp_out, temp_label)
# temp_loss_ = temp_loss*0
# loss_matrix = loss_fun_2(temp_loss, temp_loss_)*1e-4

# loss_all = loss + loss_matrix
# loss_all.backward()


# temp_model_SG = GroupConv(in_channels=64, out_channels=64, width=128, num_group=8)
# temp_model_GB = GroupBillinear(num_group=8, width=128, channels=64)
# temp_input = torch.randn(2, 64, 128, 128)
# temp_out = temp_model_SG(temp_input)
# out = temp_model_GB(temp_out)

# print('ok')

