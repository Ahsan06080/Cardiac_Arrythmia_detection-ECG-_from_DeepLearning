import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import *
from models.resnet import *       
import math  as math
from models.densenet import *  
from models.classifier_pool import *

class DenseNet_mha(nn.Module):
    def __init__(self, depth, num_classes,config, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0):
        super(DenseNet_mha, self).__init__()
        self.num_classes = num_classes
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/4
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv1d(12, in_planes, kernel_size=17, stride=1,
                               padding=8, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)),compression = 2, dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)),compression = 2, dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans3 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)),compression = 2, dropRate=dropRate)
        in_planes = int(in_planes*reduction)
        self.block4 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.residual_unit = ResidualUnit(2,183,320)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(320, self.num_classes)
        self.in_planes = in_planes
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.attention = MultiHeadAttention(config)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0]  * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        # print('conv1')
        # print(out.shape)
        out = self.trans1(self.block1(out))
        # print('trans1')
        # print(out.shape)
        out = self.trans2(self.block2(out))
        # print('trans2')
        # print(out.shape)
        out = self.trans3(self.block3(out))
        # print('trans3')
        # print(out.shape)
        out = self.block4(out)
        # print('block 4')
        # print(out.shape)
        out = self.relu(self.bn1(out))
        out,_ = self.residual_unit(out,out)
        out = self.attention(out)
        out = self.pool(out)
        out = out.view(-1,320)
        out = self.fc (out)
        out = torch.sigmoid(out)
         # print(out.shape)
        
        return out
