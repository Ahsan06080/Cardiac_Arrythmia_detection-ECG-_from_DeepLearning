import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import *
from models.resnet import *

class ResidualNetwork(nn.Module) :
    def __init__(self, n_filters_in,config,
                 dropout_keep_prob=0.8, kernel_size=17, preactivation=True,
                 postactivation_bn=False, activation_function='relu', last_layer = 'sigmoid', num_classes = 30):
        super(ResidualNetwork, self).__init__()
        self.config =config  
        self.n_filters_in = n_filters_in
        self.dropout_keep_prob = dropout_keep_prob
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function
        self.last_layer = last_layer
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        
        self.conv_1d = nn.Conv1d(n_filters_in,64, kernel_size=self.kernel_size, padding = 8)
        self.batch_norm_1d = nn.BatchNorm1d(num_features = 64)
        self.residual_unit_1 = ResidualUnit(2,64,128)
        self.residual_unit_2 = ResidualUnit(2,128,196)
        self.residual_unit_3 = ResidualUnit(2,196,256)
        self.residual_unit_4 = ResidualUnit(2,256,320)
        self.attention = MultiHeadAttention(self.config)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(320, 30)
        
    def forward(self, x) :
        
        x = self.conv_1d(x)
        x = self.batch_norm_1d(x)
        x = eval(f'F.{self.activation_function}(x)' ) 
        x,y = self.residual_unit_1(x,x)
        x,y = self.residual_unit_2(x,y)
        x,y = self.residual_unit_3(x,y)
        x,_ = self.residual_unit_4(x,y)
        
        x = self.attention(x)
        
        x = self.pool(x).squeeze()
        #print(x.shape)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x