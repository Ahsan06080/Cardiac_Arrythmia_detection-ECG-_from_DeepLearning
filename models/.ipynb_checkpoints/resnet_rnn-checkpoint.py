import torch
import torch.nn as nn
import torch.nn.functional as F
from  models.attention import *
from  models.resnet import *
from  models.rnn_module import *

class ResidualNetwork_lstm(nn.Module) :
    def __init__(self, n_filters_in,n_class = 30,
                 dropout_keep_prob=0.8, kernel_size=17, preactivation=True,
                 postactivation_bn=False, activation_function='relu', last_layer = 'sigmoid', num_classes = 30,bidirectional = True, rnn = 'lstm'):
        super(ResidualNetwork_lstm, self).__init__()
          
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
        self.lstm = LSTM_module(320, rnn = rnn,bidirectional = bidirectional)
        
    def forward(self, x) :
        
        x = self.conv_1d(x)
        x = self.batch_norm_1d(x)
        x = eval(f'F.{self.activation_function}(x)' ) 
        x,y = self.residual_unit_1(x,x)
        x,y = self.residual_unit_2(x,y)
        x,y = self.residual_unit_3(x,y)
        x,_ = self.residual_unit_4(x,y)
        #print(x.shape)
        x = self.lstm(x)
        
        return x