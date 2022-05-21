import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import *

class SkipConnection(nn.Module) :
    def __init__(self, down_sample, n_filters_in,n_filters_out):
        super(SkipConnection, self).__init__()
        
        self.down_sample = down_sample
        self.n_filters_in = n_filters_in
        self.n_filters_out = n_filters_out
        self.maxpool_1d = nn.MaxPool1d(self.down_sample, stride=self.down_sample)
        self.conv_1d = nn.Conv1d(self.n_filters_in, self.n_filters_out, kernel_size=1, stride=1)
    def forward(self, y) :
        if self.down_sample > 1:
            y = self.maxpool_1d(y)
        elif self.down_sample == 1:
            y = y
        else:
            raise ValueError("Number of samples should always decrease.")
        # Deal with n_filters dimension increase
        if self.n_filters_in != self.n_filters_out:
            # This is one of the two alternatives presented in ResNet paper
            # Other option is to just fill the matrix with zeros.
            y = self.conv_1d(y)
        return y
    
    
    
class BatchNorm_PlusActivision(nn.Module) :
    def __init__(self, n_filters_in, activision_function = 'relu',postactivation_bn = 'True'):
        super(BatchNorm_PlusActivision, self).__init__()
        
        self.activision_function = activision_function
        self.postactivation_bn =   postactivation_bn  
        self.n_filters_in = n_filters_in
    
        self.batch_norm_1d =nn.BatchNorm1d(num_features = self.n_filters_in)
    def forward(self, x) :
        if self.postactivation_bn : 
            x = eval(f'F.{self.activision_function}(x)' )
            x = self.batch_norm_1d(x)
        else :
            x = self.batch_norm_1d(x)
            x = eval(f'F.{self.activision_function}(x)' )
        return x
    
    
class ResidualUnit(nn.Module):
    def __init__(self,down_sample, n_filters_in, n_filters_out, 
                 dropout_keep_prob=0.8, kernel_size=17, preactivation=True,
                 postactivation_bn=False, activation_function='relu', last_layer = 'sigmoid'):
        super(ResidualUnit, self).__init__()
                
        self.down_sample = down_sample    
        self.n_filters_in = n_filters_in
        self.n_filters_out = n_filters_out
        self.dropout_rate = 1 - dropout_keep_prob
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function
        self.last_layer = last_layer
        self.kernel_size = kernel_size
        
        
        
        self.conv_1d_1 = nn.Conv1d(self.n_filters_in, self.n_filters_out, kernel_size=self.kernel_size, padding = 8)
        self.conv_1d_2 = nn.Conv1d(self.n_filters_out, self.n_filters_out, kernel_size=self.kernel_size, stride = self.down_sample,padding = 8)
        self.skip_connection = SkipConnection(self.down_sample, self.n_filters_in,self.n_filters_out)
        self.batch_norm_plus_activision = BatchNorm_PlusActivision(self.n_filters_out, self.activation_function, self.postactivation_bn)
        self.dropout =  nn.Dropout(self.dropout_rate)
        
    def forward(self, x, y):
        y = self.skip_connection(y)
        x = self.conv_1d_1(x)
        x = self.conv_1d_2(x)
        if self.dropout_rate > 0 :
            x = self.dropout(x)
        if self.preactivation:
            x = torch.add(x,y)
            y = x
            x = self.batch_norm_plus_activision(x)
            if self.dropout_rate > 0 :
                 x = self.dropout(x)
        else:
            x = self.batch_norm_plus_activision(x)
            x = torch.add(x,y)
            x = eval(f'F.{self.activision_function}(x)' )
            if self.dropout_rate > 0 :
                    x = self.dropout(x)
            y = x
        return x,y
    
    
class ResidualNetwork(nn.Module) :
    def __init__(self, n_filters_in,
                 dropout_keep_prob=0.8, kernel_size=17, preactivation=True,
                 postactivation_bn=False, activation_function='relu', last_layer = 'sigmoid', num_classes = 30):
        super(ResidualNetwork, self).__init__()
          
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
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(320, self.num_classes)
        
    def forward(self, x) :
        
        x = self.conv_1d(x)
        x = self.batch_norm_1d(x)
        x = eval(f'F.{self.activation_function}(x)' ) 
        x,y = self.residual_unit_1(x,x)
        x,y = self.residual_unit_2(x,y)
        x,y = self.residual_unit_3(x,y)
        x,_ = self.residual_unit_4(x,y)
        
        x = self.pool(x).squeeze()
        #print(x.shape)
        x = self.linear(x)
        
        return x