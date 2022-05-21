import torch.nn as nn
from models.resnet import *
from models.pool import *
from models.classifier_pool import *
class ResidualNetwork_classifier(nn.Module) :
    def __init__(self, n_filters_in,n_class = 30,
                 dropout_keep_prob=0.8, kernel_size=17, preactivation=True,
                 postactivation_bn=False, activation_function='relu', last_layer = 'sigmoid'):
        super(ResidualNetwork_classifier, self).__init__()
          
        self.n_filters_in = n_filters_in
        self.dropout_keep_prob = dropout_keep_prob
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function
        self.last_layer = last_layer
        self.kernel_size = kernel_size
        self.n_class = n_class
        
        self.conv_1d = nn.Conv1d(n_filters_in,64, kernel_size=self.kernel_size, padding = 8)
        self.batch_norm_1d = nn.BatchNorm1d(num_features = 64)
        self.residual_unit_1 = ResidualUnit(4,64,128)
        self.residual_unit_2 = ResidualUnit(4,128,196)
        self.residual_unit_3 = ResidualUnit(4,196,256)
        self.residual_unit_4 = ResidualUnit(4,256,320)
        self.classifier = Classifier_Pool_two(self.n_class)
        #self.linear = nn.Linear(5120, 9)
        #self.Sigmoid = nn.Sigmoid()
        
    def forward(self, x) :
        
        x = self.conv_1d(x)
        x = self.batch_norm_1d(x)
        x = eval(f'F.{self.activation_function}(x)' ) 
        x,y = self.residual_unit_1(x,x)
        x,y = self.residual_unit_2(x,y)
        x,y = self.residual_unit_3(x,y)
        x,_ = self.residual_unit_4(x,y)
        x = self.classifier(x)
        
        return x