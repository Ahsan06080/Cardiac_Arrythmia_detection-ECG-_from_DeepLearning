import torch.nn as nn
from xresnet1d import *
from pool import *
from classifier_pool import *
        
class xResPool(nn.Module) :
    def __init__(self,n_classes):
        super(xResPool,self).__init__()

        self.n_classes = n_classes
        self.model = nn.Sequential(*list(xresnet1d101().children())[:-1])
        self.classifier = Classifier_Pool_two(30)

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)

        return x