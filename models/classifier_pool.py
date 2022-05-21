import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import *
from pool import *

class Classifier_Pool_two(nn.Module) :
    def __init__(self, n_class) :
        super(Classifier_Pool_two,self).__init__()
        self.n_class = n_class
        for i in range(self.n_class):
            setattr(self,'conv_'+str(i+1), nn.Conv1d(320*4,1, kernel_size=1,stride = 1, padding = 0))
            
        self.input_layer = 128*self.n_class
        self.sigmoid = nn.Sigmoid()
        self.global_pool = GlobalPool(pool = 'AVG_MAX')
        for i in range(self.n_class):
            setattr(self,'attention_'+str(i+1), CAModule(320))
    def forward(self,feat_map):

        feat_res = feat_map
        logits = list()
        logits_pool = list()
        logit_maps = list()
        #attentions = list()
        # print(feat_map.shape)
        j = 29
        for i in range(self.n_class) :
            attention = getattr(self, 'attention_'+str(i+1))
            feat_map = attention(feat_map)
            logit_map = None
            #logit_map = classifier(feat_map)
            logit_maps.append(logit_map)
            #feat_map = attention(feat_map)
            feat = self.global_pool(feat_map, logit_map)
            #print(feat.shape)
            logits_pool.append(feat)
        for i in range(self.n_class) :
            attention = getattr(self, 'attention_'+str(j+1))
            feat_res = attention(feat_res)
            logit_map = None
            #logit_map = classifier(feat_map)
            #logit_maps.append(logit_map)
            #feat_map = attention(feat_map)
            feat = self.global_pool(feat_map, logit_map)
            #print(logits_pool[j] .shape)
            logits_pool[j]  = torch.cat([logits_pool[j],feat],dim = 1)
            j = j-1
        for i in range(self.n_class) :
            classifier = getattr(self, 'conv_' + str(i+1))
            logit = classifier(logits_pool[i])
            logit = logit.squeeze(-1)
            #feat = F.dropout(feat, p=self.fc_drop, training=self.training)
            logit = self.sigmoid(logit) 
            logits.append(logit)
            
        if logits[0].shape[0] == 1 :
            output = torch.cat(logits,dim = 0).t() 
        else :
            output = torch.cat(logits,dim = 1)                        
        return output
