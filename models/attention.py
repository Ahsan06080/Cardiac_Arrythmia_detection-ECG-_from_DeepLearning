import torch
import torch.nn as nn
import torch.nn.functional as F

class CAModule(nn.Module):
    """##Squeeze and excite CAM
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
    *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    code reference:
    https://github.com/kobiso/CBAM-keras/blob/master/models/attention_module.py
    """

    def __init__(self, num_channels, reduc_ratio=2):
        super(CAModule, self).__init__()
        self.num_channels = num_channels
        self.reduc_ratio = reduc_ratio

        self.fc1 = nn.Linear(num_channels, num_channels // reduc_ratio,
                             bias=True)
        self.fc2 = nn.Linear(num_channels // reduc_ratio, num_channels,
                             bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_map):
        # attention branch--squeeze operation
        gap_out = feat_map.view(feat_map.size()[0], self.num_channels,
                                -1).mean(dim=2)
        #print(gap_out.shape)
        # attention branch--excitation operation
        fc1_out = self.relu(self.fc1(gap_out))
        fc2_out = self.sigmoid(self.fc2(fc1_out))
        #print(fc2_out.shape)
        # attention operation
        fc2_out = fc2_out.view(fc2_out.size()[0], fc2_out.size()[1], 1)
        feat_map = torch.mul(feat_map, fc2_out)

        return feat_map



class AttentionHead(nn.Module) :
    def __init__(self, embed_dim, head_dim) :
        super().__init__()

        self.quary = nn.Linear(embed_dim, head_dim)
        self.key = nn.Linear(embed_dim, head_dim)
        self.value =  nn.Linear(embed_dim, head_dim)

    def scaled_dot_product_attention(self,q,k,v) :

        scores = torch.bmm(q,k.transpose(1,2))
        weights = F.softmax(scores,dim = -1)

        return torch.bmm(weights,v)


    def forward(self, x):
        q = self.quary(x)
        k = self.key(x)
        v = self.value(x)

        outputs = self.scaled_dot_product_attention(q, k, v)

        return outputs

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        hidden_state =  torch.permute(hidden_state,(0,2,1))
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        x =  torch.permute(x,(0,2,1))
        return x