
import torch
import torch.nn as nn
class LSTM_module(nn.Module):
    def __init__(self, input_dim, rnn = 'lstm', num_classes=30, hidden_dim=512, n_layers=1,bidirectional = True,):
        super(LSTM_module,self).__init__()

        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_state = None
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers  
        self.bidirectional = bidirectional
        if rnn == 'lstm' :
            self.lstm_layer = nn.LSTM(self.input_dim, self.hidden_dim, self.n_layers,bidirectional = self.bidirectional, batch_first=True)
        elif rnn == 'gru' :
            self.lstm_layer = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers, bidirectional = self.bidirectional,batch_first=True)
        else :
            raise Exception(f"Sorry, unknown {rnn}, use either lstm or gru")
        self.output_layer = nn.Sequential(
                nn.Linear(2* self.hidden_dim if bidirectional else hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim, momentum=0.01),
                nn.ReLU(),
                nn.AlphaDropout(0.4),
                nn.Linear(hidden_dim, num_classes),
                nn.Sigmoid(),
            )
    def forward(self,x):
        x = torch.permute(x, (0, 2, 1))
        x,_ = self.lstm_layer(x,self.hidden_state)
        x = x[:, -1]
        x = self.output_layer(x)
        return x