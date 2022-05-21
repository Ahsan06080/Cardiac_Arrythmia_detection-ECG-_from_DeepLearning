class xResPool(nn.Module) :
    def __init__(self,n_classes = 30, trancated = True):
        super(xResPool,self).__init__()
        self.trancated = trancated
        self.n_classes = n_classes
        if self.trancated :
            self.model = nn.Sequential(*list(xresnet1d101().children())[:-2])
        else :
            self.model = nn.Sequential(*list(xresnet1d101().children())[:-1])
        self.classifier = LSTM_module(256)

    def forward(self, x):
        x = self.model(x)
        #print(x.shape)
        x = self.classifier(x)

        return x