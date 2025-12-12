import torch 
from torch import nn

class senet(nn.Module):
    def __init__(self, channel, ratio = 16):
        super(senet, self).__init__()
        self.avg_poll = nn.AdaptiveAvgPool2d(1)
        self.fc       = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False),
            nn.Sigmoid(),
        )


    def forward(self, x):
        b, c, h, w = x.size()
        #b,c,h,w -> b,c,1,1
        avg = self.avg_poll(x).view([b, c])
        fc = self.fc(avg).view([b, c, 1, 1])

        return x * fc

model = senet(512)    
print(model)
inputs = torch.ones([2, 512, 26, 26])

outputs = model(inputs)