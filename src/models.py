import torch
import torch.nn as nn
from models.mcnnpytorch.src.network import Conv2d

class MCNN(nn.Module):
    '''
    Multi-column CNN 
    '''
    
    def __init__(self, bn=False):
        super(MCNN, self).__init__()
        
        self.branch1 = nn.Sequential(Conv2d( 1, 16, 9),
                                     nn.MaxPool2d(2),
                                     Conv2d(16, 32, 7),
                                     nn.MaxPool2d(2),
                                     Conv2d(32, 16, 7),
                                     Conv2d(16,  8, 7))
        
        self.branch2 = nn.Sequential(Conv2d( 1, 20, 7),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5),
                                     nn.MaxPool2d(2),
                                     Conv2d(40, 20, 5),
                                     Conv2d(20, 10, 5))
        
        self.branch3 = nn.Sequential(Conv2d( 1, 24, 5),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3),
                                     nn.MaxPool2d(2),
                                     Conv2d(48, 24, 3),
                                     Conv2d(24, 12, 3))
        
        self.fuse = nn.Sequential(Conv2d( 30, 1, 1))
        
    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1,x2,x3),1)
        x = self.fuse(x)
        
        return x