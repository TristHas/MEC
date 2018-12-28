import torch.nn as nn

def conv_block(in_channels, out_channels, bn=True):
    """
    """
    layers = [nn.Conv2d(in_channels, out_channels, 3, padding=1)]
    if bn:
        layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.ReLU()]
    return nn.Sequential(*layers)
    
class ConvMod(nn.Module):
    def __init__(self, in_channels, out_channels, depth=0, pool=2, bn=True, residual=False):
        """
            Creates one convolutional block.
            A convolutional block is made by a succession of
            elementary sequence (2D-convolution -> Batch Norm -> ReLU)
            A conv_block repeat this sequence *depth* times
            followed by a 2x2 Max Pooling at the end.
            _________________________________________
            Parameters:
                int in_channels
                int out_channels
                int depth
        """
        super(ConvMod, self).__init__()
        self.residual = residual
        if in_channels == out_channels:
            self.inp_residual = True
        else:
            self.inp_residual = False
        self.l1 = conv_block(in_channels, out_channels, bn)
        layers = [conv_block(out_channels, out_channels, bn) for i in range(depth)]
        self.layers = nn.Sequential(*layers)
        self.pooling= nn.MaxPool2d(pool) if pool else lambda x:x

    def forward(self, inp):
        """
        """
        tmp = self.l1(inp)
        x = self.layers(tmp)
        if self.residual:
            if self.inp_residual:
                x+=inp
            else:
                x+=tmp
        return self.pooling(x)

class CNN(nn.Module):
    """
    """    
    def __init__(self, in_dim=3,  hid_dim=64, nclass=14, 
                       nlayer=4, blk_depth=0, bn=True, 
                       residual=True, mode="classification"):
        self.mode = mode
        super(CNN, self).__init__()
        layers = [ConvMod(in_dim, hid_dim,  blk_depth, 2, bn, residual)]
        layers+= [ConvMod(hid_dim, hid_dim, blk_depth, 2, bn, residual) for i in range(nlayer-1)]
        layers+= [nn.Conv2d(hid_dim, nclass, 1, bias=False)]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        out = self.encoder(x)
        if self.mode == "classification":
            return out.mean((2,3))
        else:
            raise NotImplementedError