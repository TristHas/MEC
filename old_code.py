def conv_block(in_channels, out_channels, depth=0):
    '''
        Creates one convolutional block.
        A convolutional block is made by a succession of
        elementary sequence (2D-convolution -> Batch Norm -> ReLU)
        A conv_block repeat this sequence *depth* times
        followed by a 2x2 Max Pooling at the end.
        _________________________________________
        Input:
            int in_channels
            int out_channels
            int depth
        _________________________________________
        Output:
            nn.Module convolutional block
    '''
    layers = [[nn.Conv2d(in_channels, out_channels, 3, padding=1),
               nn.BatchNorm2d(out_channels),
               nn.ReLU()]]
    layers += [[nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()] for i in range(depth)]
    layers += [[nn.MaxPool2d(2)]]
    return nn.Sequential(*chain(*layers))
    
class Model(nn.Module):
    '''
    '''
    def __init__(self, in_dim=1, hid_dim=64, nclass= 14, 
                       nlayers=2, blk_depth=0):
        super(Model, self).__init__()
        layers = [conv_block(in_dim, hid_dim,  blk_depth)]
        layers+= [conv_block(hid_dim, hid_dim, blk_depth) for i in range(nlayers)]
        layers+= [conv_block(hid_dim, nclass)]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x).mean((2,3))