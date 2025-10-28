import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_batchnorm=False, **kwargs):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size, 
                  stride=stride, 
                  padding=padding)
            )
        
        if use_batchnorm:
            self.convblock.append(nn.BatchNorm2d(out_channels))
            
        self.convblock.append(nn.LeakyReLU())
    
    def forward(self, x):
        return self.convblock(x)
    


class Descriminator(nn.Module):
    def __init__(self,in_channels=3):
        super().__init__()
        
        self.first_layer = ConvBlock(in_channels=in_channels, 
                                out_channels=64, 
                                kernel_size=3, 
                                stride=1, 
                                padding=1,
                                use_batchnorm=False)
        
        
        self.hidden_layers = nn.Sequential(
            ConvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=2, use_batchnorm=True)    
        )
        
        in_channels = 64
        for out_channels in [128, 256, 512]:
            for stride in [1, 2]:
                self.hidden_layers.append(
                    ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride),
                )  
                
                in_channels = out_channels 
        
        self.final_layers = nn.Sequential(
            nn.Flatten(-3),
            nn.Linear(in_features=8192, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=1)
        )
        
        self.descriminator = nn.Sequential(self.first_layer,
                      self.hidden_layers,
                      self.final_layers)
        
    def forward(self, x):
        return self.descriminator(x)
            
            
            
def test():
    x = torch.randn(6, 3, 64, 64)
    des = Descriminator()
    out = des(x)
    
    print(f"shape {out.size()}")
    
    
if __name__ == "__main__":
    test()
        