import torch
import torch.nn as nn

# Conv Eq
# (input + 2*pad - kernel_size / stride) + 1

class InputLayer(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        self.conv_layer = nn.Conv2d(in_channels=in_channels, 
                                out_channels=64, 
                                kernel_size=9, 
                                padding=4)
        
        self.prelu = nn.PReLU(num_parameters=64)
        
        self.input_layer = nn.Sequential(self.conv_layer, 
                                        self.prelu)
        
    def forward(self, x: torch.Tensor):
        return self.input_layer(x)
    
class ResConvBlock(nn.Module):
    def __init__(self, in_channel=3):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64), # same as output channel
            nn.PReLU(num_parameters=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.PReLU(num_parameters=64),
        )
        
    def forward(self, x: torch.Tensor):
        return x + self.conv_block(x)
    
    
class ResidualBlock(nn.Module):
    def __init__(self, num_layers=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.residual_blocks = nn.Sequential(
            *[ResConvBlock() for _ in range(num_layers)],
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
    def forward(self, x: torch.Tensor):
        return x + self.residual_blocks(x)
    
    
class UpSampleLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.upsample_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.PixelShuffle(2),
            nn.PReLU(64), # output channel size will rearrage to H.W
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.PixelShuffle(2),
            nn.PReLU(64),
        )
        
        self.final_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, padding=4, stride=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.upsample_layer(x)
    
    
class Generator(nn.Module):
    def __init__(self, in_channels=3, img_size=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.generator = nn.Sequential(
            InputLayer(in_channels=in_channels),
            ResidualBlock(),
            UpSampleLayer(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, padding=4, stride=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.generator(x)
        
        
        
        
    
        
    
    

def test():
    test_input = torch.randn(1, 3, 128, 128)
    
    # # Input layer unit testing
    # input_layer = InputLayer()
    # output = input_layer(test_input)
    # print(f"Input Layer is Working , shape of output tensor is {output.size()}")
    
    # # ResConvblock unit testing
    # conv_block = ResConvBlock()
    # output = conv_block(output)
    # print(f"ResConvBlock is Working , shape of output tensor is {output.size()}")
    
    #  # ResidualBlock unit testing
    # residual_block = ResidualBlock()
    # output = residual_block(output)
    # print(f"ResidualBlock is Working , shape of output tensor is {output.size()}")
    
    # # upsample + final_layer unit testing
    # residual_block = UpSampleLayer()
    # output = residual_block(output)
    # print(f"ResidualBlock is Working , shape of output tensor is {output.size()}")
    
    # Generator
    generator = Generator()
    output = generator(test_input)
    print(f"generator is Working ,input shape :{test_input.size()}, output shape: {output.size()}")
    
    
    
    
    
    
if __name__ == "__main__":
    test()
    
        


