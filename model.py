# Importing the libraries
import torch.nn as nn

# Custom reshape layer
class Reshape(nn.Module):
    def __init__(self, *args):
        
        # Inherits all that good stuff from nn.Module
        super(Reshape, self).__init__()

        # Shape is a tuple of integers
        self.shape = args

    def forward(self, x):
        # Returns a view of the input tensor
        return x.view(self.shape)
    

# Generator
class Generator(nn.Module):
    def __init__(self, coding_size=100, n_filters=32):

        # Inherits all that good stuff from nn.Module
        super(Generator, self).__init__()

        # Defining the layers
        self.layer_stack = nn.Sequential(
            Reshape(-1, coding_size, 1, 1), # (N, 100) -> (N, 100, 1, 1)

            nn.ConvTranspose2d(coding_size, n_filters * 4, kernel_size=4, stride=1, padding=0, bias=False), # (N, 100) -> (N, 256, 4, 4)
            nn.BatchNorm2d(n_filters * 4),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(n_filters * 4, n_filters * 2, kernel_size=3, stride=2, padding=1, bias=False), # (N, 256, 4, 4) -> (N, 128, 8, 8)
            nn.BatchNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(n_filters * 2, n_filters, kernel_size=4, stride=2, padding=1, bias=False), # (N, 128, 8, 8) -> (N, 64, 16, 16)
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(n_filters, 1, kernel_size=4, stride=2, padding=1, bias=False), # (N, 64, 16, 16) -> (N, 1, 28, 28)
            nn.Tanh()
        )

    def forward(self, x):
        # Returns the output of the layer stack
        return self.layer_stack(x)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, n_filters=32):

        # Inherits all that good stuff from nn.Module
        super(Discriminator, self).__init__()
        
        # Defining the layers
        self.layer_stack = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=4, stride=2, padding=1, bias=False), # (N, 1, 28, 28) -> (N, 64, 14, 14)
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters, n_filters * 2, kernel_size=4, stride=2, padding=1, bias=False), # (N, 64, 14, 14) -> (N, 128, 7, 7)
            nn.BatchNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=3, stride=2, padding=1, bias=False), # (N, 128, 7, 7) -> (N, 256, 4, 4)
            nn.BatchNorm2d(n_filters * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters * 4, 1, kernel_size=4, stride=1, padding=0, bias=False), # (N, 256, 4, 4) -> (N, 1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, x):
        # Returns the output of the layer stack
        return self.layer_stack(x)