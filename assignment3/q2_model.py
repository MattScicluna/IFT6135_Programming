from torch import nn


class Critic(nn.Module):
    def __init__(self, h_dim=64):
        super(Critic, self).__init__()

        x = [nn.Conv2d(3, h_dim, 4, 2, 1),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(h_dim, 2*h_dim, 4, 2, 1),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(2*h_dim, 4*h_dim, 4, 2, 1),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(4*h_dim, 1, 4, 1, 0)]

        self.x = nn.Sequential(*x)

    def forward(self, x):
        return self.x(x).squeeze()


class Generator(nn.Module):
    def __init__(self, z_dim=100, h_dim=64):
        super(Generator, self).__init__()

        decoder = [nn.ConvTranspose2d(z_dim, 4*h_dim, 4, 1, 0),
                   nn.BatchNorm2d(4*h_dim),
                   nn.ReLU(True),
                   nn.ConvTranspose2d(4*h_dim, 2*h_dim, 4, 2, 1),
                   nn.BatchNorm2d(2*h_dim),
                   nn.ReLU(True),
                   nn.ConvTranspose2d(2*h_dim, h_dim, 4, 2, 1),
                   nn.BatchNorm2d(h_dim),
                   nn.ReLU(True),
                   nn.ConvTranspose2d(h_dim, 3, 4, 2, 1),
                   nn.Tanh()
                   ]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, z):
        return self.decoder(z.view(z.shape[0], z.shape[1], 1, 1))
