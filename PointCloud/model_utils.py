import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from pytorch3d.transforms import quaternion_raw_multiply
from dq_func import dualquat_multiply

class TNet(nn.Module):
    ''' T-Net learns a Transformation matrix with a specified dimension '''
    def __init__(self, dim, num_points=2500):
        super(TNet, self).__init__()

        # dimensions for transform matrix
        self.dim = dim 

        self.conv1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, dim**2)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.max_pool = nn.MaxPool1d(kernel_size=num_points)

    def forward(self, x):
        bs = x.shape[0]

        # print(x.shape)

        # pass through shared MLP layers (conv1d)
        # x = self.bn1(F.relu(self.conv1(x)))
        # x = self.bn2(F.relu(self.conv2(x)))
        # x = self.bn3(F.relu(self.conv3(x)))

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # max pool over num points
        x = self.max_pool(x).view(bs, -1)
        
        # pass through MLP
        # x = self.bn4(F.relu(self.linear1(x)))
        # x = self.bn5(F.relu(self.linear2(x)))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        # initialize identity matrix
        iden = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()

        x = x.view(-1, self.dim, self.dim) + iden

        return x

class DQRegMLP(nn.Module):
    def __init__(self,  hidden_dim=512):
        super(DQRegMLP, self).__init__()
        self.input_dim = 8
        self.output_dim = 8
        self.hidden_dim = hidden_dim
        self.freq = 4

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)      
            )

        self.encoder = nn.Sequential(
                    nn.Linear(self.input_dim * 4 * 2, self.hidden_dim),
                    nn.ReLU()
                )

    def sin_encoding(self, x):
        x = torch.cat([
            torch.sin(x), torch.cos(x),
            torch.sin(2*x), torch.cos(2*x),
            torch.sin(4*x), torch.cos(4*x),
            torch.sin(8*x), torch.cos(8*x),
            ], dim=1)

        return x   

    def forward(self, x):
        orig = x.clone()
        x = self.sin_encoding(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x + orig

class QRegMLP(nn.Module):
    def __init__(self, multi_decoder = True,  hidden_dim=512):
        super(QRegMLP, self).__init__()
        self.multi_decoder = multi_decoder
        if multi_decoder:
            self.input_dim = 7
            self.output_dim_1 = 3
            self.output_dim_2 = 4
            self.hidden_dim = hidden_dim
            self.freq = 4

            self.decoder_1 = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim // 2, self.output_dim_1)
            )

            self.decoder_2 = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, self.output_dim_2)
                )

        else:
                self.input_dim = 7
                self.output_dim = 7
                self.hidden_dim = hidden_dim
                self.freq = 4

                self.decoder = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.output_dim)
                )

        self.encoder = nn.Sequential(
                    nn.Linear(self.input_dim * 4 * 2, self.hidden_dim),
                    nn.LeakyReLU()
                )
    
    def sin_encoding(self, x):

        x = torch.cat([
            torch.sin(x), torch.cos(x),
            torch.sin(2*x), torch.cos(2*x),
            torch.sin(4*x), torch.cos(4*x),
            torch.sin(8*x), torch.cos(8*x),
            ], dim=1)

        return x
        
    def forward(self, x):
        orig = x.clone()
        x = self.sin_encoding(x)
        x = self.encoder(x)
        if self.multi_decoder:
            xyz = self.decoder_1(x)
            q = self.decoder_2(x)
            return xyz + orig[:, :3], nn.functional.normalize(q + orig[:, 3:], dim=1)

        else:
            x = self.decoder(x)
            xyz = x[:, :3]
            q = x[:, 3:]
            if self.add:
                return xyz + orig[:, :3], nn.functional.normalize(q + orig[:, 3:], dim=1)
            else:
                return xyz + orig[:, :3], nn.functional.normalize(quaternion_raw_multiply(orig[:, 3:], q))
            
class RRegMLP(nn.Module):
    def __init__(self,  hidden_dim=512):
        self.add = True
        super(RRegMLP, self).__init__()
        self.input_dim = 9
        self.output_dim_1 = 3
        self.output_dim_2 = 6
        self.hidden_dim = hidden_dim

        self.decoder_1 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim // 2, self.output_dim_1)
        )

        self.decoder_2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.output_dim_2)
            )

        self.encoder = nn.Sequential(
                    nn.Linear(self.input_dim * 4 * 2, self.hidden_dim),
                    nn.LeakyReLU()
                )
    
    def sin_encoding(self, x):

        x = torch.cat([
            torch.sin(x), torch.cos(x),
            torch.sin(2*x), torch.cos(2*x),
            torch.sin(4*x), torch.cos(4*x),
            torch.sin(8*x), torch.cos(8*x),
            ], dim=1)

        return x
        
    def forward(self, x):
        orig = x.clone()
        x = self.sin_encoding(x)
        x = self.encoder(x)

        xyz = self.decoder_1(x)
        r6d = self.decoder_2(x)
        return xyz + orig[:, :3], r6d + orig[:, 3:]

class RegMLP(nn.Module):
    def __init__(self, multi_decoder = True,  hidden_dim=512):
        super(RegMLP, self).__init__()
        self.multi_decoder = multi_decoder
        if multi_decoder:
            self.input_dim = 6
            self.output_dim_1 = 3
            self.output_dim_2 = 3
            self.hidden_dim = hidden_dim
            self.freq = 4

            self.decoder_1 = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim // 2, self.output_dim_1)
            )

            self.decoder_2 = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, self.output_dim_2),
                nn.Tanh()
                )

        else:
                self.input_dim = 6
                self.output_dim = 6
                self.hidden_dim = hidden_dim
                self.freq = 4

                self.decoder = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(self.hidden_dim, self.output_dim)
                )

        self.encoder = nn.Sequential(
                    nn.Linear(self.input_dim * 4 * 2, self.hidden_dim),
                    nn.LeakyReLU()
                )
    
    def sin_encoding(self, x):

        x = torch.cat([
            torch.sin(x), torch.cos(x),
            torch.sin(2*x), torch.cos(2*x),
            torch.sin(4*x), torch.cos(4*x),
            torch.sin(8*x), torch.cos(8*x),
            ], dim=1)

        return x
        
    def forward(self, x):
        orig = x.clone()
        x = self.sin_encoding(x)
        x = self.encoder(x)
        if self.multi_decoder:
            xyz = self.decoder_1(x)
            rpy = self.decoder_2(x)
            return xyz + orig[:, :3], rpy + orig[:, 3:]

        else:
            x = self.decoder(x)
            xyz = x[:, :3]
            rpy = x[:, 3:]
            return xyz + orig[:, :3], rpy + orig[:, 3:]

class RegMLP_tiny(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(RegMLP_tiny, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.freq = 4

        self.fc1 = nn.Linear(input_dim * 4 * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.linear = nn.Linear(input_dim, output_dim)

    def sin_encoding(self, x):

        x = torch.cat([
            torch.sin(x), torch.cos(x),
            torch.sin(2*x), torch.cos(2*x),
            torch.sin(4*x), torch.cos(4*x),
            torch.sin(8*x), torch.cos(8*x),
            ], dim=1)
        return x

    def forward(self, x):
        orig = x
        x = self.sin_encoding(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = x + orig
        return x

if __name__ == '__main__':
    torch.manual_seed(42)
    x = torch.rand(1, 3, 2500)
    model = RegMLP(2500)
    out = model(x)
    print(out.shape)