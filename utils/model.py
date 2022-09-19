import torch
from torch_geometric.data import Batch, Data
from torch_points3d.applications.pointnet2 import PointNet2
from torch_points3d.applications.kpconv import KPConv

class Model:
    def __init__(self, architecture,  input_nc, num_layers, output_nc):
        self.architecture = architecture
        self.input_nc = input_nc
        self.num_layers = num_layers
        self.output_nc = output_nc

        if self.architecture == "pointnet_unet":
            self.model = PointNet2(architecture="unet", input_nc=self.input_nc, num_layers=self.num_layers, output_nc=self.output_nc)
        elif self.architecture == "kpconv_unet":
            self.model = KPConv(architecture="unet", input_nc=0, output_nc=self.output_nc, num_layers=self.num_layers)
        else:
            print("unknown architecture")
    
    def forward(self, cloud, features=None):
        if features is None:
            features = cloud

        if self.architecture == "pointnet_unet":
            data = Data(pos=cloud, x=cloud)
            output = self.model.forward(data)
            output.x = torch.transpose(output.x, 1, 2)

        elif self.architecture == "kpconv_unet": # does not account for features (?)
            pos = cloud.squeeze()
            x = torch.ones((pos.shape[0], 1))
            batch = torch.zeros((pos.shape[0])).long()
            data = Batch(x=x, pos=pos, batch=batch)
            output = self.model.forward(data)
            output.x = output.x.unsqueeze(0)

        return output
    
    def load_state_dict(self, weights):
        self.model.load_state_dict(weights)

    def to(self, device):
        self.model.to(device)
    
    def train(self):
        self.model.train()

    def test(self):
        self.model.test()

    def eval(self):
        self.model.eval()
    
    def __str__(self):
        print(self.model)

    def parameters(self):
        return self.model.parameters()
    
    def state_dict(self):
        return self.model.state_dict()