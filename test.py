import torch

import torch
import numpy as np
from torch_points3d.applications.pointnet2 import PointNet2
from torch_geometric.data import Batch, Data
#import torch_points3d

from utils.params import Params
from utils.logger import Logger
from utils.data_loader import PointcloudDataset
from utils.visualization import visualize


if __name__ == '__main__':
    params = Params("config.yaml")
    logger = Logger(False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # data loader (to be updated)
    data_loader = PointcloudDataset("data/test/preprocessed_data/", device=device)

    # define architecture and learning parameters
    input_nc = params.yaml['pointnet2']['input_nc'] # additional features
    output_nc = params.yaml['pointnet2']['output_nc']

    model = PointNet2(architecture="unet", input_nc=input_nc, num_layers=3, output_nc=output_nc)
    model.load_state_dict(torch.load("data/weights/model_2022-09-15_20-52-11.pt"))
    model.to(device)
    criterion = torch.nn.MSELoss()

    print("number of parameters: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)) )


    # loop through data loader
    for i in range(data_loader.__len__()):        
        [verts, labels] = data_loader[i]
        data = Data(pos=verts, x=verts)

        output = model.forward(data)

        loss = criterion(torch.transpose(output.x, 1, 2), labels)
        print("loss: " + str(loss))

        if (params.yaml['test']['save_output']):
            folder_path = params.yaml['IO']['path_to_save_test_predictions']
            np.save(folder_path + "/prediction_" + data_loader.get_instance_name(i) + ".npy", output.x.cpu().squeeze().detach().numpy().transpose())

        # visualize IO
        if (params.yaml['test']['visualization']):
            distance_predicted = output.x.cpu().squeeze().detach().numpy().transpose()
            distance_from_annotation = labels.cpu().squeeze().detach().numpy()
            vertices = verts.cpu().squeeze().numpy()

            # plot the manual annotation
            print('\nVisualize the manual annotation and the distance wrt the keypoints')
            visualize(vertices, distance_from_annotation)
            
            # plot the network prediction
            print('\nVisualize the network prediction and the keypoints (argmax of prediction)')
            visualize(vertices, distance_predicted)
