import torch

from utils.params import Params
from utils.state import State
from utils.logger import Logger
from utils.data_loader import PointcloudDataset
from utils.model import Model

import wandb

debug_mode = False

# for reproductible results
import random
random.seed(0)
torch.manual_seed(0)

if __name__ == '__main__':
    architecture = "pointnet_unet"
    #architecture = "kpconv_unet" # weights not available for kpconv_unet, please create an issue if needed

    # start logging and connect to wandb
    params = Params("config.yaml")
    logger = Logger(params.yaml['save_log'] and not debug_mode)
    logger.log_params(params)
    if not debug_mode:
        wandb.init(project="test", entity="rfalque")

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device used: " + device.__str__())

    # data loader (to be updated)
    data_loader_training = PointcloudDataset("data/test/preprocessed_data/", device, augmentation=True)
    data_loader_testing  = PointcloudDataset("data/test/preprocessed_data/", device=device)

    # define model architecture and learning parameters
    model = Model(architecture=architecture, 
                  input_nc=params.yaml[architecture]['input_nc'], 
                  num_layers=params.yaml[architecture]['num_layers'], 
                  output_nc=params.yaml[architecture]['output_nc'])

    # set model parameters from previously learned stuff
    if (params.yaml['start_training_from_previous_training_set']):
        print("Fine tunning of the parameters using pre-trained model")
        model.load_state_dict(torch.load(params.yaml[architecture]['pretrained_path']))
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=params.yaml['learning_rate'], momentum=0.9)
    criterion = torch.nn.MSELoss()

    # store informations about the state
    state = State()
    state.number_of_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    state.batch_size = data_loader_training.__len__()
    state.device = device.__str__()
    state.epochs = params.yaml['epochs']
    state.loss = criterion.__str__()
    
    logger.log_state(state)

    if not debug_mode:
        wandb.config = {
            "learning_rate": params.yaml['learning_rate'],
            "epochs": params.yaml['epochs'],
            "batch_size": data_loader_training.__len__()
        }

    # loop through epochs
    for state.epoch in range(1, state.epochs+1):
        print("Progress: starting epoch " + str(state.epoch))

        # test the performance of the network on the training dataset
        train_loss = torch.Tensor([0.]).to(device)
        for i in range(data_loader_training.__len__()):
            [verts, labels] = data_loader_training[i]
            output = model.forward(verts)
            train_loss += float(criterion(output.x, labels))
        train_loss_to_save = train_loss.cpu().numpy()[0] / data_loader_training.__len__()

        # test the performance of the network on the testing dataset
        test_loss = torch.Tensor([0.]).to(device)
        for i in range(data_loader_testing.__len__()):
            [verts, labels] = data_loader_testing[i]
            output = model.forward(verts)
            test_loss += float(criterion(output.x, labels))
        test_loss_to_save = test_loss.cpu().numpy()[0] / data_loader_testing.__len__()

        if not debug_mode:
            wandb.log({"loss/training_loss": train_loss_to_save, "loss/testing_loss": test_loss_to_save})

        # train for one epoch
        cummulative_loss = torch.Tensor([0.]).to(device)
        for i in range(data_loader_training.__len__()):
            if (params.yaml['visualization']):
                data_loader_training.visualize(i)

            # prepare the optimizer
            optimizer.zero_grad()  # zero the gradient buffers

            # package data
            [verts, labels] = data_loader_training[i]

            # forward pass
            output = model.forward(verts)
    
            # back-propagation
            loss = criterion(output.x, labels)
            cummulative_loss += float(loss)
            loss.backward()
            optimizer.step()
            
        # backup model every epochs
        if (not debug_mode):
            temp_path = params.yaml['IO']['path_to_save_temp_model'] + 'model_' + logger.timestamp + '.pt'
            torch.save(model.state_dict(), temp_path)

        state.average_loss = cummulative_loss.cpu().numpy()[0]/data_loader_training.__len__()

        logger.log_temp(state)

    temp_path = params.yaml['IO']['path_to_save_model'] + 'model_' + logger.timestamp + '.pt'
    torch.save(model.state_dict(), temp_path)
