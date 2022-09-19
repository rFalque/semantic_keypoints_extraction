import os
import open3d as o3d
import numpy as np
import torch
from utils.data_augmentation import DataAugmentation
import utils.visualization

colors = { # iterate with `keys = list(colors)` and `colors[keys[0]]`
  "red"     : [1.0, 0.0, 0.0],
  "green"   : [0.0, 1.0, 0.0],
  "blue"    : [0.0, 0.0, 1.0],
  "yellow"  : [1.0, 1.0, 0.0],
  "pink"    : [1.0, 0.0, 1.0],
  "aqua"    : [0.0, 1.0, 1.0],
  "brown"   : [0.5, 0.5, 0.1],
  "orange"  : [1.0, 0.7, 0.1],
  "purple"  : [0.9, 0.4, 0.9],
  "white"   : [1.0, 1.0, 1.0]
}

data_augmentation_options = {
    'scale' : True,
    'flip' : True,
    'shear_forward' : True,
    'shear_sideway' : True,
    'calibration' : True,
    'view_drop' : False
}

def expMap( angle_axis ):
    angle_axis = angle_axis.squeeze()
    rot_mat = np.identity(3)
    norm_vect = np.linalg.norm(angle_axis)
    if norm_vect!=0:
        sMat = np.matrix([[0.0, -angle_axis[2], angle_axis[1]],
                          [angle_axis[2], 0.0, -angle_axis[0]],
                          [-angle_axis[1], angle_axis[0], 0.0]])
        rot_mat = rot_mat + ( (np.sin(norm_vect)/norm_vect) * sMat) + ( ( (1-np.cos(norm_vect)) / pow(norm_vect,2)) * sMat @ sMat)
    # if(norm(angle_axis) ~= 0)
    #     rot_mat = axang2rotm([(angle_axis'/norm(angle_axis)), norm(angle_axis)]);
    # else
    #     rot_mat = eye(3);
    # end
    return rot_mat

class PointcloudDataset():
    def __init__(self, inputs_dir, device="cpu", mode="train", augmentation=False):
        self.device = device
        self.inputs_dir = inputs_dir
        self.inputs_paths = os.listdir(inputs_dir)
        self.inputs_paths.sort()
        self.mode = mode
        self.augmentation = augmentation
        
    def __len__(self):
        return len(self.inputs_paths)

    # as a design principle, we set the data loader to account for data batches. For single instances:
    # pointcloud.shape =      torch.Size([1, n, 3])
    # labels_distance.shape = torch.Size([1, n, m])
    # annotation.shape =      torch.Size([1, m, 3])
    def __getitem__(self, idx):
        # inputs
        data_path = os.path.join(self.inputs_dir, self.inputs_paths[idx])
        npzfile = np.load(data_path)
        cloud = torch.tensor(npzfile['vertices']).unsqueeze(0).float().to(device=self.device)

        # outputs
        if self.mode == "prediction":
            return cloud

        elif self.mode == "test":
            # load labels
            labels_distance = npzfile['distance_on_the_manifold']
            labels_distance = torch.tensor(labels_distance).unsqueeze(0).float().to(device=self.device)
            for label in range(6):
                labels_distance[:,:,label] = labels_distance[:,:,label]/torch.max(labels_distance[:,:,label]) # if modified, check line 88

            epsilon = 10
            labels = torch.exp(-torch.square(epsilon*labels_distance))

            # load annotations and stack it to the cloud
            annotation_indexes = npzfile['indexes']
            annotations = cloud[:, annotation_indexes]

            return cloud, labels, annotations

        else:
            # load labels
            labels_distance = npzfile['distance_on_the_manifold']
            labels_distance = torch.tensor(labels_distance).unsqueeze(0).float().to(device=self.device)

            if not self.augmentation:
                # apply Gaussian onto the distance
                for label in range(6):
                    labels_distance[:,:,label] = labels_distance[:,:,label]/torch.max(labels_distance[:,:,label]) # if modified, check line 69

                epsilon = 10
                labels = torch.exp(-torch.square(epsilon*labels_distance))

                return cloud, labels

            else:
                # add noise specific to the camera calibration
                camera_labels = npzfile['labels']
                camera_ids = np.unique(camera_labels)
                if data_augmentation_options['calibration']:
                    pos_noise_std = 0.02
                    rot_noise_std = 0.03
                    for camera_id in camera_ids:
                        R_temp = torch.tensor(np.identity(3) @ expMap(rot_noise_std*np.random.normal(size=[3,1]))).float().to(device=self.device)
                        t_temp = torch.tensor(pos_noise_std*np.random.normal(size=[1,3])).float().to(device=self.device)
                        centroid = cloud[:, camera_labels == camera_id].mean(1)
                        cloud_temp = cloud[:, camera_labels == camera_id] - centroid
                        cloud_temp = cloud_temp @ R_temp
                        cloud_temp = cloud_temp + centroid
                        cloud_temp = cloud_temp + t_temp
                        cloud[:, camera_labels == camera_id] = cloud_temp

                # set up augmentation methods
                data_augmentation = DataAugmentation()
                if data_augmentation_options['scale']:
                    data_augmentation.transform_scale()
                if data_augmentation_options['flip']:
                    data_augmentation.transform_flip()
                if data_augmentation_options['shear_forward']:
                    data_augmentation.transform_shear_forward()
                if data_augmentation_options['shear_sideway']:
                    data_augmentation.transform_shear_sideway()
                #data_augmentation.transform_translate()

                # set annotation indexes:
                annotation_indexes = npzfile['indexes']
                
                # implementation of the transform directly on the GPU (there is a small error between this and the one from open3d)
                Ttorch = torch.transpose(torch.tensor(data_augmentation.build_transform(), device=self.device).float(), 0,1)
                cloud_to_multiply = torch.cat((cloud, torch.ones(1, cloud.shape[1], 1, device=self.device)), 2) # Convert all 3D points to homogeneous coordinates
                morphed_cloud = torch.matmul(cloud_to_multiply, Ttorch)[:,:,0:3].contiguous()

                # update labels based on morphing of the cloud
                for label in range(6):
                    distance_before = torch.linalg.norm(cloud[:, annotation_indexes[label], :] - cloud, dim=2)
                    distance_after =  torch.linalg.norm(morphed_cloud[:, annotation_indexes[label], :] - morphed_cloud, dim=2)
                    labels_distance[:,:,label] = labels_distance[:,:,label] - distance_before + distance_after
                    labels_distance[:,:,label] = labels_distance[:,:,label]/torch.max(labels_distance[:,:,label])

                if data_augmentation.flip is True:
                    labels_distance = labels_distance[:, :, [5, 4, 2, 3, 1, 0]]

                # apply Gaussian onto the distance
                epsilon = 10
                labels = torch.exp(-torch.square(epsilon*labels_distance))

                # drop points based on camera_labels
                if data_augmentation_options['view_drop']:
                    for camera_id in camera_ids:
                        if np.random.random() > 0.8:
                            morphed_cloud = morphed_cloud[:, camera_labels != camera_id]
                            labels = labels[:, camera_labels != camera_id]
                            camera_labels = camera_labels[camera_labels != camera_id]

                return morphed_cloud, labels


    def visualize(self, idx):
        [cloud, labels] = self[idx]
        cloud_cpu = cloud.cpu().squeeze().numpy()
        labels_cpu = labels.cpu().squeeze().detach().numpy()
        utils.visualization.visualize(cloud_cpu, labels_cpu)

    
    def load_and_visualize(self, path):
        pcd = o3d.io.read_point_cloud(path)
        o3d.visualization.draw_geometries([pcd])


    def get_instance_name(self, idx):
        return self.inputs_paths[idx][:-4]
