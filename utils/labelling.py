import open3d as o3d
import numpy as np
from plyfile import PlyData
import potpourri3d as pp3d
import pandas as pd
from os import listdir
from os.path import isfile, join
import random


epsilon = 10
sampling_size = 20000

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

def create_sphere(center): # need to add automatic scalling
    sphere = o3d.geometry.TriangleMesh.create_sphere(0.03)
    sphere.translate(center)
    sphere.compute_vertex_normals()
    return sphere

def list_folder(path):
    instance_list = [f[:-4] for f in listdir(path) if isfile(join(path, f))]
    return instance_list

def load_pcd(path):
    # load the data
    plydata = PlyData.read(path)
    # split the plydata data structure
    pointcloud = {}
    pointcloud['vertices'] = np.stack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).transpose()
    pointcloud['normals'] = np.stack([plydata['vertex']['nx'], plydata['vertex']['ny'], plydata['vertex']['nz']]).transpose()
    pointcloud['labels'] = plydata['vertex']['label']
    pointcloud['curvature'] = plydata['vertex']['curvature']
    # return the pointcloud
    return pointcloud

def split_pointcloud_per_camera(pointcloud):
    camera_ids = np.unique(pointcloud['labels'])
    # build a list of pointclouds for each camera
    pointclouds_list = []
    for camera_id in camera_ids:
        pointcloud_temp = {}
        pointcloud_temp['vertices'] =  pointcloud['vertices'][pointcloud['labels'] == camera_id, :]
        pointcloud_temp['normals'] =   pointcloud['normals'][pointcloud['labels'] == camera_id, :]
        pointcloud_temp['labels'] =    pointcloud['labels'][pointcloud['labels'] == camera_id]
        pointcloud_temp['curvature'] = pointcloud['curvature'][pointcloud['labels'] == camera_id]
        pointclouds_list.append(pointcloud_temp)
    # return the list
    return pointclouds_list

def join_pointclouds(pointclouds_list):
    print()

def find_index_in_cloud(pointcloud, annotations):
    indexes = []
    for i in range(annotations.shape[0]):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud['vertices'])
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        [k, idx, _] = pcd_tree.search_knn_vector_3d(annotations[i,:], 1)
        indexes.append(idx[0])
    return indexes

def downsample(pointcloud_in, input_sampling_size):
    input_size = pointcloud_in['vertices'].shape[0]
    # create indexes to keep
    if input_size >= input_sampling_size:
        indexes = np.sort(random.sample(range(input_size), input_sampling_size))
    else:
        number_of_points_to_add = input_sampling_size - input_size
        indexes = np.sort(list(range(input_size)) + list(random.sample(range(input_size), number_of_points_to_add)))
    #update the pointcloud
    pointcloud_out = {}
    pointcloud_out['vertices'] =  pointcloud_in['vertices'][indexes, :]
    pointcloud_out['normals'] =   pointcloud_in['normals'][indexes, :]
    pointcloud_out['labels'] =    pointcloud_in['labels'][indexes]
    pointcloud_out['curvature'] = pointcloud_in['curvature'][indexes]
    return pointcloud_out

def compute_distance_on_the_manifold(pointcloud, indexes):
    # set variables
    annotation_number = len(indexes)
    points = pointcloud['vertices']
    solver = pp3d.PointCloudHeatSolver(points)
    # compute the distance for each annotation
    distance_on_the_manifold = np.tile(0.0, (points.shape[0], annotation_number))
    gaussian_distance = np.tile(0.0, (points.shape[0], annotation_number))
    for i in range(annotation_number):
        # compute the distance on the manifold
        dists = solver.compute_distance(indexes[i])
        distance_on_the_manifold[:, i] = dists
        # turn into Gaussian: https://en.wikipedia.org/wiki/Radial_basis_function
        dists = dists/dists.max()
        dists = np.exp(-np.square(epsilon*dists))
        gaussian_distance[:, i] = dists
    return distance_on_the_manifold, gaussian_distance


def process_labels(folder_path):
    instances = list_folder(folder_path + "/clouds")
    for instance in instances:
        # load data
        print("process " + instance)
        pointcloud = load_pcd(folder_path + "/clouds/" + instance + ".ply")
        annotations = np.asarray(pd.read_csv(folder_path + "/annotations/" + instance + ".csv", header=None))
        new_point_cloud = downsample(pointcloud, sampling_size)
        indexes = find_index_in_cloud(new_point_cloud, annotations)
        distance_on_the_manifold, gaussian_distance = compute_distance_on_the_manifold(new_point_cloud, indexes)
        np.savez(folder_path + "/preprocessed_data/" + instance,
                 vertices =  new_point_cloud['vertices'],
                 normals =   new_point_cloud['normals'],
                 labels =    new_point_cloud['labels'],
                 curvature = new_point_cloud['curvature'],
                 distance_on_the_manifold = distance_on_the_manifold,
                 gaussian_distance = gaussian_distance,
                 indexes = indexes)


def process_labels_old(folder_path):
    instances = list_folder(folder_path + "/clouds")
    for instance in instances:
        # load data
        print("process: " + instance)
        pcd = o3d.io.read_point_cloud(folder_path + "/clouds/" + instance + ".ply")
        annotation = np.asarray(pd.read_csv(folder_path + "/annotations/" + instance + ".csv", header=None))

        # set variables
        annotation_number = annotation.shape[0]
        points = np.vstack((annotation, np.asarray(pcd.points)))
        solver = pp3d.PointCloudHeatSolver(points)

        labels = np.tile(0.0, (np.asarray(pcd.points).shape[0], annotation_number))
        labels_distances = np.tile(0.0, (np.asarray(pcd.points).shape[0], annotation_number))
        for i in range(annotation_number):
            # compute the distance
            dists = solver.compute_distance(i)
            dists = dists[annotation_number:]
            labels_distances[:, i] = dists

            # turn into Gaussian: https://en.wikipedia.org/wiki/Radial_basis_function
            epsilon = 10
            dists = dists/dists.max()
            dists = np.exp(-np.square(epsilon*dists))
            labels[:, i] = dists

        # save to folder
        np.save(folder_path + "/labels/" + instance + ".npy", labels)
        np.save(folder_path + "/labels_distances/" + instance + ".npy", labels_distances)



def visualize_labels(folder_path, instance):
    # load data
    pcd = o3d.io.read_point_cloud(folder_path + "/clouds/" + instance + ".ply")
    annotation = np.asarray(pd.read_csv(folder_path + "/annotations/" + instance + ".csv", header=None))

    # set variables
    annotation_number = annotation.shape[0]
    objects_to_visualize = []
    np_colors = a = np.tile(1.0, np.asarray(pcd.points).shape)
    keys = list(colors)

    for i in range(annotation_number):
        sphere = create_sphere(annotation[i])
        sphere.paint_uniform_color(np.asarray(colors[keys[i]])*0.7)
        objects_to_visualize.append(sphere)

    pcd.colors = o3d.utility.Vector3dVector(np_colors)
    objects_to_visualize.append(pcd)
    o3d.visualization.draw_geometries(objects_to_visualize)
