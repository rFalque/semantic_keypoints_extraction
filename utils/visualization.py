
import open3d as o3d
import numpy as np

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
    
def visualize(verts, labels, plot_keypoints=True, plot_distance=True):
    annotation_number = labels.shape[1]
    np_colors = a = np.tile(1.0, verts.shape)
    keys = list(colors)
    spheres = []
    for i in range(annotation_number):
        # update global colormap
        color_to_add = np.asarray(colors[keys[i]]) - np.asarray(colors["white"])
        values_to_add = np.asarray([labels[:,i], labels[:,i], labels[:,i]]).transpose()
        print(f"max value for {keys[i]} label {labels[:,i].max()}")
        if plot_distance:
            np_colors = np_colors + values_to_add * color_to_add
        else:
            np_colors = np_colors

        # add spheres to plot
        distance_predicted = np.asarray(labels[:,i])
        max_value = np.argmax(distance_predicted)
        sphere = create_sphere(verts[max_value])
        sphere.paint_uniform_color(np.asarray(colors[keys[i]])*0.7)
        spheres.append(sphere)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.colors = o3d.utility.Vector3dVector(np_colors)
    pcd.estimate_normals()

    # create visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0]) # set bg as black
    vis.add_geometry(pcd)
    if plot_keypoints:
        for sphere in spheres:
            vis.add_geometry(sphere)
    vis.run()
    param = ctr.convert_to_pinhole_camera_parameters()
    trajectory = o3d.camera.PinholeCameraTrajectory()
    trajectory.parameters = trajectory.parameters + [param]
    o3d.io.write_pinhole_camera_trajectory("data/camera_params/latest.json", trajectory)
    vis.destroy_window()

def take_screenshot(verts, labels, params, path_name, plot_keypoints=True, plot_distance=True):
    annotation_number = labels.shape[1]
    option = o3d.visualization.RenderOption()
    option.point_size = 100
    np_colors = a = np.tile(1.0, verts.shape)
    keys = list(colors)
    spheres = []
    for i in range(annotation_number):
        # update global colormap
        color_to_add = np.asarray(colors[keys[i]]) - np.asarray(colors["white"])
        values_to_add = np.asarray([labels[:,i], labels[:,i], labels[:,i]]).transpose()
        if plot_distance:
            np_colors = np_colors + values_to_add * color_to_add
        else:
            np_colors = np_colors
        # add spheres to plot
        distance_predicted = np.asarray(labels[:,i])
        max_value = np.argmax(distance_predicted)
        sphere = create_sphere(verts[max_value])
        sphere.paint_uniform_color(np.asarray(colors[keys[i]])*0.7)
        spheres.append(sphere)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.colors = o3d.utility.Vector3dVector(np_colors)
    pcd.estimate_normals()

    # create visualization
    vis = o3d.visualization.Visualizer()
    trajectory = o3d.io.read_pinhole_camera_trajectory(params)
    vis.create_window(width=trajectory.parameters[0].intrinsic.width, height=trajectory.parameters[0].intrinsic.height)
    ctr = vis.get_view_control()

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0]) # set bg as black
    vis.add_geometry(pcd)

    if plot_keypoints:
        for sphere in spheres:
            vis.add_geometry(sphere)

    ctr.convert_from_pinhole_camera_parameters(trajectory.parameters[0])
    vis.capture_screen_image(path_name, True)
