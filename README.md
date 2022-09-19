# semantic_keypoints_extraction

## dependencies

```bash
mkdir logs
```



* [torch-points3d](https://github.com/torch-points3d/torch-points3d): for the pointnet model
* [pot pourri](https://github.com/nmwsharp/potpourri3d): for heat distance -> `pip install potpourri3d`
* [open3d](http://www.open3d.org/): for data loading / data visualization
* [pytorch3d](https://pytorch3d.org/): (better data loading?)
* openmesh (I think I gave on using that?)

## preprocessing of the input data:
A link to some pre-processed data is available [here](https://drive.google.com/drive/folders/1Xs2D7U52KBMnQZW5A3TNd-ioFB4GQAeB?usp=sharing). Otherwise the required step for the preprocessing are as follow:

* copy the annotation files in the `annotations` folder
* copy the ply files in the `cloud` folder
* downsample the pointcloud with meshlab `./meshlab_preprocessing/cloud_downsampling.sh`
* precompute the geodesic distance to the annotation `python process_labels.py`

## folder structure

A sample dataset can be downloaded [here](https://drive.google.com/drive/folders/1Xs2D7U52KBMnQZW5A3TNd-ioFB4GQAeB?usp=sharing).

The data folder has the following structure:
```
project
└───data
    │
    └───clouds
    |   │   file111.ply
    |   │   file112.ply
    |   │   ...
    │
    └───labels
    |   │   file111.npy
    |   │   file112.npy
    |   │   ...
    │
    └───annotations
        │   file111.csv
        │   file112.csv
        │   ...
```

## Notable runs (trained for 50 epochs)

Pointnet:
* benchmark     | model_2022-06-08_11-41-26.pt
* scale         | model_2022-06-08_15-28-30.pt
* flip          | model_2022-06-08_18-15-06.pt
* shear-forward | model_2022-06-08_15-49-45.pt
* shear-sideway | model_2022-06-08_16-07-50.pt
* all           | model_2022-06-08_18-33-42.pt

kpconv:
* benchmark     | model_2022-06-10_13-58-12.pt
* scale         | model_2022-06-11_11-06-59.pt
* flip          | model_2022-06-11_14-06-19.pt
* shear-forward | model_2022-06-11_17-03-35.pt
* shear-sideway | model_2022-06-11_19-09-34.pt
* all           | model_2022-06-11_21-13-03.pt


## data augmentation

* non-uniform scaling
* jittering
* point dropout

Using transformation matrices
* shearing from 3D matrix (Identity matrix plus some noise) + addition of the euclidean distance to the heat map (check how we can compute that cheaply)
* scaling
* random flip with respect to ZX plan (cows flipped in the other direction) -> also flip the order of the annotations

* https://arxiv.org/pdf/2112.06029.pdf
* https://arxiv.org/pdf/2008.06374.pdf

## todo

tasks to do:


* check if the distance on the manifold is absolute or relative
* integrate data augmentation
    * update the distance on the manifold
    * camera dropout
    * camera calibration noise
        * add visualization within the plyfile_loader.py file
        * add camera plots

* clean up remaining:
    * test.py
    * prediction_only.py
    * data folder
* add normals to the cloud (discussion with Alen?)
* save latest model name somewhere (write in config.yaml?)
* increase swp size (?)
* test downsmpling (?)

For the paper
* add noise on calibration values (small delta on so3)
* smooth guidance for embedded deformation (?) -> Cedric's idea

Done:
* run the network on cpu in a docker container (done by Alex)
* Update the computation of the loss function for the training to compute it simultaneously to the training loss function (now currently on a sliding window)
* skew 3D matrix for data augmentation (see data augmentation)
* check the literature on data augmentation
* check other pointcloud methods (e.g., [kpconv](https://arxiv.org/abs/1904.08889), also implemented in [torch-points3d](https://github.com/nicolas-chaulet/torch-points3d/blob/master/examples/kpconv_segmentation_forward.py)).
* solve problem with eval, seems to be related to batchNorm (see discussion [here](https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/29))
* major clean up of the code
* debug kpconv on gpu
* change data loader to move to device
* move everything to GPU
* downsample all the clouds
* update data loader
* upload data folder on the cloud
* test pointnet implementation
* process data folder
* print loss in log file for each epoch
* check if I can overfit one cloud
* update the yaml file (see: https://stackoverflow.com/a/28559739)
* process next batch of data ([20211109_tullimba](/home/raphael/dev/0_data/beef_livestock/20211109_tullimba))
* add balls on argmax for visualization
* write documentation for Meshlab scripting
* save prediction, add independent plots of the prediction for the meeting
