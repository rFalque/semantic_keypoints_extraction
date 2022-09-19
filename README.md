# Semantic keypoint extraction

Implementation of supervised semantic keypoint extraction

## dependencies

* [torch-points3d](https://github.com/torch-points3d/torch-points3d): for the pointnet model
* [pot pourri](https://github.com/nmwsharp/potpourri3d): for heat distance -> `pip install potpourri3d`
* [open3d](http://www.open3d.org/): for data loading / data visualization
* [pytorch3d](https://pytorch3d.org/): (better data loading?)

## run the test file

To try an inference, run the following script:
```bash
python test.py
```

## preprocessing of the input data:

* copy the annotation files in the `annotations` folder
* copy the ply files in the `cloud` folder
* precompute the geodesic distance to the annotation `python process_labels.py`

## folder structure

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
    └───preprocessed_data
    |   │   file111.npz
    |   │   file112.npz
    |   │   ...
    │
    └───annotations
        │   file111.csv
        │   file112.csv
        │   ...
```
