# Semantic keypoint extraction

Implementation of [supervised semantic keypoint extraction](https://arxiv.org/pdf/2211.08634.pdf).

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


## Citation

If you are using our system in your research, consider citing our paper.

```bibtex
@article{Falque2022,
  title={Semantic keypoint extraction for scanned animals using multi-depth-camera systems},
  author={Falque, Raphael and Vidal-Calleja, Teresa and Alempijevic, Alen},
  journal={arXiv preprint arXiv:2211.08634},
  year={2022}
}
```
