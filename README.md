# TetraSphere: A Neural Descriptor for O(3)-Invariant Point Cloud Analysis

The official implementation of the ["TetraSphere: A Neural Descriptor for O(3)-Invariant Point Cloud Analysis"](https://arxiv.org/abs/2211.14456) paper, accepted to CVPR 2024

[[arXiv]](https://arxiv.org/abs/2211.14456) 


## Teaser

![TetraSphere](misc/teaser.png)


## Requirements
To run the code, install the dependencies by running the following:

```
conda create -n tetrasphere
conda deactivate
conda activate tetrasphere
conda install python=3.9
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
conda install h5py scikit-learn future tqdm wget
pip install tensorboardx pytorch_lightning torchmetrics datetime
```


## Datasets

Inspect `config.py` to find the preset paths to the datasets, and edit if you like.

- The datasets can be downloaded with the convenience application `download_datasets.py`

If you want to download them manually, simply use the links below.

ModelNet-40 can be downloaded ![here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip).

To acquire the ScanObjectNN dataset, download the file h5_files.zip from [here](http://hkust-vgd.github.io/scanobjectnn/h5_files.zip). 

(For reference, the download link was provided by the authors [here](https://github.com/hkust-vgd/scanobjectnn/issues/31).)

To get the ShapeNet-Part you should register on the shapenet.org webpage.
However, as this dataset seems to be inaccessible through browsing the website, we found [this link](https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip) in the github repos of multiple authors working on point cloud segmentation.




## Run

Point cloud classification:

`python train_mn40.py` - ModelNet40

`python train_objbg.py` - ScanObjectNN, `objbg` variant

`python train_pbt50rs.py` - ScanObjectNN, `pb-t50-rs` variant

Part segmentation:

`python train_partseg.py`



