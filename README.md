# PolarMix

[PolarMix: A General Data Augmentation Technique for LiDAR Point Clouds](https://arxiv.org/abs/2208.00223)

[Aoran Xiao](https://xiaoaoran.github.io/Home/), [Jiaxing Huang](https://jxhuang0508.github.io/), [Dayan Guan](https://dayan-guan.github.io/), [Kaiwen Cui](https://scholar.google.com/citations?user=-9KXqLsAAAAJ&hl=zh-CN), [Shijian Lu](https://personal.ntu.edu.sg/shijian.lu/), [Ling Shao](https://scholar.google.com/citations?user=z84rLjoAAAAJ&hl=en)

NeurIPS 2022

## News

**[2022-09-20]** Code is released.  
**[2022-09-15]** Our paper is accepted to NeurIPS 2022.


## Usage

#### Installation

Please visit and follow installation instruction in [this repo](https://github.com/mit-han-lab/spvnas).


### Data Preparation

#### SemanticKITTI  
- Please follow the instructions from [here](http://www.semantic-kitti.org) to download the SemanticKITTI dataset (both KITTI Odometry dataset and SemanticKITTI labels) and extract all the files in the `sequences` folder to `/dataset/semantic-kitti`. You shall see 22 folders 00, 01, …, 21; each with subfolders named `velodyne` and `labels`.  
- Change the data root path in configs/semantic_kitti/default.yaml


### Training

#### SemanticKITTI

We release the training code for SPVCNN and MinkowskiNet with PolarMix. You may run the following code to train the model from scratch. 

SPVCNN:
```bash
python train.py configs/semantic_kitti/spvcnn/cr0p5.yaml --run-dir runs/semantickitti/spvcnn_polarmix --distributed False
```
MinkowskiNet:
```bash
python train.py configs/semantic_kitti/minkunet/cr0p5.yaml --run-dir run/semantickitti/minkunet_polarmix --distributed False
```

- Note we only used one 2080Ti for training and testing. Training from scratch takes around 1.5 days. You may try larger batch size or distributed learning for faster training.

### Testing Models

You can run the following command to test the performance of SPVCNN/MinkUNet models with PolarMix.

```bash
torchpack dist-run -np 1 python test.py --name ./runs/semantickitti/spvcnn_polarmix
torchpack dist-run -np 1 python test.py --name ./runs/semantickitti/minkunet_polarmix
```

We provide pre-trained models of MinkUNet and SPVCNN. You may [download](https://drive.google.com/drive/folders/1SHaGbgUUxoVNt-Y30XZDedRz7iffQ3JI?usp=sharing) and place them under './runs/semantickitti/' for testing. 

mIoUs over validation set of SemanticKITTI are reported as follows:

|             | w/o PolarMix | w/ PolarMix |
| :---------: | :----------: | :---------: |
| `MinkUNet`  |     58.9     |  65.0       |   
| `SPVCNN`    |     60.7     |  66.2       | 


### Visualizations

Follow instructions in [this repo](https://github.com/mit-han-lab/spvnas).



## Citation

If you use this code for your research, please cite our paper.

```
@article{xiao2022polarmix,
  title={PolarMix: A General Data Augmentation Technique for LiDAR Point Clouds},
  author={Xiao, Aoran and Huang, Jiaxing and Guan, Dayan and Cui, Kaiwen and Lu, Shijian and Shao, Ling},
  journal={arXiv preprint arXiv:2208.00223},
  year={2022}
}
```

## Thanks
We thank the opensource project [TorchSparse](https://github.com/mit-han-lab/torchsparse) and [SPVNAS](https://github.com/mit-han-lab/spvnas).


## Related Repos
Find our other repos for point cloud understanding!
- [Unsupervised Representation Learning for Point Clouds: A Survey](https://github.com/xiaoaoran/3d_url_survey)
- [SynLiDAR: Synthetic LiDAR sequential point cloud dataset with point-wise annotations (AAAI2022)](https://github.com/xiaoaoran/SynLiDAR)
- [FPS-Net: A convolutional fusion network for large-scale LiDAR point cloud segmentation](https://github.com/xiaoaoran/FPS-Net)
