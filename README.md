# Multiple Instance Learning (MIL)
## About
This program is an unofficial implementation of this paper.

* [Sultani, Waqas, Chen Chen, and Mubarak Shah. "Real-world anomaly detection in surveillance videos." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.](https://arxiv.org/abs/1801.04264)

## I3D
Instead of C3D, I use the following I3D and pre-trained models.
* [I3D](https://github.com/piergiaj/pytorch-i3d/blob/master/pytorch_i3d.py)
* [MUSDL](https://github.com/nzl-thu/MUSDL)

## Extract features as preprocessing

``` shell
$ python extractor.py normal.txt normal.pt
$ python extractor.py anomalous.txt anomalous.pt
```

### Example

Include the path and label of the image (1 if it contains an anomaly, 0 if it does not) separated by a space.
The paths of the images should be sorted in the order of the frames of the video.

#### normal.txt
``` normal.txt
...
/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Train/Train001/002.tif 0
/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Train/Train001/003.tif 0
/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Train/Train001/004.tif 0
...
```

#### anomalous.txt

``` anomalous.txt
...
/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Test/Test003/089.tif 0
/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Test/Test003/090.tif 0
/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Test/Test003/091.tif 1
...
```

## Implement learning and evaluating

``` shell
$ python main.py normal.pt anomalous.pt

100%|██████| 48/48 [00:00<00:00, 167.17batch/s, loss=1.01]
AUC score: 0.629
```