# [SiameseTracking4UAV](https://link.springer.com/article/10.1007/s10462-023-10558-5)
# Siamese Object Tracking for Unmanned Aerial Vehicle: A Review and Comprehensive Analysis

### Changhong Fu, Kunhan Lu, Guangze Zheng, Junjie Ye, Ziang Cao, Bowen Li, and Geng Lu

This work has been accepted and published by Artificial Intelligence Review (JCR Q1, IF = 12).

This code library gives our experimental results and most of the publicly available Siamese trackers.

The trackers are in folder **experiments** and the results are in **results**.

Paper link: https://link.springer.com/article/10.1007/s10462-023-10558-5

View-only link: https://rdcu.be/dhWgD

If you want to use our code libary, experimental results, and related contents, please cite our paper using the format as follows:

```
@article{Fu2023SiameseOT,  
        title={{Siamese Object Tracking for Unmanned Aerial Vehicle: A Review and Comprehensive Analysis}},   
        author={Fu, Changhong and Lu, Kunhan and Zheng, Guangze and Ye, Junjie and Cao, Ziang and Li, Bowen and Lu, Geng},  
        journal={Artificial Intelligence Review},  
        year={2023},  
        pages={1-61},
        doi={10.1007/s10462-023-10558-5}  
}
```

## Results_OPE_AGX

The trackers are tested on the following platform.

- Ubuntu 18.04
- 8-core Carmel ARM v8.2 64-bit CPU
- 512-core Volta GPU
- 32G RAM
- CUDA 10.2
- Python 3.6.3
- Pytorch 0.7.0/1.6.0

**All the Siamese trackers' results are obtained using an [NVIDIA Jetson AGX Xavier](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-agx-xavier/).**

## Figures

Here shows some of the tracking results of 19 SOTA Siamese trackers.

Comparison of the performance under all the six authoritative UAV tracking benchmarks.
<img src="./figures/Precision and FPS.png">

<img src="./figures/Normalized Precision and FPS.png">

<img src="./figures/Success and FPS.png">

The average performance comparison of the five real-time Siamese trackers under all the six authoritative UAV tracking benchmarks.
<img src="./figures/Attributes.png">

## Environment setup
This code has been tested on an [NVIDIA Jetson AGX Xavier](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-agx-xavier/) with Ubuntu 18.04, Python 3.6.3, Pytorch 0.7.0/1.6.0, CUDA 10.2.

Please install related libraries before running this code: 
```bash
pip install pyyaml yacs tqdm colorama matplotlib cython tensorboardX easydict
```

## Test

Download pretrained models form the links in `experiments` directory or download pretrained models from official code site and put them into `experiments` directory.

Download testing datasets and put them into `test_dataset` directory. If you want to test the tracker on a new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to set test_dataset.

The papers whose benchmarks are used in the experimental evaluations are listed here.

#### UAV123@10fps & UAV20L

Paper: A Benchmark and Simulator for UAV Tracking

Paper site: https://link.springer.com/chapter/10.1007%2F978-3-319-46448-0_27 .

Code and benchmark site: https://cemse.kaust.edu.sa/ivul/uav123 .

#### DTB70

Paper: Visual Object Tracking for Unmanned Aerial Vehicles: A Benchmark and New Motion Models

Paper site: https://dl.acm.org/doi/10.5555/3298023.3298169 .

Code and benchmark site: https://github.com/flyers/drone-tracking .

#### UAVDT

Paper: The Unmanned Aerial Vehicle Benchmark: Object Detection and Tracking

Paper site: https://link.springer.com/article/10.1007/s11263-019-01266-1 .

Code and benchmark site: https://sites.google.com/site/daviddo0323/projects/uavdt .

#### VisDrone-SOT2020

Paper: VisDrone-SOT2020: The Vision Meets Drone Single Object Tracking Challenge Results

Paper site: https://link.springer.com/chapter/10.1007/978-3-030-66823-5_44 .

Code and benchmark site: http://aiskyeye.com/ .

#### UAVTrack112

Paper: Onboard Real-Time Aerial Tracking With Efficient Siamese Anchor Proposal Network

Paper site: https://ieeexplore.ieee.org/abstract/document/9477413 .

Code and benchmark site: https://github.com/vision4robotics/SiamAPN .

####  UAVDark135

Paper: All-Day Object Tracking for Unmanned Aerial Vehicle

Paper site: https://ieeexplore.ieee.org/document/9744417 .

Code and benchmark site: https://github.com/vision4robotics/ADTrack_v2 .

####  UAVDarkTrack2021

Paper: Tracker Meets Night: A Transformer Enhancer for UAV Tracking

Paper site: https://ieeexplore.ieee.org/document/9696362 .

Code and benchmark site: https://github.com/vision4robotics/SCT .

####  NAT2021

Paper: Unsupervised Domain Adaptation for Nighttime Aerial Tracking

Paper site: https://ieeexplore.ieee.org/document/9879981 .

Code and benchmark site: https://vision4robotics.github.io/NAT2021/ .

### Option 1

Use the corresponding 'tools/test_<tracker_name>.py' to test the performance of the tracker.
Take the test of SiamAPN as an example:

```
python tools/test_siamapn.py                      \
  --dataset UAVTrack112                           \ # dataset_name
  --datasetpath ./test_dataset                    \ # dataset_path
  --config ./experiments/SiamAPN/config.yaml      \ # tracker_config
  --snapshot ./experiments/SiamAPN/model.pth      \ # tracker_model
  --trackername SiamAPN                             # tracker_name
```

The testing result will be saved in the `results/<dataset_name>/<tracker_name>` directory.

The settings required by different trackers will be different. For details, please refer to the examples in 'tools/test.sh'

### Option 2

Similar to Option 1, a more convenient way of testing is provided using 'tools/test.py' to test all the trackers.

```
python tools/test.py                              \
  --dataset UAVTrack112                           \ # dataset_name
  --datasetpath ./test_dataset                    \ # dataset_path
  --config ./experiments/SiamAPN/config.yaml      \ # tracker_config
  --snapshot ./experiments/SiamAPN/model.pth      \ # tracker_model
  --trackername SiamAPN                             # tracker_name
```

## Evaluation 

If you want to evaluate the trackers mentioned above, please put those results into `results` directory as `results/<dataset_name>/<tracker_name>`.

```
python tools/eval.py                              \
  --dataset UAVTrack112                           \ # dataset_name
  --datasetpath path/of/your/dataset              \ # dataset_path
  --tracker_path ./results                        \ # result_path
  --tracker_prefix 'SiamAPN'                        # tracker_name
```

## Contact

If you have any questions, please contact me.

Kunhan Lu

Email: lukunhan@tongji.edu.cn .

## Acknowledgements
- The code is implemented based on [pysot](https://github.com/STVIR/pysot). We would like to express our sincere thanks to the contributors.

- We would like to thank Jilin Zhao, Kunhui Chen, Haobo Zuo, and Sihang Li for their help in building this code library.

- We also thank the contribution of Matthias Muller, Siyi Li, Dawei Du, Heng Fan et al. for their previous work of the benchmarks UAV123@10fps, UAV20L, DTB70, UAVDT, and VisDrone-SOT2020-test.
