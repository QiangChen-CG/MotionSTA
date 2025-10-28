# Installation

<!-- TOC -->

- [Requirements](#requirements)
- [Prepare environment](#prepare-environment)
- [Data Preparation](#data-preparation)

<!-- TOC -->

## Requirements

- Linux
- Python 3.7+
- PyTorch 1.6.0, 1.7.0, 1.7.1, 1.8.0, 1.8.1, 1.9.0 or 1.9.1.
- CUDA 9.2+
- GCC 5+
- [MMCV](https://github.com/open-mmlab/mmcv) (Please install mmcv-full>=1.3.17,<1.6.0 for GPU)

## Prepare environment

a. Create a conda virtual environment and activate it.

```shell
conda create -n MotionSTA python=3.7 -y
conda activate MotionSTA
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).
```shell
conda install pytorch={torch_version} torchvision cudatoolkit={cu_version} -c pytorch
```

E.g., install PyTorch 1.7.1 & CUDA 10.1.
```shell
conda install pytorch=1.7.1 torchvision cudatoolkit=10.1 -c pytorch
```

**Important:** Make sure that your compilation CUDA version and runtime CUDA version match.

c. Build mmcv-full

- mmcv-full

We recommend you to install the pre-build package as below.

For CPU:
```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/{torch_version}/index.html
```
Please replace `{torch_version}` in the url to your desired one.

For GPU:
```shell
pip install "mmcv-full>=1.3.17,<=1.5.3" -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```
Please replace `{cu_version}` and `{torch_version}` in the url to your desired one.

For example, to install mmcv-full with CUDA 10.1 and PyTorch 1.7.1, use the following command:
```shell
pip install "mmcv-full>=1.3.17,<=1.5.3" -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.1/index.html
```

See [here](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) for different versions of MMCV compatible to different PyTorch and CUDA versions.
For more version download link, refer to [openmmlab-download](https://download.openmmlab.com/mmcv/dist/index.html).


d. Install other requirements

```shell
pip install -r requirements.txt
```

## Data Preparation

Download datasets

For both the HumanML3D dataset and the KIT-ML dataset, you could find the details as well as download link [[here]](https://github.com/EricGuo5513/HumanML3D).

Download the above resources and arrange them in the following file structure:

```text
MotionSTA
└── text2motion
    ├── checkpoints
    │   ├── kit
    │   │   └── kit_MotionSTA
    │   │       ├── meta
    │   │       │   ├── mean.npy
    │   │       │   └── std.npy
    │   │       ├── model
    │   │       │   └── latest.tar
    │   │       └── opt.txt
    │   └── t2m
    │       └── t2m_MotionSTA
    │           ├── meta
    │           │   ├── mean.npy
    │           │   └── std.npy
    │           ├── model
    │           │   └── latest.tar
    │           └── opt.txt
    └── data
        ├── glove
        │   ├── our_vab_data.npy
        │   ├── our_vab_idx.pkl
        │   └── out_vab_words.pkl
        ├── pretrained_models
        │   ├── kit
        │   │   └── text_mot_match
        │   │       └── model
        │   │           └── finest.tar
        │   └── t2m
        │   │   ├── text_mot_match
        │   │   │   └── model
        │   │   │       └── finest.tar
        │   │   └── length_est_bigru
        │   │       └── model
        │   │           └── finest.tar
        ├── HumanML3D
        │   ├── new_joint_vecs
        │   │   └── ...
        │   ├── new_joints
        │   │   └── ...
        │   ├── texts
        │   │   └── ...
        │   ├── Mean.npy
        │   ├── Std.npy
        │   ├── test.txt
        │   ├── train_val.txt
        │   ├── train.txt
        │   └── val.txt
        └── KIT-ML
            ├── new_joint_vecs
            │   └── ...
            ├── new_joints
            │   └── ...
            ├── texts
            │   └── ...
            ├── Mean.npy
            ├── Std.npy
            ├── test.txt
            ├── train_val.txt
            ├── train.txt
            └── val.txt
```