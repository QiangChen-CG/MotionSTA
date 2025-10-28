# MotionSTA

<!-- TOC -->

- [Installation](#installation)
- [Training](#prepare-environment)
- [Acknowledgement](#acknowledgement)

<!-- TOC -->

## Installation

Please refer to [install.md](install.md) for detailed installation.

## Training

You can run the training code on a single GPU:

```shell
python -u tools/train.py \
    --name t2m_MotionSTA \
    --batch_size 128 \
    --times 25 \
    --num_epochs 50 \
    --dataset_name t2m
```

Here, `times` means the duplication times of the original dataset. To retain the number of iterations, you can set `times` to 25 for 1 GPU.

## Evaluation

```shell
# The GPU_ID parameter designates the target GPU for running
python -u tools/evaluation.py checkpoints/t2m/t2m_MotionSTA/opt.txt GPU_ID

```

## Visualization

```shell
# Currently, the visualization capability is exclusively available for models trained on the HumanML3D dataset.
# The motion length is strictly constrained by the maximum length present in the training dataset, namely 196 frames.
# You can omit `gpu_id` to run visualization on your CPU


python -u tools/visualization.py \
    --opt_path checkpoints/t2m/t2m_MotionSTA/opt.txt \
    --text "a person is jumping" \
    --motion_length 196 \
    --result_path "test_sample.gif" \
    --npy_path "test_sample.npy" \
    --gpu_id 0
```

## Acknowledgement

This code is developed on top of [Generating Diverse and Natural 3D Human Motions from Text](https://github.com/EricGuo5513/text-to-motion)