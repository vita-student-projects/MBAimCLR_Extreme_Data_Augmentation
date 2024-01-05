# MBAimCLR (based on [MotionBert](https://github.com/Walter0807/MotionBERT) and [AimCLR](https://github.com/Levigty/AimCLR/tree/main))

## Table of Contents :

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Training Dataset](#3-training-dataset)
4. [Training](#4-training)
5. [Linear Evaluation](#5-linear-evaluation)

## 1. Introduction

This project propose a model that uses the self-supervised learning framework from AimCLR to apply extreme data augmentations to the motion encoder DSTformer from MotionBERT. It is trained and evaluate on the NTU RGB+D 60 and 120 dataset. For more information about the model and its performance you can read the files in the folder report.

This project was made in the context of a semester project done at [VITA](https://www.epfl.ch/labs/vita/) lab, for the Robotic Master at EPFL. The goal was to study the impact the framework on a state of the art transformer method for action recognition task.

## 2. Installation

To install the project and be able to run it, you need to follow the following steps:

1. Clone the repository:
2. Install the requirements:

the required packages and python version :

```bash
conda create -n mbpip python=3.7 anaconda
conda activate mbpip
# Please install PyTorch according to your CUDA version.
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```
and torchlight which is a function wrapper for pytorch :

```bash
# Install torchlight
$ cd torchlight
$ python setup.py install
```

4. If you want to use a already trained model, download the checkpoint from [here](https://drive.google.com/drive/u/0/folders/1QsNa07lTkTcnFVrvxEZ1RhJnvW1jtbug).

## 3. Training Dataset

The already generated dataset can be downloaded from : [NTU-60](https://drive.google.com/drive/folders/1WrTG9g-dit7RnaXZ6MR5STOiuaEptfuf) and [NTU-120](https://drive.google.com/drive/folders/1dn8VMcT9BYi0KHBkVVPFpiGlaTn2GnaX). To generate the dataset yourself, see the procedure [here](https://github.com/Levigty/AimCLR/blob/main/README.md) under Data Preparation

## 4. Training

1. Fill the config file corresponding to what you want to train with the paths to the dataset.
4. Run the following command:

#### From scracth :
```bash
python main.py pretrain_mbaimclr --config config/<your_config_file>.yaml
```
#### From a checkpoint :

It is the same command, you just need to adapt the config file with resume set to True and the path to the weights.

### Visualize logs with Tensorboard

The code will automatically create two folders named `train` and `val` in the working directory you put in the config file. You can visualize the logs with Tensorboard by running the following command:

```bash
tensorboard --logdir=<path_to_the_working_directory>/
```

## 5. Linear Evaluation

In this part we will explain how to evaluate the model.

1. Fill the corresponding config file in the `config` folder by putting the path of the model and the data.
4. Run the following command:

```bash
python main.py linear_evaluation_mb --config config/<your_config_file>.yaml
```