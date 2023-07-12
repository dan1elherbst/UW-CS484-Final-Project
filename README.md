# Self-supervised Monocular Depth Estimation

University of Waterloo, CS484: Final Project

Daniel Herbst (`dherbst@uwaterloo.ca`, `daniel.herbst@tum.de`)

This repository contains my final project for the course *CS484: Computational Vision* 
which I took in fall term 2022 as an exchange student at the University of Waterloo. 
Although this course was mainly about conventional (non-ML) methods for Computer Vision, 
the task for this final project was very open, which is why I chose to implement a 
comparatively new, very popular method: The self-supervised approach for monocular depth 
estimation presented in the 2017 CVPR paper 

*Unsupervised Monocular Depth Estimation with Left-Right Consistency* 
by Clément Godard, Oisin Mac Aodha, and Gabriel J. Brostow 
(https://arxiv.org/pdf/1609.03677.pdf).

While the code is more or less exactly what I submitted as my final project, I 
still had some Colab time left after the submission and decided to retrain the model
on a bit more data until I went out of resources. Indeed, these results, while not being 
quite as good as the ones presented in the original paper, are remarkable considering my
computing and time constraints, and the final trained model also does a reasonable job at 
generalizing to unseen data.

## Introduction

Depth estimation is a very typical problem encountered in Computer Vision that has a 
variety of applications, the most prominent one being autonomous driving. While it is 
necessary for autonomous vehicles to map their surroundings in order to navigate 
correctly, there are many ways to achieve this: One can use sophisticated sensors as radar
or even Lidar technologies. However, a completely different and potentially cheaper 
approach is to equip vehicles with cameras and use Computer Vision methods to estimate 
depth from these images, casting these estimated depths out in order to create 3D mappings
of the environment. 

Although there are conventional methods for depth estimation (or equivalently disparity 
estimation) from calibrated stereo images, machine learning approaches using convolutional 
neural networks (CNNs) have been getting increasingly popular over the last years, 
outperforming these conventional methods. While one would need ground truths for a naïve 
supervised learning approach to estimate the disparity maps, such disparity maps might not 
always be available or very tedious/expensive to annotate. The very influential paper from
2017 above resolves this issue by training to estimate disparity maps for rectified stereo
image pairs -- but only taking the left image as an input for the estimation of the 
disparity maps, using the right image in the loss function to check consistency. The 
elegance of this approach is that when such a model is deployed, only the left image is 
needed to perform the actual depth (disparity) estimation which is a clear advantage over 
conventional stereo methods that always require image pairs. This method is called 
self-supervised since there is no annotated ground truth depth information needed for 
training, but only the image pairs -- which makes the collection of massive amounts of 
training data very easy and cheap. 

The original paper comes with a reference implementation in TensorFlow 
(https://github.com/mrharicot/monodepth). While I partly used this to infer tiny 
implementation details that were not entirely clear from the paper, the inspiration 
for the general structure of e.g. the model implementation also comes from this reference 
implementation. Furthermore, this project aims to

- provide a simple and lightweight implementation of the method in the paper using PyTorch
- give a brief overview of this method without the need to read the original paper, 
combining explanations with implementation.

## Getting started

### File structure

- The `mylibs` directory contains most of the code for data loading (`data_loading.py`) as
well as the model (`model.py`) and its loss functions (`losses.py`).
- `download_data.sh` is a script to download the training data I used and transforming it 
into the format needed for data loading.
- `download_models.sh` can be used to download my trained models in order to experiment 
with the results or use the trained models for own evaluations.
- `environment.yml` defines all external Python libraries needed for this project.
- `main.ipynb` is the main file containing all explanations, the training code, as well as
results for this project.

### Data

I trained on a chunk of the popular KITTI dataset which was also used for training in the 
original paper. My chunk consists of 15884 image pairs of street scenes recorded on a car
in Karlsruhe, Germany (this is roughly half of the total training data used by the 
authors). Although training on more data will most likely yield much better results, I 
could not evaluate the method with more data due to time constraints and the lack of 
computational resources (as training on this smaller amount of images, downsampling them 
to roughly half of their original sizes, already took multiple days).

The data which I used for training can be conveniently downloaded and put into the correct
folder structure for my code using the downloading script `download_data.sh`.

```console
bash download_data.sh
```

All data files will be put into the `data` subdirectory.

### Trained models

I saved state dicts of my trained model on Google Drive. The trained model (as well as the
loss curve of training) can be downloaded using `download_models.sh`.

```console
bash download_models.sh
```

The trained model will be put into the `trained_models` subdirectory.

### Code libraries

The dependencies of this project are

- `jupyter` and `jupyterlab`
- `matplotlib`
- `numpy`
- `pandas`
- `pillow`
- `pytorch`
- `torchvision`
- `tqdm`

An environment containing all necessary external libraries in the version I used for the
project can be installed from `environment.yml` and activated using conda:

```console
conda env create -f environment.yml
conda activate UW-CS484-final-project
```
