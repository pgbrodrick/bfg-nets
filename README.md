# rsCNN

## Motivation

rsCNN is a package for using convolutional neural networks (CNNs) in remote sensing projects. This is an internal research tool that manages several components of the research pipeline, from building data to training models to reporting and visualizing results. The primary goals of rsCNN are to save time by not rewriting boilerplate code, reduce errors by reusing trusted code, and explore models more deeply by using standardized configuration files. 

As an internal research tool, rsCNN is a living codebase that is currently targeted toward our research projects and interests. It also comes with the disclaimer that it is a by-product of our existing research commitments and is not guaranteed to be bug-free, as well as not having extensive test coverage or other best practices. While we're working in that direction, we wanted to provide everyone access to rsCNN in the hopes that others would find it useful.

We welcome contributions to rsCNN. Please reach out if you'd like to discuss the package further.

## Alternatives

rsCNN is an extension of [ecoCNN](https://github.com/pgbrodrick/ecoCNN), which was published with the paper "[Uncovering ecological patterns with convolutional neural networks](https://www.sciencedirect.com/science/article/pii/S0169534719300862?via%3Dihub)". ecoCNN is recommended for those that want a simpler code base, or something that is written more linearly.  rsCNN is recommended for those who want a deeper dive, are looking for a living codebase, or who want to use the code as something closer to a standardized package.

## Installation

### GPU-compatible

This option is recommended because GPUs are orders of magnitude more efficient at training and applying neural networks.

1. Install anaconda or miniconda.
1. Startup a GPU node if working on a distributed computing environment.
1. `conda env create --name=myenv --file=environment_GPU.yaml`
1. `conda activate myenv`
1. `conda env update --file=environment.yaml`

## CPU-compatible

1. Install anaconda or miniconda.
1. `conda env create --name=envname --file=environment.yaml`
