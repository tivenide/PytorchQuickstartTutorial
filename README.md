# PytorchQuickstartTutorial
This repository contains the basic ideas for creating a containerized CUDA machine learning workflow with pytorch and docker.
A docker installation and setting up the NVIDIA environment is required.
The project was developed under Python 3.11 and Linux Ubuntu 22.04 lts.
It is based on: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
## Usage
Building container
```
sudo docker build -t pqtapp:latest .
```
Ensure the presence of the local file structure
```commandline
workdir_pqtapp/
├── input
│   └── FashionMNIST
│       └── raw
│           ├── t10k-images-idx3-ubyte
│           ├── t10k-images-idx3-ubyte.gz
│           ├── t10k-labels-idx1-ubyte
│           ├── t10k-labels-idx1-ubyte.gz
│           ├── train-images-idx3-ubyte
│           ├── train-images-idx3-ubyte.gz
│           ├── train-labels-idx1-ubyte
│           └── train-labels-idx1-ubyte.gz
└── output
    └── model.pth (here exports the container the output)


```

Running container
```
sudo docker run --gpus all -v $(pwd)/workdir_pqtapp/input:/PytorchQuickstartTutorial/data/input -v $(pwd)/workdir_pqtapp/output:/PytorchQuickstartTutorial/data/output pqtapp
```