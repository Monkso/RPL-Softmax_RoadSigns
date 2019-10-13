# Radial Prediction Layer

This repository is part of the supplementary material for the paper: Radial Prediction Layer. It contains three Jupyter Notebooks to partially reproduce results reported on the spiral, MNIST and ImageNet datasets. Notebooks using the spiral and MNIST datasets an be executed on CPU. For the experiments with the ImageNet dataset we recommend using PyTorch with GPU support.

## Installation

To run the notebooks you need the corresponding dataset. For the Spiral and MNIST datasets all required resources will be generated or downloaded in the notebooks itself. If you do not want to run the ImageNet-Notebook you can skip the section 'ImageNet Data Preparation'. No additional actions are needed. Simple create an environment to run the notebooks or use the docker container provided by us.

### ImageNet Data Preparation
To reproduce the results on the ImageNet dataset you need to download the ILSVRC data of 2017 and create the subset used in the paper.

1. Download the data of 2017 from the [ImageNet-Website](http://image-net.org/download-images)

2. Clone the git repository into a local directory and change to the data dir of the project.
```bash
git clone https://gitlab.com/peroyose/radial_prediction_layers.git
cd radial_prediction_layers/data
```
3. Move the compressed ImageNet data and unzip it
```bash
mv /path/to/ILSVRC2017_CLS-LOC.tar.gz  .
tar -xvzf ILSVRC2017_CLS-LOC.tar.gz
```
4. Run the necessary preprocessing
```bash
cd ILSVRC/Data/CLS-LOC/val/
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
cd ..
wget -qO- https://gitlab.com/peroyose/radial_prediction_layers/blob/master/data/create_imagenet_subset.sh | bash
```

This process should remove all examples of the *artifact category* tree (WorldnetID: ’n00021939’) from the training data, move all validation examples into class folders and move validation examples from *artifact category* into a new folder called `val_novel`. All examples in `val_novel` are used as unknown and unlabeled data points to evaluate the open world properties of an RPL-network. Your preprocessed ImageNet-Data should be placed in `data` folder of the project to work with the docker container.

### Virtual Environment

You need to install some pip packages (see [`requirement.txt`](https://gitlab.com/peroyose/radial_prediction_layers/blob/master/requirements.txt)) to run all of the notebooks. We recommand the use of a virtual environment. One example installation process using [conda](https://www.anaconda.com/) as virtual environment on Linux (Manjaro) is the following. Depending on the operating system, the process may differ slightly.

1. Create a new virtual environment using Python 3.7
```bash
conda create --name rpl python=3.7.3
conda activate rpl
```

2. Install all necessary basic packages
```bash
pip install jupyter numpy scikit-learn matplotlib tabulate tqdm
```

3. Install PyTorch 1.1 with GPU support (for details regarding your OS see [PyTorch](https://pytorch.org/))
```bash
pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp37-cp37m-linux_x86_64.whl
```

4. Clone the git repository into a local directory (if you not already done that during the ImageNet data preparation) and run jupyter
```bash
git clone https://gitlab.com/peroyose/radial_prediction_layers.git
cd radial_prediction_layers
jupyter notebook
```

### Run Docker Container
The docker is build to run on a linux system.

```bash
docker run --rm -p 127.0.0.1:8888:8888 docker.io/deepprojects/rpl:dev
```

CUDA support in Docker requires the [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) plugin.

```bash
docker run --rm -p 127.0.0.1:8888:8888 -v $(pwd)/data:/home/radial/rpl/data --runtime=nvidia docker.io/deepprojects/rpl:dev-gpu
```

Then open `http://localhost:8888` in your Browser.


## Developer Documentation

Install development environment and manage the `rpl` package via [Poetry](https://github.com/sdispater/poetry)

```bash
poetry install  # creates virtual environment
```

Build and publish on PyPI:

```bash
# change version in pyproject.toml
poetry build
poetry publish
```

### Build Docker Image from Source

Build:

```bash
docker build -t docker.io/deepprojects/rpl:dev .
docker build -t docker.io/deepprojects/rpl:dev-gpu -f Dockerfile.gpu .
```

Publish:

```bash
docker push docker.io/deepprojects/rpl:dev
docker push docker.io/deepprojects/rpl:dev-gpu
```
