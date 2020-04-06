
# Versions 2
- Ubuntu 18.04.2 LTS
- 2080TI
- NVIDIA-SMI 430.50
- Cuda V10.1.105
- CUDNN 7.6.5.33

- pytorch 1.3.1
- gcc 6.5.0 (7.4.0 not work for sponv)
- cmake version 3.13.3

- Python 3.7.5

# General packages
## gcc and g++
Gcc 7.4 is not compatiable with cuda10.1 while building sponv. Use Ubuntu 6.5.0-2ubuntu1~18.04

# NVIDIA & CUDA
ref1: https://medium.com/repro-repo/install-cuda-10-1-and-cudnn-7-5-0-for-pytorch-on-ubuntu-18-04-lts-9b6124c44cc  
ref2: https://gist.github.com/Mahedi-61/2a2f1579d4271717d421065168ce6a73   
## clean
```
sudo apt-get purge nvidia*
sudo apt remove nvidia-*
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt-get autoremove && sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*
```

## NVIDIA Driver 430.50
```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-driver-430
reboot
nvidia-smi
```

## CUDA 10.1
```
https://developer.nvidia.com/cuda-toolkit-archive
wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.105-418.39/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
```

```
# CUDA Config - ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64"

source ~/.bashrc                         
cd /usr/local/cuda-10.1/samples
sudo make
/usr/local/cuda-10.1/samples/bin/x86_64/linux/release/deviceQuery
```

## CUDNN 7.6.5.33
```
wget 3 debs
sudo dpkg -i libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7-dev_7.6.5.32-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7-doc_7.6.5.32-1+cuda10.1_amd64.deb
cd /usr/src/cudnn_samples_v7/mnistCUDNN/
sudo make clean && sudo make
./mnistCUDNN
```

## Cmake 3.13.5
```
Download
./bootstrap  (no sudo)
make -j
sudo make install
```

## Eenvironments
```
Install miniconda
conda create --name Scan2BIM -y
conda activate Scan2BIM
conda install ipython pip
```

## dependencies
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install scipy numba pillow matplotlib scikit-image
pip install ninja yacs cython tqdm opencv-python  fire tensorboardX protobuf
conda install -c open3d-admin open3d
```

# Relied projects
## maskrcnn
ref: https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md
New features are added, .so generated by original project would not work.
```  
export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install cityscapesScripts
cd $INSTALL_DIR
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

cd Detection_3D/maskrcnn-benchmark
./build.sh
./cp_so.sh
```

## second
ref: https://github.com/traveller59/second.pytorch  
Setup cuda for numba: add following to ~/.bashrc: 
``` bash
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so 
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so 
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice 
```


## SparseConvCnn
ref: https://github.com/facebookresearch/SparseConvNet  
New features are added, SCN.**.so from original project would not work.
``` bash
cd Detection_3D/SparseConvNet
conda install google-sparsehash -c bioconda
conda install -c anaconda pillow
bash develop.sh 
copy SCN.cpython-37m-x86_64-linux-gnu.so
```

## SpConv        
ref: https://github.com/traveller59/spconv  
later, SpConv and SparseConvCnn should only need to install one
```
git clone https://github.com/traveller59/spconv.git --recursive
sudo apt-get install libboost-all-dev
python setup.py bdist_wheel
cd ./dist
pip install spconv-1.1-cp37-cp37m-linux_x86_64.whl
```

## Optinal
- Pymesh: https://pymesh.readthedocs.io/en/latest/installation.html
- pip install plyfile
