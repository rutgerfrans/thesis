conda create -n dist_mnist
conda activate dist_mnist
conda install openmpi mpi4py -c conda-forge
pip3 install numpy
pip3 install syndicate-py