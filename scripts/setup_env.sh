conda create -n dist_mnist -c conda-forge python=3.11 openmpi mpi4py
conda activate dist_mnist
pip3 install numpy
pip3 install syndicate-py