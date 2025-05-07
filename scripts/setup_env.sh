conda create -n dist_mnist -c conda-forge python=3.11 openmpi mpi4py
conda activate dist_mnist
pip install numpy
pip install syndicate-py