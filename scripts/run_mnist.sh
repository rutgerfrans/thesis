#!/bin/bash
cd ..

COMM=$(python3 -c "import config; print(config.COMM)")
NETWORK_ARCHITECTURE=$(python3 -c "import config; print(config.NETWORK_ARCHITECTURE)")
N_EPOCHS=$(python3 -c "import config; print(config.N_EPOCHS)")
N_PARTITIONS=$(python3 -c "import config; print(config.N_PARTITIONS)")
DEBUG=$(python3 -c "import config; print(config.DEBUG)")

echo "==============================="
echo "Running MNIST"
echo "Comm Prot: $COMM"
echo "NN ARCH: $NETWORK_ARCHITECTURE"
echo "N_EPOCHS: $N_EPOCHS"
echo "N_PARTITIONS: $N_PARTITIONS"
echo "==============================="

if [ "$COMM" == "ipc" ]; then
    python3 -m src.drivers.driver
elif [ "$COMM" == "mpi" ]; then
    mpiexec -n $N_PARTITIONS python3 -m src.drivers.driver
fi
