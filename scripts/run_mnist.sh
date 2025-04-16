#!/bin/bash
cd ..

COMM=$(python3 -c "import config; print(config.COMM)")
NETWORK_ARCHITECTURE=$(python3 -c "import config; print(config.NETWORK_ARCHITECTURE)")
N_EPOCHS=$(python3 -c "import config; print(config.N_EPOCHS)")
N_PARTITIONS=$(python3 -c "import config; print(config.N_PARTITIONS)")
DEBUG=$(python3 -c "import config; print(config.DEBUG)")

if [ "$COMM" == "ipc" ]; then
    echo "Running $COMM driver with $N_EPOCHS epochs, $N_PARTITIONS partitions, and $NETWORK_ARCHITECTURE architecture."
    echo "Debug mode is $DEBUG."
    python3 -m src.drivers.ipc_driver
elif [ "$COMM" == "mpi" ]; then
    echo "Running $COMM driver with $N_EPOCHS epochs, $N_PARTITIONS partitions, and $NETWORK_ARCHITECTURE architecture."
    echo "Debug mode is $DEBUG."
    mpiexec -n $N_PARTITIONS python3 -m src.drivers.mpi_driver
else
    echo "Invalid communication method. Please set COMM to either 'ipc' or 'mpi'."
    exit 1
fi
