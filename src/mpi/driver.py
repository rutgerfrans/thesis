from src.mpi.mpi_comm import MPIFederatedCommunicator

def main():
    communicator = MPIFederatedCommunicator()
    communicator.run()

if __name__ == "__main__":
    main()