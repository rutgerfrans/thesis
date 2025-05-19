from src.comms.comm import BaseFederatedCommunicator
from mpi4py import MPI

class MPIFederatedCommunicator(BaseFederatedCommunicator):
    def __init__(self):
        self.mpi_comm = MPI.COMM_WORLD
        self.data = None
        self.local_model = None

    def distribute_data(self, data, epoch):
        self.data = self.mpi_comm.scatter(data, root=0)
        if self.data[0] is not None:
            self.local_model = self.train_model(self.data[1], self.data[0])

    def collect_models(self): # maybe create a try except here
        return self.mpi_comm.gather(self.local_model, root=0)
    
    def create_data_stack(self, global_model, partitions):
        return [[None, None]] + [(part, global_model) for part in partitions]
    
    def update_model(self, models, partitions, test_set, epoch):
        if self.mpi_comm.Get_rank() == 0:
            if None in models: models.remove(None)
            return super().update_model(models, partitions, test_set, epoch)
        
def ulfm_support_test():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 1) Check that mpi4py exposes any MPIX_ symbols
    sym = [name for name in dir(MPI) if name.startswith("MPIX_")]
    if not sym:
        print(f"[{rank}] No ULFM symbols found! ULFM not available.")
        return
    if rank == 0:
        print("ULFM symbols available:", sym)

    # 2) Attach an error handler so failures raise exceptions
    errh = MPI.Errhandler.Create(MPI.ERRORS_RETURN)
    comm.Set_errhandler(errh)

    # 3) Simulate a failure on rank 1
    if rank == 1:
        # we deliberately do something illegal
        import sys; sys.exit(1)

    # 4) Try a collective and catch the ULFM exception
    try:
        comm.Allreduce([rank, MPI.INT], None, op=MPI.SUM)
        if rank == 0:
            print("Allreduce succeeded (no failure detected)")
    except MPI.Exception as e:
        print(f"[{rank}] Caught MPI.Exception: {e!r}")
        # 5) Revoke the communicator, shrink it, and show new size
        comm.Revoke()
        comm.Fail_ack()
        newcomm = comm.Shrink()
        if rank != 1:  # rank 1 has died
            print(f"[{rank}] New communicator size after shrink: {newcomm.Get_size()}")

if __name__ == "__main__":
    ulfm_support_test()
    
