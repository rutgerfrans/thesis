import subprocess, os
import src.utils as utils
from src.comms.comm import BaseFederatedCommunicator

class IPCFederatedCommunicator(BaseFederatedCommunicator):
    def __init__(self, test_module=None):
        self.global_model_file = "tmp/global_model.pkl"
        self.partition_file = "tmp/partition_"
        self.local_model_file = "tmp/local_model_"
        self.partition_files = []
        self.local_model_files = []
        self.test_module = test_module

    def create_partition_files(self, partitions):
        self.partition_files = []
        for i, part in enumerate(partitions):
            utils.save_pickle(part, self.partition_file+f"{i}.pkl")
            self.partition_files.append(self.partition_file+f"{i}.pkl")
        return self.partition_files
    
    def create_data_stack(self, global_model, partitions):
        return [[part, global_model] for part in self.create_partition_files(partitions)]
    
    def distribute_data(self, data_stack, epoch):
        processes = []
        workers = []
        self.cleanup(self.local_model_files + [self.global_model_file])
        self.local_model_files = []
        for i, data in enumerate(data_stack):
            self.local_model_files.append(self.local_model_file+f"{i}.pkl")
            cmd = ["python3", "-m", "src.worker", "--partition", data[0], "--local_model_file", self.local_model_file+f"{i}.pkl"]
            if data[1]: 
                utils.save_pickle(data[1], self.global_model_file)
                cmd += ["--global_model", self.global_model_file]

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            workers.append(proc)
            self.test_module.simulate(epoch, workers)
            #if i != 1: self.test_module.crash_worker(proc)
            
            processes.append((proc, i, self.local_model_file+f"{i}.pkl"))
        self.wait_for_completion(processes)
        return self.local_model_files
   
    def wait_for_completion(self, processes):
        j = 0
        for proc, i, _ in processes:
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                print(f"Error in worker {i}:", " STDOUT:", stdout.encode("utf-8"), " STDERR:", stderr.encode("utf-8"))
                j=j+1
        print("Epoch finished with ", len(processes)-j, " out of the ", len(processes), " processes succesfully finished")

    def collect_models(self):
        models = []
        for fname in self.local_model_files:
            if os.path.exists(fname):
                models.append(utils.load_pickle(fname))
        return models

    def cleanup(self, files):
        for f in files:
            if os.path.exists(f):
                os.remove(f)
