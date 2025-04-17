import subprocess, pickle, os
import config
import src.utils as utils
from src.comms.comm import BaseFederatedCommunicator

class IPCFederatedCommunicator(BaseFederatedCommunicator):
    def __init__(self):
        self.global_model_file = "tmp/global_model.pkl"
        self.partition_file = "tmp/partition_"
        self.local_model_file = "tmp/local_model_"

    def create_partition_files(self, partitions):
        partition_files = []
        for i, part in enumerate(partitions):
            utils.save_pickle(part, self.partition_file+f"{i}.pkl")
            partition_files.append(self.partition_file+f"{i}.pkl")
        return partition_files
   
    def distribute_data(self, global_model, partition_files):
        processes = []
        local_model_files = []

        for i, pfile in enumerate(partition_files):
            local_model_files.append(self.local_model_file+f"{i}.pkl")

            cmd = ["python3", "-m", "src.worker", "--partition", pfile, "--local_model_file", self.local_model_file+f"{i}.pkl"]
            if global_model: 
                utils.save_pickle(global_model, self.global_model_file)
                cmd += ["--global_model", self.global_model_file]

            proc = subprocess.Popen(cmd, stdout=None if config.DEBUG else subprocess.PIPE,
                                         stderr=None if config.DEBUG else subprocess.PIPE,
                                         text=True)
            processes.append((proc, i, self.local_model_file+f"{i}.pkl"))
        return processes, local_model_files

    def wait_for_completion(self, processes):
        for proc, i, _ in processes:
            stdout, stderr = proc.communicate()
            if proc.returncode != 0 and config.DEBUG:
                print(f"Error in worker {i}:")
                print("STDOUT:", stdout.decode("utf-8"))
                print("STDERR:", stderr.decode("utf-8"))

    def collect_models(self, local_model_files):
        models = []
        for fname in local_model_files:
            models.append(utils.load_pickle(fname))
        return models

    def cleanup(self, files):
        for f in files:
            if os.path.exists(f):
                os.remove(f)
