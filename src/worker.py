import argparse
import src.utils as utils
from src.comms.comm import BaseFederatedCommunicator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", required=True)
    parser.add_argument("--local_model_file", required=True)
    parser.add_argument("--global_model", default=None)
    args = parser.parse_args()
    
    communicator = BaseFederatedCommunicator()
    partition = utils.load_pickle(args.partition)
    global_model = utils.load_pickle(args.global_model) if args.global_model else None
    global_model = communicator.train_model(global_model, partition)
    
    utils.save_pickle(global_model, args.local_model_file)

if __name__ == "__main__":
    main()
