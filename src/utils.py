import pickle

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
    
def save_pickle(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)