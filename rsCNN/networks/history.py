import pickle


def load_history(filepath):
    with open(filepath, 'rb') as file_:
        history = pickle.load(file_)
    return history


def save_history(history, filepath):
    with open(filepath, 'wb') as file_:
        pickle.dump(history, file_)
