import json


def load_history(filepath):
    with open(filepath, 'r') as file_:
        history = json.load(file_)
    return history


def save_history(history, filepath):
    with open(filepath, 'wb') as file_:
        json.dump(history, file_)


