import numpy as np
import pandas as pd

d = None
df = None

def process_1K():
    df = pd.read_csv('1K_dataset.csv')
    df["IGS"] = df["NER"].apply(lambda x: set(x.lower().replace('[', '').replace(']', '').replace('\"', '').replace(', ', ',').split(",")))
    d = df["IGS"].values
    np.save('1K_IGS', d)
    return d

def process_full():
    df = pd.read_csv('full_dataset.csv')
    df["IGS"] = df["NER"].apply(lambda x: set(x.lower().replace('[', '').replace(']', '').replace('\"', '').replace(', ', ',').split(",")))
    d = df["IGS"].values
    np.save('full_IGS', d)
    return d

def initialize_1K():
    d = np.load('1k_IGS.npy', allow_pickle=True)
    return d

def initialize_full():
    d = np.load('full_IGS.npy', allow_pickle=True)
    return d

def get_indices(ing_list):
    ingredients = set(ing_list)
    return np.where([ingredients.issubset(x) for x in d])

def get_recipes(ing_list):
    indices = get_indices(ing_list)
    df = pd.read_csv('1K_dataset.csv')
    return df.loc[indices[0], ["title", "link"]]


if __name__ == '__main__':
    d = initialize_1K()
    print(get_recipes(["milk", "butter"]))
