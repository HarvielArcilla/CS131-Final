import numpy as np
import pandas as pd
import pickle 
from itertools import chain 
from itertools import combinations
from functools import reduce

d = None
df = None
freq = {}

def process_optimized_full():
    df = pd.read_csv('full_dataset.csv')
    with open('full_dataset.pkl', 'wb') as fi:
        pickle.dump(df, fi)
    freq = {}
    for i in range(df.shape[0]):
        for ing in df["NER"][i].lower().replace('[', '').replace(']', '').replace('\"', '').replace(', ', ',').split(","):
            if ing in freq: freq[ing].append(i)
            else: freq[ing] = [i]
        print(i)
    with open('prcomputed_indices_full.pkl', 'wb') as f:
        pickle.dump(freq, f)
    return df, freq


def initialize_full():
    df = pd.read_csv('full_dataset.csv')
    d = np.load('full_IGS.npy', allow_pickle=True)
    return df, d


def initialize_optimized_full():
    df = pd.read_pickle('full_dataset.pkl')
    with open('prcomputed_indices_full.pkl', 'rb') as f:
        freq = pickle.load(f)
    return df, freq

def initialize_optimized(rel_path):
    df_pkl = pd.read_pickle(rel_path + 'full_dataset.pkl')
    with open(rel_path + 'prcomputed_indices_full.pkl', 'rb') as f:
        freq_pkl = pickle.load(f)
    global df 
    df = df_pkl
    global freq 
    freq = freq_pkl

# from itertools
def powerset(iterable):
    "powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))


def get_indices(ing_list):
    ingredients = set(ing_list)
    return np.where([ingredients.issubset(x) for x in d])


def get_indices_optimized(ing_list):
    if set(ing_list).issubset(freq.keys()):
        base = freq[ing_list[0]]
        for ing in ing_list:
            base = np.intersect1d(base, freq[ing])
        return base
    else: return []


def get_recipes(ing_list):
    indices = get_indices(ing_list)
    return df.loc[indices[0], ["title", "link"]]


def get_recipes_optimized(ing_list):
    indices = get_indices_optimized(ing_list)
    return df.loc[indices, ["title", "link"]]


if __name__ == '__main__':
    # process_optimized_full()
    df, freq = initialize_optimized_full()
    while True:
        food = input("Enter a food (Use commas w/o space): ")
        # print(get_recipes_optimized(food.split(",")))
        # print(powerset(food.split(",")))
        dfs = []
        food_list = food.split(",")
        if len(food_list) >= 3:
            for i in powerset(food_list):
                if len(i) >= 3:
                    dfs.append(get_recipes_optimized(i))
            print(reduce(lambda x, y: x.merge(y, how='outer', on=['title','link']), dfs))
        else: 
            print(get_recipes_optimized(food_list))


    
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

def process_optimized_1K():
    df = pd.read_csv('1K_dataset.csv')
    freq = {}
    for i in range(df.shape[0]):
        for ing in df["NER"][i].lower().replace('[', '').replace(']', '').replace('\"', '').replace(', ', ',').split(","):
            if ing in freq: freq[ing].append(i)
            else: freq[ing] = [i]
        print(i)
    with open('prcomputed_indices_1K.pkl', 'wb') as f:
        pickle.dump(freq, f)
    return df, freq

def initialize_1K():
    df = pd.read_csv('1K_dataset.csv')
    d = np.load('1k_IGS.npy', allow_pickle=True)
    return df, d

def initialize_optimized_1K():
    df = pd.read_csv('full_dataset.csv')
    with open('prcomputed_indices_1K.pkl', 'rb') as f:
        freq = pickle.load(f)
    return df, freq
