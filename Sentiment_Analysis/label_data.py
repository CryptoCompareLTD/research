"""
Used to manually label headlines. Have fun!!

"""

import pickle
import pandas as pd
import numpy as np

def label_data(data):
    label = []
    for d in data:
        print(d, '\n')
        label.append(raw_input("Is this positive (p), negative (n) or neutral (k)? \n"))
    return label 

def main():
    df = pd.read_pickle('Data/headlines.p')
    data = df['headline'].values.tolist()
    label = np.array(label_data(data))
    df['label'] = label
    df.head(10)
    df.to_pickle('labelled_headlines.p')

if __name__ == "__main__":
    main()