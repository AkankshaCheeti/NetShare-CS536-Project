''' This module contains the class necessary to implement CountMinSketch

@author: Peter Xenopoulos
@website: www.peterxeno.com
'''

from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
import sys, configparser, json, random, copy, math, os, pickle
import argparse
import socket
import mmh3

random.seed(42)

class CountMinSketch(object):
    ''' Class for a CountMinSketch data structure
    '''
    def __init__(self, width, depth, seeds):
        ''' Method to initialize the data structure
        @param width int: Width of the table
        @param depth int: Depth of the table (num of hash func)
        @param seeds list: Random seed list
        '''
        self.width = width  #cols
        self.depth = depth  #rows
        self.table = np.zeros([depth, width])  # Create empty table
        self.seed = seeds # np.random.randint(w, size = d) // create some seeds

    def increment(self, key):
        ''' Method to add a key to the CMS
        @param key str: A string to add to the CMS
        '''
        for i in range(0, self.depth):
            index = mmh3.hash(key, self.seed[i]) % self.width
            self.table[i, index] = self.table[i, index]+1

    def estimate(self, key):
        ''' Method to estimate if a key is in a CMS
        @param key str: A string to check
        '''
        min_est = self.width
        for i in range(0, self.depth):
            index = mmh3.hash(key, self.seed[i]) % self.width
            if self.table[i, index] < min_est:
                min_est = self.table[i, index]
        return min_est

    def merge(self, new_cms):
        ''' Method to combine two count min sketches
        @param new_cms CountMinSketch: Another CMS object
        '''
        return self.table + new_cms        

def evaluate_cms(args, file_name):
    seeds = np.random.randint(args.width, size=args.depth)
    cms = CountMinSketch(args.width, args.depth, seeds=seeds)

    # read raw data (correct the path)
    data_df = pd.read_csv(os.path.join(args.dataset, file_name))

    unique_values = set()
    
    print("Populating CMS")
    for _, series in tqdm(data_df.iterrows(), total=1e6):
        value = ''.join([str(series[x]) for x in args.keys])
        cms.increment(value)
        # also populate the unique set
        unique_values.add(value)

    print("Creating Dictionary")
    countDictionary = defaultdict(lambda: 0)
    for _, value in tqdm(data_df.iterrows(), total=1e6):
        value = ''.join([str(series[x]) for x in args.keys])
        countDictionary[value] += 1

    print("Calculating CMS Error")
    errorSum, unique_value_count = 0, 0
    for value in tqdm(unique_values):
        error = abs(cms.estimate(value) - countDictionary[value]) / countDictionary[value]
        errorSum += error
        unique_value_count += 1

    real_error = errorSum / unique_value_count
    return real_error


def main():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--dataset", type=str)
    CLI.add_argument("--keys", type=str, nargs='*')
    CLI.add_argument("--width", type=int)
    CLI.add_argument("--depth", type=int)
    print(vars(CLI.parse_args()))
        
    # convert incoming args to a dictionary
    args = CLI.parse_args()
    print("Evaluating real data..")
    raw_error = evaluate_cms(args, file_name='raw.csv')
    print("Evaluating synthetic data..")
    syn_error = evaluate_cms(args, file_name='syn.csv')
    print(f"Raw Error = {round(raw_error, 2)}")
    print(f"Syn Error = {round(syn_error, 2)}")
    relative_error = round(abs(syn_error - raw_error) / (1e-5 + raw_error) * 100, 2)
    print(f"Relative Error = {relative_error}%")


if __name__ == '__main__':
    main()
    