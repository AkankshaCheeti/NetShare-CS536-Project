from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
import sys, configparser, json, random, copy, math, os, pickle
import argparse
import socket

import random
import math
from collections import Counter

from cms_mmh3 import CountMinSketch as CMS_MMH3
from cms_horner import CountMinSketch as CMS_HORNER
from cms_csiphash import CountMinSketch as CMS_CSIPHASH

CMS_COLLECTION = {
    "mmh3":     CMS_MMH3,
    "csiphash": CMS_CSIPHASH,
    "horner":   CMS_HORNER
}

def evaluate_cms_single_key(cms, dataset, key, file_name):
    # read raw data (correct the path)
    data_df = pd.read_csv(os.path.join(dataset, file_name))
    unique_values = set()
    
    print("Populating CMS")
    for value in tqdm(data_df[key], total=1000000):
        value_in_bytes = value.to_bytes(4, byteorder='big')
        cms.increment(value_in_bytes)
        # also populate the unique set
        unique_values.add(value_in_bytes)

    print("Creating Dictionary")
    countDictionary = defaultdict(lambda: 0)
    for value in tqdm(data_df[key], total=1000000):
        value_in_bytes = value.to_bytes(4, byteorder='big')
        countDictionary[value_in_bytes] += 1

    print("Calculating CMS Error")
    errorSum, unique_value_count = 0, 0
    for value_in_bytes in tqdm(unique_values):
        # assert cms.get_min_count(value_in_bytes) == 0
        # assert countDictionary[value_in_bytes] == 0
        error = abs(cms.estimate(value_in_bytes) - countDictionary[value_in_bytes])
        errorSum += error
        unique_value_count += 1

    real_error = errorSum / unique_value_count
    return real_error


def evaluate_cms_multiple_keys(cms, dataset, keys, file_name):
    # read raw data (correct the path)
    data_df = pd.read_csv(os.path.join(dataset, file_name))
    # group the data by the keys
    print("Grouping the Data Frame by keys")
    grouped_data_df = data_df.groupby(keys)
    unique_values = set()
    
    print("Populating CMS")
    for _tuple, series in tqdm(grouped_data_df[keys]):
        value_in_bytes = ''.join([str(x) for x in _tuple])
        for _ in range(len(series)):
            cms.increment(value_in_bytes)
        # also populate the unique set
        unique_values.add(value_in_bytes)

    print("Creating Dictionary")
    countDictionary = defaultdict(lambda: 0)
    for _tuple, series in tqdm(grouped_data_df[keys]):
        value_in_bytes = ''.join([str(x) for x in _tuple])
        for _ in range(len(series)):
            countDictionary[value_in_bytes] += 1

    print("Calculating CMS Error")
    errorSum, unique_value_count = 0, 0
    for value_in_bytes in tqdm(unique_values):
        # assert cms.get_min_count(value_in_bytes) == 0
        # assert countDictionary[value_in_bytes] == 0
        error = abs(cms.estimate(value_in_bytes) - countDictionary[value_in_bytes])
        errorSum += error
        unique_value_count += 1

    real_error = errorSum / unique_value_count
    return real_error


def main():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--dataset", type=str)
    CLI.add_argument("--keys", type=str, nargs='*')
    CLI.add_argument("--hash", type=str)
    CLI.add_argument("--width", type=int)
    CLI.add_argument("--depth", type=int)
    args = CLI.parse_args()
    print(vars(args))
    
    cms = CMS_COLLECTION[args.hash](args.width, args.depth)
    
    # convert incoming args to a dictionary
    if len(args.keys) == 1:
        print("Evaluating real data..")
        raw_error = evaluate_cms_single_key(cms, args.dataset, key=args.keys[0], file_name='raw.csv')
        print("Evaluating synthetic data..")
        syn_error = evaluate_cms_single_key(cms, args.dataset, key=args.keys[0], file_name='syn.csv')
    else:
        print("Evaluating real data..")
        raw_error = evaluate_cms_multiple_keys(cms, args.dataset, keys=args.keys, file_name='raw.csv')
        print("Evaluating synthetic data..")
        syn_error = evaluate_cms_multiple_keys(cms, args.dataset, keys=args.keys, file_name='syn.csv')
    print(f"Raw Error = {round(raw_error, 2)}")
    print(f"Syn Error = {round(syn_error, 2)}")
    relative_error = round(abs(syn_error - raw_error) / (1e-5 + raw_error) * 100, 2)
    print(f"Relative Error = {relative_error}%")


if __name__ == '__main__':
    main()
