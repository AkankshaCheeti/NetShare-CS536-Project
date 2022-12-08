from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
import sys, configparser, json, random, copy, math, os, pickle
import matplotlib.pyplot as plt
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

def evaluate_cms_single_key(dataset, cms_hash_function, cms_width_scale, cms_depth, key, heavy_hitter_percentile, file_name):
    # read raw data (correct the path)
    data_df = pd.read_csv(os.path.join(dataset, file_name))
    
    unique_values = set()
    print("Extracting Unique Values from Dataset")
    for value in tqdm(data_df[key], total=1000000):
        value_in_bytes = value.to_bytes(4, byteorder='big')
        # also populate the unique set
        unique_values.add(value_in_bytes)
    
    cms_width = int(cms_width_scale * len(unique_values))
    print(f"Unique Keys in this dataset = {len(unique_values)}")
    print(f"Building CMS with Width={cms_width} Depth={cms_depth}")
    if cms_hash_function == "csiphash":
        cms = CMS_COLLECTION[cms_hash_function](cms_depth, cms_width)
    else:
        cms = CMS_COLLECTION[cms_hash_function](cms_width, cms_depth)

    print("Populating CMS")
    for value in tqdm(data_df[key], total=1000000):
        value_in_bytes = value.to_bytes(4, byteorder='big')
        cms.increment(value_in_bytes)

    print("Converting CMS to Dictionary")
    cms_dictionary = {}
    for value_in_bytes in tqdm(unique_values):
        cms_dictionary[value_in_bytes] = cms.estimate(value_in_bytes)

    print("Evaluating Top-{}% Heavy Hitters in CMS".format(heavy_hitter_percentile))
    sorted_cms_dictionary = {k: v for k, v in sorted(cms_dictionary.items(), key=lambda item: item[1], reverse=True)}

    print("Creating All-bins Dictionary")
    countDictionary = defaultdict(lambda: 0)
    for value in tqdm(data_df[key], total=1000000):
        value_in_bytes = value.to_bytes(4, byteorder='big')
        countDictionary[value_in_bytes] += 1
    
    print("Evaluating Top-{}% Heavy Hitters in All-bins Dictionary".format(heavy_hitter_percentile))
    sorted_countDictionary = {k: v for k, v in sorted(countDictionary.items(), key=lambda item: item[1], reverse=True)}

    # get top-n count number
    top_n_count = int(heavy_hitter_percentile/100 * len(unique_values))

    print("Extracting Top-{}% Heavy Hitters from CMS".format(heavy_hitter_percentile))
    cms_dictionary_heavy_hitters = []
    for i, (key, value) in enumerate(sorted_cms_dictionary.items()):
        cms_dictionary_heavy_hitters.append(key)
        if i >= top_n_count:
            break
    
    print("Extracting Top-{}% Heavy Hitters from All-bins Dictionary".format(heavy_hitter_percentile))
    countDictionary_heavy_hitters = []
    for i, (key, value) in enumerate(sorted_countDictionary.items()):    
        countDictionary_heavy_hitters.append(key)
        if i >= top_n_count:
            break

    #  Heavy-hitter error = (number of flows in both lists)/((number of flows in gt)+( number of flows in cms) -(number of flows in both))
    common_keys = list(set(cms_dictionary_heavy_hitters) & set(countDictionary_heavy_hitters))
    
    print("Top-{}% Heavy Hitters:".format(heavy_hitter_percentile))
    heavy_hitter_accuracy = 100 * len(common_keys) / (len(countDictionary_heavy_hitters) + len(cms_dictionary_heavy_hitters) - len(common_keys))
    heavy_hitter_error = 100 - heavy_hitter_accuracy
    print(f"\t{heavy_hitter_percentile}% Heavy-Hitter Count = {len(cms_dictionary_heavy_hitters)}")
    print(f"\tMatches Found = {len(common_keys)}")
    print(f"\t{heavy_hitter_percentile}% Heavy-Hitter Error Rate = {round(heavy_hitter_error, 2)}%")
    
    return heavy_hitter_error


def evaluate_cms_multiple_keys(dataset, cms_hash_function, cms_width_scale, cms_depth, keys, heavy_hitter_percentile, file_name):
    # read raw data (correct the path)
    data_df = pd.read_csv(os.path.join(dataset, file_name))
    # group the data by the keys
    print("Grouping the Data Frame by keys")
    grouped_data_df = data_df.groupby(keys)
    
    unique_values = set()
    print("Extracting Unique Values from Dataset")
    for _tuple, series in tqdm(grouped_data_df[keys]):
        value_in_bytes = ''.join([str(x) for x in _tuple]).encode()
        # also populate the unique set
        unique_values.add(value_in_bytes)
    
    cms_width = int(cms_width_scale * len(unique_values))
    print(f"Unique Keys in this dataset = {len(unique_values)}")
    print(f"Building CMS with Width={cms_width} Depth={cms_depth}")
    if cms_hash_function == "csiphash":
        cms = CMS_COLLECTION[cms_hash_function](cms_depth, cms_width)
    else:
        cms = CMS_COLLECTION[cms_hash_function](cms_width, cms_depth)
    
    print("Populating CMS")
    for _tuple, series in tqdm(grouped_data_df[keys], len(unique_values)):
        value_in_bytes = ''.join([str(x) for x in _tuple]).encode()
        for _ in range(len(series)):
            cms.increment(value_in_bytes)

    print("Converting CMS to Dictionary")
    cms_dictionary = {}
    for value_in_bytes in tqdm(unique_values):
        cms_dictionary[value_in_bytes] = cms.estimate(value_in_bytes)

    print("Evaluating Heavy Hitters in CMS")
    sorted_cms_dictionary = {k: v for k, v in sorted(cms_dictionary.items(), key=lambda item: item[1], reverse=True)}

    print("Creating All-bins Dictionary")
    countDictionary = defaultdict(lambda: 0)
    for _tuple, series in tqdm(grouped_data_df[keys], len(unique_values)):
        value_in_bytes = ''.join([str(x) for x in _tuple]).encode()
        for _ in range(len(series)):
            countDictionary[value_in_bytes] += 1
    
    print("Evaluating Heavy Hitters in All-bins Dictionary")
    sorted_countDictionary = {k: v for k, v in sorted(countDictionary.items(), key=lambda item: item[1], reverse=True)}

    # get top-n count number
    top_n_count = int(heavy_hitter_percentile/100 * len(unique_values))

    print("Extracting Top-{}% Heavy Hitters from CMS".format(heavy_hitter_percentile))
    cms_dictionary_heavy_hitters = []
    for i, (key, value) in enumerate(sorted_cms_dictionary.items()):    
        cms_dictionary_heavy_hitters.append(key)
        if i >= top_n_count:
            break
    
    print("Extracting Top-{}% Heavy Hitters from All-bins Dictionary".format(heavy_hitter_percentile))
    countDictionary_heavy_hitters = []
    for i, (key, value) in enumerate(sorted_countDictionary.items()):    
        countDictionary_heavy_hitters.append(key)
        if i >= top_n_count:
            break

    #  Heavy-hitter error = (number of flows in both lists)/((number of flows in gt)+( number of flows in cms) -(number of flows in both))
    common_keys = list(set(cms_dictionary_heavy_hitters) & set(countDictionary_heavy_hitters))
    
    print("Top-{}% Heavy Hitters:".format(heavy_hitter_percentile))
    heavy_hitter_accuracy = 100 * len(common_keys) / (len(countDictionary_heavy_hitters) + len(cms_dictionary_heavy_hitters) - len(common_keys))
    heavy_hitter_error = 100 - heavy_hitter_accuracy
    print(f"\t{heavy_hitter_percentile}% Heavy-Hitter Count = {len(cms_dictionary_heavy_hitters)}")
    print(f"\tMatches Found = {len(common_keys)}")
    print(f"\t{heavy_hitter_percentile}% Heavy-Hitter Error Rate = {round(heavy_hitter_error, 2)}%")
    
    return heavy_hitter_error


def main():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--dataset", type=str)
    CLI.add_argument("--results", type=str)
    CLI.add_argument("--keys", type=str, nargs='*')
    CLI.add_argument("--hash", type=str)
    CLI.add_argument("--width_scales", type=float, nargs='*')
    CLI.add_argument("--depth", type=int)
    CLI.add_argument("--percentile", type=float)
    args = CLI.parse_args()
    print(vars(args))
    
    raw_errors, syn_errors = [], []
    for width_scale in args.width_scales:
        # convert incoming args to a dictionary
        print(f"\n[Run {width_scale}] Using Width Scale = {width_scale}")
        if len(args.keys) == 1:
            print("Evaluating real data with a single key..")
            raw_error = evaluate_cms_single_key(dataset=args.dataset, 
                                                cms_hash_function=args.hash, 
                                                cms_width_scale=width_scale, 
                                                cms_depth=args.depth, 
                                                key=args.keys[0], 
                                                heavy_hitter_percentile=args.percentile, 
                                                file_name='raw.csv')
            print("Evaluating synthetic data with a single key..")
            syn_error = evaluate_cms_single_key(dataset=args.dataset, 
                                                cms_hash_function=args.hash, 
                                                cms_width_scale=width_scale, 
                                                cms_depth=args.depth, 
                                                key=args.keys[0], 
                                                heavy_hitter_percentile=args.percentile, 
                                                file_name='syn.csv')
        else:
            print("Evaluating real data with multiple keys..")
            raw_error = evaluate_cms_multiple_keys(dataset=args.dataset, 
                                                cms_hash_function=args.hash, 
                                                cms_width_scale=width_scale, 
                                                cms_depth=args.depth, 
                                                keys=args.keys, 
                                                heavy_hitter_percentile=args.percentile, 
                                                file_name='raw.csv')
            print("Evaluating synthetic data with multiple keys..")
            syn_error = evaluate_cms_multiple_keys(dataset=args.dataset, 
                                                cms_hash_function=args.hash, 
                                                cms_width_scale=width_scale, 
                                                cms_depth=args.depth, 
                                                keys=args.keys, 
                                                heavy_hitter_percentile=args.percentile, 
                                                file_name='syn.csv')
        print(f"Raw Error = {round(raw_error, 2)}")
        print(f"Syn Error = {round(syn_error, 2)}")
        raw_errors.append(raw_error)
        syn_errors.append(syn_error)
        
        if raw_error == 0:
            relative_error = round(abs(syn_error - raw_error) * 100, 2)
        elif raw_error > syn_error:
            relative_error = 0
        else:
            relative_error = round(abs(syn_error - raw_error) * 100 / raw_error, 2)
        print(f"Relative Error = {relative_error}%")

    print("Width Scales = ", args.width_scales)
    print("Raw Errors = ", raw_errors)
    print("Syn Errors = ", syn_errors)
    
    plt.plot(args.width_scales, raw_errors, 'bo-', label ='Raw Errors')
    plt.plot(args.width_scales, syn_errors, 'ro-', label ='Syn Errors')
    
    plt.legend(fontsize=12, loc="upper right")
    plt.xlabel("Width Scale", fontsize=16)
    plt.ylabel("Error %", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plot_name = "cms_line_{}_top_{}%.jpg".format(args.hash, args.percentile)
    plt.savefig(os.path.join(args.results, plot_name), bbox_inches="tight", dpi=300)


if __name__ == '__main__':
    main()
