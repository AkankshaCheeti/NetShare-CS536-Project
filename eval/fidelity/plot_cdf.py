#plot cdf
import sys, configparser, json, random, copy, math, os, pickle
random.seed(42)

import argparse
import tensorflow as tf
import numpy as np

# avoid type3 fonts
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams.update({'font.size': 15})
import matplotlib.pyplot as plt

# color-blindness friendly
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
# colors = {
#     'blue':   [55,  126, 184],  #377eb8 
#     'orange': [255, 127, 0],    #ff7f00
#     'green':  [77,  175, 74],   #4daf4a
#     'pink':   [247, 129, 191],  #f781bf
#     'brown':  [166, 86,  40],   #a65628
#     'purple': [152, 78,  163],  #984ea3
#     'gray':   [153, 153, 153],  #999999
#     'red':    [228, 26,  28],   #e41a1c
#     'yellow': [222, 222, 0]     #dede00
# }  

# https://www.iana.org/assignments/protocol-numbers/protocol-numbers.xhtml
dict_pr_str2int = {
    "ESP": 50,
    "GRE": 47,
    "ICMP": 1,
    "IPIP": 4,
    "IPv6": 41,
    "TCP": 6,
    "UDP": 17,
    "RSVP": 46,
    "Other": 255,
    "255": 255, # TEMP
}

import pandas as pd
import statsmodels.api as sm

from tqdm import tqdm
from gensim.models import Word2Vec
from collections import Counter, OrderedDict
from scipy.stats import wasserstein_distance
from scipy.spatial import distance

from field import *
from output import *
from embedding_helper import build_annoy_dictionary_word2vec, get_original_obj


N_TOPK_SERVICE_PORTS = 5

def vals2cdf(vals):
    dist_dict = dict(Counter(vals))
    dist_dict = {k: v for k, v in sorted(dist_dict.items(), key = lambda x: x[0])}
    x = dist_dict.keys()

    pdf = np.asarray(list(dist_dict.values()), dtype=float) / float(sum(dist_dict.values()))
    cdf = np.cumsum(pdf)

    return x, cdf

# syn_df_dict: {name: dict}
def plot_cdf(raw_df, syn_df_dict, xlabel, ylabel, plot_loc, metric, x_logscale=False, y_logscale=False):
    print(metric)
    plt.clf()

    if metric == "flow_size":
        x, cdf = vals2cdf(raw_df.groupby(["srcip", "dstip", "srcport", "dstport", "proto"]).size().values)
    else:
        x, cdf = vals2cdf(raw_df[metric])

    plt.plot(x, cdf, label="Real", color=CB_color_cycle[0], linewidth=5)
    idx = 1
    for method, syn_df in syn_df_dict.items():
        if method == "CTGAN-B":
            label_method = "CTGAN"
        else:
            label_method = method
        
        if metric == "pkt" or metric == "byt":
            syn_df[metric] = np.round(syn_df[metric])

        if metric == "flow_size":
            x, cdf = vals2cdf(syn_df.groupby(["srcip", "dstip", "srcport", "dstport", "proto"]).size().values)
        else:
            x, cdf = vals2cdf(syn_df[metric])

        if method == "NetShare":
            plt.plot(x, cdf, label=label_method, color=CB_color_cycle[4], linewidth=3)
            
        else:
            plt.plot(x, cdf, label=label_method, color=CB_color_cycle[idx], linewidth=1.5)
            
        idx += 1
    
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if x_logscale:
        plt.xscale('log')
    if y_logscale:
        plt.yscale('log')

    plt.savefig(plot_loc, bbox_inches="tight", dpi=300)


def run_ugr16_flowsize_pkt_byt(args):
    raw_df = pd.read_csv(os.path.join(args.dataset, "raw.csv"))
    os.makedirs(args.results, exist_ok=True)

    syn_df_dict = {}
    syn_df = pd.read_csv(os.path.join(args.dataset, "syn.csv"))
    syn_df_dict["NetShare"] = syn_df

    for metric, xlabel in {
            "flow_size": "# of records with the same five tuple",
            "pkt": "# of packets per flow",
            "byt": "# of bytes per flow",
            # "td": "Flow duration (seconds)",
        }.items():
        plot_cdf(
            raw_df=raw_df,
            syn_df_dict=syn_df_dict,
            xlabel=xlabel,
            ylabel="CDF",
            plot_loc=os.path.join(
                args.results, 
                "cdf_ugr16_{}.jpg".format(metric)),
            metric=metric,
            x_logscale=(metric != "td")
        )


def run_caida_flowsize(args):
    print("Dataset:", args.dataset)
    os.makedirs(args.results, exist_ok=True)
    
    raw_df = pd.read_csv(os.path.join(args.dataset, "raw.csv"))
    syn_df_dict = {}
    syn_df = pd.read_csv(os.path.join(args.dataset, "syn.csv"))
    syn_df_dict["NetShare"] = syn_df
    
    for metric, xlabel in {
            "flow_size": "Flow size (# of packets perflow)"
        }.items():
        plot_cdf(
            raw_df=raw_df,
            syn_df_dict=syn_df_dict,
            xlabel=xlabel,
            ylabel="CDF",
            plot_loc=os.path.join(
                args.results, 
                "cdf_caida_{}.jpg".format(metric)),
            metric=metric,
            x_logscale=(metric != "td")
        )

def main():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--type", type=str)
    CLI.add_argument("--dataset", type=str)
    CLI.add_argument("--results", type=str)
    # convert incoming args to a dictionary
    args = CLI.parse_args()
    plot_methods = {
        "NETFLOW":  run_ugr16_flowsize_pkt_byt,
        "PCAP":     run_caida_flowsize
    }
    # pick the right method and call arguments
    plot_methods[args.type](args)


if __name__ == '__main__':
    main()    