import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from collections import defaultdict
from scipy.stats import spearmanr
from sklearn.utils import shuffle
from tqdm import tqdm
import argparse
import os

from encoding import encode_categorical
from training import train_models

QUANTIZATION = 6
BOTNET_KEYS = ['srcip', 'dstip']
MAX_BIN = int(1500 / 2**QUANTIZATION)

BENIGN_CLASS = 0
MALICIOUS_CLASS = 1

TRAIN_TEST_SPLIT = 0.2

def read_flows(dataset):
    # read raw data (correct the path)
    data_df = pd.read_csv(os.path.join(dataset))
    # group the data by the keys
    print("Grouping the Data Frame by keys")
    grouped_data_df = data_df.groupby(BOTNET_KEYS)
    
    print("Analyzing flows")
    flows = []
    for _tuple, series in tqdm(grouped_data_df[BOTNET_KEYS]):
        flow_histogram = {k:0 for k in range(MAX_BIN)}
        for pkt_len in series['pkt_len']:
            quant_pkt_len = int(pkt_len) >> QUANTIZATION
            if quant_pkt_len < MAX_BIN:
                flow_histogram[quant_pkt_len] += 1
        # convert histogram dictionary to list
        flow = [flow_histogram[x] for x in range(MAX_BIN)]
        flows.append(flow)
    flows_array = np.array(flows)
    return flows_array


def process_botnet_dataset(benign_dataset, malicious_dataset):
    benign_flows_array = read_flows(dataset=benign_dataset)
    malicious_flows_array = read_flows(dataset=malicious_dataset)
    
    benign_labels = np.zeros(shape=(len(benign_flows_array), 1))
    benign_dataset = np.append(benign_flows_array, benign_labels, axis=1)
    
    malicious_labels = np.ones(shape=(len(malicious_flows_array), 1))
    malicious_dataset = np.append(malicious_flows_array, malicious_labels, axis=1)
    
    dataset = np.append(benign_dataset, malicious_dataset, axis=0)
    
    shuffled_dataset = shuffle(dataset, random_state=0)
    
    train_set, test_set = train_test_split(shuffled_dataset, test_size=TRAIN_TEST_SPLIT)
    train_X, train_Y = train_set[:,:MAX_BIN], np.squeeze(train_set[:,MAX_BIN:], axis=1)
    test_X, test_Y = test_set[:,:MAX_BIN], np.squeeze(test_set[:,MAX_BIN:], axis=1)
    
    print(f"X_train Size = {train_X.shape}, X_test Size = {test_X.shape}, Y_train Size = {train_Y.shape}, Y_test Size = {test_Y.shape}")
    return (train_X, train_Y, test_X, test_Y)


def main():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--dataset", type=str)
    CLI.add_argument("--results", type=str)
    CLI.add_argument("--runs", type=int)
    args = CLI.parse_args()
    print(vars(args))

    (RAW_X_train_s, RAW_Y_train_s, RAW_X_test_s, RAW_Y_test_s) = process_botnet_dataset(benign_dataset=os.path.join(args.dataset, 'benign-raw.csv'),
                                                                                        malicious_dataset=os.path.join(args.dataset, 'malicious-raw.csv'))
    (SYN_X_train_s, SYN_Y_train_s, SYN_X_test_s, SYN_Y_test_s) = process_botnet_dataset(benign_dataset=os.path.join(args.dataset, 'benign-raw.csv'),
                                                                                        malicious_dataset=os.path.join(args.dataset, 'malicious-syn.csv'))
    
    correlations = []
    max_correlation = 0.0
    best_accs_raw_test, best_accs_syn_test = {}, {}
    for i in range(1, args.runs+1):
        # train models on both datasets
        (accs_raw_train, accs_raw_test) = train_models(RAW_X_train_s, RAW_Y_train_s, RAW_X_test_s, RAW_Y_test_s)
        (accs_syn_train, accs_syn_test) = train_models(SYN_X_train_s, SYN_Y_train_s, SYN_X_test_s, SYN_Y_test_s)
        # compare raw and synthetic accuracies
        print(f"Model Accuracies on Raw Dataset = {accs_raw_test}")
        print(f"Model Accuracies on Syn Dataset = {accs_syn_test}")
        (spearman_correlation, pvalue) = spearmanr(list(accs_raw_test.values()), list(accs_syn_test.values()))
        print(f"[Run {i}] Spearman Correlation of Raw and Synthetic traces = {round(spearman_correlation, 2)} (pvalue = {pvalue})")
        correlations.append(spearman_correlation)
        # update the best performing model accuracies
        if spearman_correlation > max_correlation:
            max_correlation = spearman_correlation
            best_accs_raw_test = accs_raw_test
            best_accs_syn_test = accs_syn_test

    print(f"\n\nSpearman Correlations over {args.runs} runs = {correlations}\n")
    average_spearman_correlation = np.mean(correlations)
    print(f"\nAverage Spearman Correlation of Raw and Synthetic traces over {args.runs} runs = {round(average_spearman_correlation, 2)}\n")

    bar_width = 0.35
    X = np.arange(len(best_accs_raw_test))
    plt.figure(figsize=(6, 4))
    plt.bar(X-bar_width/2, best_accs_raw_test.values(), label="Raw Data", width=bar_width, color='lightcoral', align='center')
    plt.bar(X+bar_width/2, best_accs_syn_test.values(), label="Syn Data", width=bar_width, color='steelblue', align='center')
    plt.xticks(X, best_accs_raw_test.keys())
    
    plt.title(f"Spearman Correlation = {round(max_correlation, 2)}", fontsize=14, y=1.11)
    plt.legend(fontsize=12, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2)
    plt.xlabel("ML Models", fontsize=16)
    plt.ylabel("Test Accuracy", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)   
    plt.savefig(os.path.join(args.results, "anomaly_botnet_bar.jpg"), bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()