import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scipy.stats import spearmanr
import argparse
import os

from encoding import encode_categorical
from training import train_models


def process_ugr16_dataset(dataset):
    df = pd.read_csv(dataset)
    # df.head()

    N1 = 2500
    testset_size = 0.2

    target = 'type'
    # cat_cols = ['srcip', 'dstip', 'srcport', 'dstport', 'proto']
    cat_cols = ['srcport', 'dstport', 'proto']
    cont_cols = ['ts', 'td', 'pkt', 'byt']

    scale_X = True
    enc_type = 'label'

    if N1 > df['type'].value_counts()['blacklist']:
        N1 = df['type'].value_counts()['blacklist']

    # keep ixs unchanged across raw, syn (imp), sim (not as imp)?
    ixs_0 = np.where(df['type'] == 'background')[0]
    ixs_1 = np.where(df['type'] == 'blacklist')[0]
    ixs_0 = np.random.choice(ixs_0, N1, replace=False)
    ixs_1 = np.random.choice(ixs_1, N1, replace=False)
    df_new = df.iloc[np.concatenate([ixs_0, ixs_1]), :].copy()

    df_train, df_test = train_test_split(df_new, test_size=testset_size)

    smalls_threshold = 0.001

    # everything not in large would be converted to smalls
    # computationally efficient and helps cater for unseen values
    large_dict = {}
    for field in cat_cols:
        large_dict_curr = {k:v for (k,v) in df_train[field].value_counts(normalize=True).items() if v>=smalls_threshold}
        df_train.loc[~df_train[field].isin(large_dict_curr.keys()), field] = 'smalls'
        large_dict[field] = large_dict_curr

    # smalls mapping on test data (can do this in loop above but would not be able to replicate it in production)
    for field in list(large_dict.keys()):
        df_test.loc[~df_test[field].isin(large_dict[field].keys()), field] = 'smalls'
        
    # scale continuous features and encode categorical features
    if scale_X:
        scaler_X_cont = StandardScaler()
        scaler_X_cont_emb = StandardScaler()
        X_train_cont_s = scaler_X_cont.fit_transform(df_train[cont_cols])
        X_test_cont_s = scaler_X_cont.transform(df_test[cont_cols])
    else:
        X_train_cont_s = df_train[cont_cols]
        X_test_cont_s = df_test[cont_cols]
        
    X_train_cat_s, enc_cat = encode_categorical(df_train[cat_cols], enc_type)
    X_train_cat_s = np.array(X_train_cat_s)
    X_test_cat_s = enc_cat.transform(df_test[cat_cols])
    X_test_cat_s = np.array(X_test_cat_s)

    X_train_s = np.concatenate((X_train_cont_s, X_train_cat_s), axis=1)
    X_test_s = np.concatenate((X_test_cont_s, X_test_cat_s), axis=1)

    Y_train = df_train[target].copy().values
    Y_test = df_test[target].copy().values

    Y_train_s = (Y_train == 'blacklist').astype('int')
    Y_test_s = (Y_test == 'blacklist').astype('int')

    print(f"X_train Size = {X_train_s.shape}, X_test Size = {X_test_s.shape}, Y_train Size = {Y_train_s.shape}, Y_test Size = {Y_test_s.shape}")

    # check data imbalance
    ratio_1_0_train = sum(Y_train_s == 1) / sum(Y_train_s == 0)
    ratio_1_0_test = sum(Y_test_s == 1) / sum(Y_test_s == 0)
    print(f"Train 1/0 Split = {ratio_1_0_train}; Test 1/0 Split = {ratio_1_0_test}")

    return (X_train_s, Y_train_s, X_test_s, Y_test_s)


def main():
    CLI = argparse.ArgumentParser()
    CLI.add_argument("--dataset", type=str)
    CLI.add_argument("--results", type=str)
    CLI.add_argument("--runs", type=int)
    args = CLI.parse_args()
    print(vars(args))
    
    (RAW_X_train_s, RAW_Y_train_s, RAW_X_test_s, RAW_Y_test_s) = process_ugr16_dataset(dataset=os.path.join(args.dataset, 'raw.csv'))
    (SYN_X_train_s, SYN_Y_train_s, SYN_X_test_s, SYN_Y_test_s) = process_ugr16_dataset(dataset=os.path.join(args.dataset, 'syn.csv'))
    
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
        # compare raw and synthetic accuracies
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
    plt.savefig(os.path.join(args.results, "anomaly_ugr16_bar.jpg"), bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()