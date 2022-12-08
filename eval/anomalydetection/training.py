import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC, OneClassSVM
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier, IsolationForest
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, plot_confusion_matrix, roc_auc_score
from scipy.stats import spearmanr


def train_models(X_train_s, Y_train_s, X_test_s, Y_test_s):
    # ML models for cl
    models = [LogisticRegression, 
            XGBClassifier, 
            LGBMClassifier, 
            GaussianNB, 
            DecisionTreeClassifier, 
            RandomForestClassifier, 
            AdaBoostClassifier, 
            MLPClassifier]
    models_names = ['Logistic Regression', 
                    'XGBClassifier', 
                    'LGBMClassifier', 
                    'Gaussian Naive Bayes',
                    'Decision Tree', 
                    'Random Forest', 
                    'AdaBoost', 
                    'MLPClassifier']
    models_type = ['ML' for x in models]
    models_params = [{} for x in models]

    # Sequential training for ML models
    num_models = len(models)
    assert num_models == len(models_type)
    assert num_models == len(models_params)

    dict_accs_tr, dict_accs_te = {}, {}
    for i in range(num_models):
        print('Training ' + models_names[i])
        
        clf = models[i](**models_params[i])
        clf.fit(X_train_s, Y_train_s)
        Y_tr_pred_s = clf.predict(X_train_s)
        Y_te_pred_s = clf.predict(X_test_s)

        dict_accs_tr[models_names[i]] = round(accuracy_score(Y_tr_pred_s, Y_train_s), 2)
        dict_accs_te[models_names[i]] = round(accuracy_score(Y_te_pred_s, Y_test_s), 2)
        print(dict_accs_tr[models_names[i]])
        print(dict_accs_te[models_names[i]])

        print(confusion_matrix(Y_te_pred_s, Y_test_s))

    return dict_accs_tr, dict_accs_te