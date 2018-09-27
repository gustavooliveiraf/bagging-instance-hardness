import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import itertools
import seaborn as sns
from collections import Counter
from scipy.stats import  wilcoxon

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE 

from sklearn import tree
from sklearn import linear_model

from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score

from sklearn.metrics import cohen_kappa_score

from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

class Main:
    def __init__(self, x, y):
        self.data = data
        self.y = np.array(y)
        self.x = np.array(x)
        self.kNeighbors = 5
        self.threshold = 0.5
        self.kdnGreater = [] # ThanThreshold
        self.kdnLess = [] # ThanThreshold
        self.pool_size = 100

    def kappa_pruning(self, k_fold, n_times, pool_size, m):
        comb = combinations(range(pool_size), 2)
        pruning = []
        pruningKdnGreater = []
        pruningKdnLess = []
        for i in range(n_times):
            skf = StratifiedKFold(n_splits=k_fold,shuffle=True)

            for train_index, test_index in skf.split(self.x, self.y):
                X_train, X_test = self.x[train_index], self.x[test_index]
                Y_train, Y_test = self.y[train_index], self.y[test_index]

                x_train, y_train = SMOTE().fit_sample(X_train, Y_train)
                # x_test, y_test = SMOTE().fit_sample(X_test, Y_test)

                kdnGreater, kdnLess = self.k_Disagreeing_neighbors_kDN(x_train, y_train)

                X_validationGreater, X_validationLess = self.x[kdnGreater], self.x[kdnLess]
                Y_validationGreater, Y_validationLess = self.y[kdnGreater], self.y[kdnLess]

                BagPercep = BaggingClassifier(linear_model.Perceptron(max_iter=5), pool_size)
                BagPercep.fit(x_train, y_train)
                for tupla in comb:
                    kappa = cohen_kappa_score(BagPercep.estimators_[tupla[0]].predict(x_train), BagPercep.estimators_[tupla[1]].predict(x_train))
                    pruning.append(tupla + (kappa,))

                    kappa = cohen_kappa_score(BagPercep.estimators_[tupla[0]].predict(X_validationGreater), BagPercep.estimators_[tupla[1]].predict(X_validationGreater))
                    pruningKdnGreater.append(tupla + (kappa,))

                    kappa = cohen_kappa_score(BagPercep.estimators_[tupla[0]].predict(X_validationLess), BagPercep.estimators_[tupla[1]].predict(X_validationLess))
                    pruningKdnLess.append(tupla + (kappa,))
                break
        
        pruning.sort(key=lambda tup: tup[2])

        return (pruning[:m], pruningKdnGreater[:m], pruningKdnLess[:m])

    def k_Disagreeing_neighbors_kDN(self, x_train, y_train):
        neigh = KNeighborsClassifier(n_neighbors=self.kNeighbors)
        neigh.fit(self.x, self.y)

        kdnGreater = []
        kdnLess = []
        for i in range(len(y_train)):
            kdn = sum(self.y[neigh.kneighbors([x_train[i]], return_distance=False)[0]]!=y_train[i])/self.kNeighbors
            if (kdn > self.threshold):
                kdnGreater.append(i)
            else:
                kdnLess.append(i)

        return (kdnGreater, kdnLess)

# =============================================================================================================================

    def sort_score(self, pool, x_test, y_test):
        temp = pool.estimators_[:]

        score_tuple = []
        for i, percep_i in enumerate(temp):
            score_tuple.append((i, percep_i.score(x_test, y_test)))

        score_tuple.sort(key=lambda tup: tup[1], reverse=True)
        score = []
        for i in score_tuple:
            score.append(i[0])

        return score

    def reduce_error_pre(self, k_fold, n_times):
        score_pruning = 0
        score_pool = 0
        for i in range(n_times):
            skf = StratifiedKFold(n_splits=k_fold,shuffle=True)

            for train_index, test_index in skf.split(self.x, self.y):
                X_train, X_test = self.x[train_index], self.x[test_index]
                Y_train, Y_test = self.y[train_index], self.y[test_index]

                x_train, y_train = SMOTE().fit_sample(X_train, Y_train)
                # x_train, y_train = X_train, Y_train
                # x_test, y_test = SMOTE().fit_sample(X_test, Y_test)

                # kdnGreater, kdnLess = self.k_Disagreeing_neighbors_kDN(x_train, y_train)

                # X_validationGreater, X_validationLess = x_train[kdnGreater], x_train[kdnLess]
                # Y_validationGreater, Y_validationLess = y_train[kdnGreater], y_train[kdnLess]

                BagPercep = BaggingClassifier(linear_model.Perceptron(max_iter=5), self.pool_size)
                BagPercep.fit(x_train, y_train)

                score_pool += BagPercep.score(X_test, Y_test)
                # score_pool += score_pool_temp

                score = self.sort_score(BagPercep, x_train, y_train)
                score_pruning += self.reduce_error(BagPercep, score, x_train, y_train, x_train, y_train)
                # score_pruning += score_pruning_temp

                print(score_pool, score_pruning)

        return (score_pool/k_fold, score_pruning/k_fold)

    def reduce_error(self, pool, score_index, x_train, y_train, x_validation, y_validation):
        BagPercepCurrent = BaggingClassifier(linear_model.Perceptron(max_iter=5), self.pool_size)
        BagPercepCurrent.fit(x_train, y_train)

        ensemble_index = set()
        ensemble_index.add(score_index[0])

        ensemble = []
        ensemble.append(pool.estimators_[score_index[0]])
        while (True):
            index_best_score = 0
            BagPercepCurrent.estimators_ = ensemble
            best_score = BagPercepCurrent.score(x_validation, y_validation)

            for i in list(score_index):
                if i not in ensemble_index:
                    BagPercepCurrent.estimators_ = ensemble + [pool.estimators_[i]]
                    score_current = BagPercepCurrent.score(x_validation, y_validation)

                    if best_score < score_current:
                        index_best_score = i
            if index_best_score != 0:
                ensemble_index.add(index_best_score)
                ensemble.append(pool.estimators_[index_best_score])
            else:
                return best_score
            if len(ensemble_index) == self.pool_size:
                return best_score
                break

def test_kappa(modelo):
    # modelo.kNeighborsClassifier()
    best_classifiers = modelo.kappa_pruning(10, 1, 10, 5) #retorna os 3 conjuntos de classificadores a b c
    ensemble = set()
    ensembleGreater = set()
    ensembleLess = set()
    for i in best_classifiers[0]:
        ensemble.add(i[0])
        ensemble.add(i[1])

    for i in best_classifiers[1]:
        ensembleGreater.add(i[0])
        ensembleGreater.add(i[1])

    for i in best_classifiers[2]:
        ensembleLess.add(i[0])
        ensembleLess.add(i[1])

    print(list(ensemble))

def test_reduce(modelo):
    print(modelo.reduce_error_pre(10, 1))

# test
data = pd.read_csv('./cm1.csv')
# data = pd.read_csv('./jm1.csv')
# data = pd.read_csv('./kc2.csv')

enc = LabelEncoder()
data.CLASS = enc.fit_transform(data.CLASS)

y = np.array(data["CLASS"])
x = np.array(data.drop(axis=1, columns = ["CLASS"]))

scaler = StandardScaler()
x = scaler.fit_transform(x)

modelo = Main(x, y)

test_reduce(modelo)