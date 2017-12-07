import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import cycle

import os

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from xgboost import XGBClassifier
# from keras.models import load_model

import joblib
from joblib import Parallel, delayed


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

names = ["Nearest Neighbors", "SVM", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost", "Naive Bayes", "LDA", "XGBoost"]
filenames = ["nearestneighbors", "svm", "decisiontree", "randomforest", "neuralnet", "adaboost", "naivebayes", "lda", "xgb"]
algos = ["Nearest Neighbors", "SVM", "Decision Tree",  "Single Layer Perceptron", "Naive Bayes", "LDA", "XGBoost",]# "Fine-tuned InceptionNet"]

inception_filenames = ["inception_epoch-02", "inception_epoch-05", "inception_epoch-08", "inception_epoch-final"]
colors = cycle(['b', 'g', 'r', 'c', 'm', 'y',])


# inception_model_path = '/path'
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, probability=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    XGBClassifier()
    ]

# inception_models = [
#     load_model(inception_model_path + 'finetuned_model_epoch-02.hdf5'),
#     load_model(inception_model_path + 'finetuned_model_epoch-05.hdf5'),
#     load_model(inception_model_path + 'finetuned_model_epoch-08.hdf5'),
#     load_model(inception_model_path + 'finetuned_model_epoch-final.hdf5')
#     ]

def calculate_p_r(y_test, y_predict):
    # A "micro-average": quantifying score on all classes jointly
    precision, recall, _ = precision_recall_curve(y_test.ravel(), y_predict.ravel())
    average_precision = average_precision_score(y_test, y_predict, average="micro")
    return precision, recall, average_precision

def train_save(model, filename, representation_name):
    if not(os.path.exists(representation_name)):
        os.makedirs(representation_name)
    try:
        model.fit(X_train, y_train)
    except ValueError:
        model.fit(X_train, y_train_array)
    joblib.dump(model, os.path.join(representation_name, "{}.joblib".format(filename)))
    try:
        print("{0}: {1} acc".format(filename, model.score(X_test, y_test)))
    except ValueError:
        print("{0}: {1} acc".format(filename, model.score(X_test, y_test_array)))


def plot_curve(precision, recall, average_precision, representation_name):

    ## plotting code (once per representation
    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('$f_1={0:0.1f}$'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('$iso-f_1 curves$')


    # plot each algorithm
    for i, (color, algo) in enumerate(zip(colors, algos)):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for {0} (area = {1:0.2f})'
                   ''.format(algo, average_precision[i]))


    # fig = plt.gcf()
    #fig.subplots_adjust(right=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.title("Precision-Recall curves for the fine-tuned InceptionNet \nand some other approaches based on (non-fine-tuned) InceptionNet representations")
    plt.title("Precision-Recall curves for some approaches based on {0} representations".format(representation_name))
    lgd = plt.legend(lines, labels, loc='upper left', prop=dict(size=14), bbox_to_anchor=(1.02, 1))

    plt.savefig('{0}algos.png'.format(representation_name), bbox_extra_artists=(lgd,), bbox_inches="tight")





one_hot, labels, _ = joblib.load("./labels.joblib")
y = np.argmax(one_hot, axis=1)
y_array = np.array(one_hot)
representation_files = ['pca_representation.joblib', "inception_representations.joblib", "resnet_representations.joblib"]

for representation_filename, representation_name in zip(representation_files, ['pca', 'inception', 'resnet']):

    # get the training data
    X = joblib.load(representation_filename)

    X_train, X_test, y_train, y_test =  train_test_split(X, y, random_state=42)
    X_train, X_test, y_train_array, y_test_array =  train_test_split(X, y_array, random_state=42)


    # train all the classifiers
    Parallel(n_jobs=-1)(delayed(train_save)(m, f,representation_name) for (m,f) in zip(classifiers, filenames))

for representation_filename, representation_name in zip(representation_files, ['pca', 'inception', 'resnet']):

    # get the training data
    X = joblib.load(representation_filename)

    X_train, X_test, y_train, y_test =  train_test_split(X, y, random_state=42)
    X_train, X_test, y_train_array, y_test_array =  train_test_split(X, y_array, random_state=42)

    # evaluate the classifiers
    precision, recall, average_precision = [], [], []
    for model_filename in filenames:
        try:
            model = joblib.load(os.path.join(representation_name, "{0}.joblib".format(model_filename)))
            y_predict = model.predict_proba(X_test)
        # except FileNotFoundError:
            # model = load_model('/home/jlam17/Documents/Class/fall_2017/cs542/final_project/finetuned_model_epoch-final.hdf5')
            # y_predict = model.predict(np.array(X_images).reshape(-1, 299, 299, 3))
        except AttributeError:
            y_predict = model.predict(X_test)
        try:
            p, r, ap  = calculate_p_r(y_test, y_predict)
        except ValueError:
            p, r, ap  = calculate_p_r(y_test_array, y_predict)
        precision.append(p)
        recall.append(r)
        average_precision.append(ap)
    plot_curve(precision, recall, average_precision, representation_name)

