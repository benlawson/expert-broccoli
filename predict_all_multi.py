import matplotlib
matplotlib.use("Agg")

import os

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier

from xgboost import XGBClassifier
from keras.models import load_model

from sklearn.multiclass import OneVsRestClassifier

import joblib
from joblib import Parallel, delayed

from predict_help import calculate_p_r, plot_curve, filter_labels

WITHOUT_MOVIE = True
if WITHOUT_MOVIE:
    prefix = "no_movie"
else:
    prefix = ""

filenames = ["dummy", "nearestneighbors", "svm", "decisiontree", "neuralnet", "naivebayes", "lda", "xgb"]

classifiers = [
    DummyClassifier(),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, probability=True),
    DecisionTreeClassifier(max_depth=5),
    # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    # AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    XGBClassifier()
    ]

# note different from single label
def train_save(model, filename, representation_name):
    folder = os.path.join(prefix, representation_name, "multiclass")
    os.makedirs(folder, exist_ok=True)
    try:
        model.fit(X_train, y_train)
    except ValueError:
        model = OneVsRestClassifier(model)
        model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(folder, "{}.joblib".format(filename)))
    print("{0}: {1} acc".format(filename, model.score(X_test, y_test)))


one_hot, labels, _ = joblib.load(os.path.join(prefix, "labels_multiclass.joblib"))
y = np.argmax(one_hot, axis=1)
y_array = np.array(one_hot)

if WITHOUT_MOVIE:
    representation_files = ['no_movie/pca_representation.joblib', "no_movie/inception_representations.joblib", "no_movie/resnet_representations.joblib"]
else:
    representation_files = ['pca_representation.joblib', "inception_representations.joblib", "resnet_representations.joblib"]


# first is pca, then inception, then resnet
looper = list(zip(representation_files, ['pca', 'inception', 'resnet']))[:-1]

# for representation_filename, representation_name in looper:

#     # get the training data
#     # all classifiers must use vector output
#     X = joblib.load(representation_filename)
#     X, y = filter_labels(X, y_array)
#     print(X.shape)
#     print(y.shape)

#     X_train, X_test, y_train, y_test =  train_test_split(X, y, random_state=42)

#     # train all the classifiers
#     Parallel(n_jobs=-1)(delayed(train_save)(m, f,representation_name) for (m,f) in zip(classifiers, filenames))

for representation_filename, representation_name in looper:

    algos = ["Dummy", "Nearest Neighbors", "SVM", "Decision Tree",  "Single Layer Perceptron", "Naive Bayes", "LDA", "XGBoost",]

    # get the training data
    X = joblib.load(representation_filename)
    X, y = filter_labels(X, y_array)
    X_train, X_test, y_train, y_test =  train_test_split(X, y, random_state=42)

    if representation_name == 'inception':
        X_img =joblib.load(os.path.join(prefix, "inception_preprocessed.joblib"))
        X_img, y = filter_labels(X_img, y_array)
        X_train_img, X_test_img, y_train, y_test = train_test_split(X_img, y, random_state=42)

    if representation_name == 'resnet':
        X_img =joblib.load(os.path.join(prefix, "resnet_preprocessed.joblib"))
        X_img, y = filter_labels(X_img, y_array)
        X_train_img, X_test_img, y_train, y_test = train_test_split(X_img, y, random_state=42)

    # evaluate the classifiers
    precision, recall, average_precision = [], [], []

    for model_filename in filenames:
        model = joblib.load(os.path.join(prefix, representation_name,"multiclass", "{0}.joblib".format(model_filename)))
        try:
            y_predict = model.predict_proba(X_test)
        except AttributeError:
            y_predict = model.predict(X_test)
        try:
            p, r, ap  = calculate_p_r(y_test, y_predict)
        except:
            # some classifiers have weird output shapes
            y_predict = np.array(y_predict)[:,:,0].T
            p, r, ap  = calculate_p_r(y_test, y_predict)

        precision.append(p)
        recall.append(r)
        average_precision.append(ap)

    # extra loop for fine-tuned stuff
    if representation_name == 'inception':
        model = load_model(os.path.join(prefix, "inception_multi_model-final.hdf5"))
        y_predict = model.predict(np.array(X_test_img))
        p, r, ap  = calculate_p_r(y_test, y_predict)
        precision.append(p)
        recall.append(r)
        average_precision.append(ap)
        algos += [ "Fine-tuned InceptionNet"]

    if representation_name == 'resnet':

        model = load_model(os.path.join(prefix, "resnet_multi_model.hdf5"))
        y_predict = model.predict(np.array(X_test_img))
        p, r, ap  = calculate_p_r(y_test, y_predict)
        precision.append(p)
        recall.append(r)
        average_precision.append(ap)
        algos += [ "Fine-tuned ResNet"]

    print('plotting curve for {}'.format(representation_name))
    joblib.dump((precision, recall, average_precision, representation_name+"multiclass", algos), ( os.path.join(prefix, representation_name,"multiclass", "predict_all_multi_stuff.joblib")))

    plot_curve(precision, recall, average_precision, representation_name+"multiclass", algos, prefix)

