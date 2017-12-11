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
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier

from xgboost import XGBClassifier
from keras.models import load_model

import joblib
# from joblib import Parallel, delayed

from predict_help import calculate_p_r, Stopwatch, plot_curve

WITHOUT_MOVIE = True
if WITHOUT_MOVIE:
    prefix = "without_movie"
else:
    prefix = "with_movie"

filenames = ["dummy", "nearestneighbors", "svm", "decisiontree",  "neuralnet",  "naivebayes", "lda", "xgb"]


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

one_hot, labels, _ = joblib.load(os.path.join(prefix, "labels.joblib"))
y = np.argmax(one_hot, axis=1)
y_array = np.array(one_hot)

representation_files = ['{0}/pca_representation.joblib'.format(prefix), "{0}/inception_representations.joblib".format(prefix), "{0}/resnet_representations.joblib".format(prefix)]

# set what representations to run over
looper = list(zip(representation_files, ['pca', 'inception', 'resnet']))[:-1]

# for representation_filename, representation_name in looper:

#     # get the training data
#     X = np.array(joblib.load(representation_filename))
#     X_train, X_test, y_train, y_test =  train_test_split(X, y, random_state=42)
#     X_train, X_test, y_train_array, y_test_array =  train_test_split(X, y_array, random_state=42)


#     # train all the classifiers
#     with Stopwatch() as sw:
#         Parallel(n_jobs=-1)(delayed(train_save)(m, f,X_train, y_train, y_train_array, representation_name, prefix) for (m,f) in zip(classifiers,filenames))

for representation_filename, representation_name in looper:

    # get the training data

    X = joblib.load(representation_filename)

    X_train, X_test, y_train, y_test =  train_test_split(X, y, random_state=42)
    X_train, X_test, y_train_array, y_test_array =  train_test_split(X, y_array, random_state=42)

    algos = ["Dummy", "Nearest Neighbors", "SVM", "Decision Tree", "Single Layer Perceptron", "Naive Bayes", "LDA", "XGBoost",]# "Fine-tuned InceptionNet"]

    if representation_name == 'inception':
        X_img =joblib.load(os.path.join(prefix, "inception_preprocessed.joblib"))
        X_train_img, X_test_img, y_train, y_test = train_test_split(X_img, y, random_state=42)

    if representation_name == 'resnet':
        X_img =joblib.load(os.path.join(prefix, "resnet_preprocessed.joblib"))
        X_train_img, X_test_img, y_train, y_test = train_test_split(X_img, y, random_state=42)

    # evaluate the classifiers
    precision, recall, average_precision = [], [], []
    for model_filename in filenames:
        try:
            model = joblib.load(os.path.join(prefix, representation_name, "{0}.joblib".format(model_filename)))
            y_predict = model.predict_proba(X_test)
        except AttributeError:
            y_predict = model.predict(X_test)
        try:
            p, r, ap  = calculate_p_r(y_test, y_predict)
        except ValueError:
            p, r, ap  = calculate_p_r(y_test_array, y_predict)

        precision.append(p)
        recall.append(r)
        average_precision.append(ap)

    # extra loop for fine-tuned stuff
    if representation_name == 'inception':
        model = load_model(os.path.join(prefix, "inception_single_model-final.hdf5"))
        with Stopwatch() as sw:
            y_predict = model.predict(np.array(X_test_img))
        p, r, ap  = calculate_p_r(y_test_array, y_predict)
        precision.append(p)
        recall.append(r)
        average_precision.append(ap)
        algos += [ "Fine-tuned InceptionNet"]

    if representation_name == 'resnet':
        model = load_model(os.path.join(prefix, "resnet_single_model-final.hdf5"))
        with Stopwatch() as sw:
            y_predict = model.predict(np.array(X_test_img))
        p, r, ap  = calculate_p_r(y_test_array, y_predict)
        precision.append(p)
        recall.append(r)
        average_precision.append(ap)
        algos += [ "Fine-tuned ResNet"]

        joblib.dump((precision, recall, average_precision, representation_name, algos), os.path.join(prefix, representation_name, "predict_all_stuff.joblib"))
    # precision, recall, average_precision, representation_name, algos = joblib.load( os.path.join(representation_name, "predict_all_stuff.joblib"))
    plot_curve(precision, recall, average_precision, representation_name, algos, prefix)

