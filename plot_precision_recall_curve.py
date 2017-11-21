import matplotlib
matplotlib.use("Agg")


# extensive re-use of code from the following resource:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import numpy as np
import joblib
# from sklearn.preprocessing import label_binarize
import pandas as pd
from keras.models import load_model

from itertools import cycle
# setup plot details
# colors = cycle(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan'])
colors = cycle(['b', 'g', 'r', 'c', 'm', 'y',])


algos = ["Nearest Neighbors", "SVM", "Decision Tree",  "Single Layer Perceptron", "Naive Bayes", "LDA", "XGBoost", "Fine-tuned InceptionNet"]

filenames = ["nearestneighbors", "svm", "decisiontree", "neuralnet",  "naivebayes", "lda", "xgb", 'inceptionnet_lol_no_joblib']

n_classes = 11
X_train, X_test, y_train, y_test = joblib.load("testdata.joblib")
_, X_images, _, y_images = joblib.load("./imgs_testdata.joblib")
print(y_images == y_test)

# load classifier
# For each class
def calculate_p_r(y_test, y_predict):
    # A "micro-average": quantifying score on all classes jointly
    print(y_test.shape)
    print(y_predict.shape)
    precision, recall, _ = precision_recall_curve(y_test.ravel(), y_predict.ravel())
    average_precision = average_precision_score(y_test, y_predict, average="micro")
    return precision, recall, average_precision

precision, recall, average_precision = [], [], []

for model_filename in filenames:
    try:
        model = joblib.load("{0}.joblib".format(model_filename))
        y_predict = model.predict_proba(X_test)
    except FileNotFoundError:
        model = load_model('/home/jlam17/Documents/Class/fall_2017/cs542/final_project/finetuned_model_epoch-final.hdf5')
        y_predict = model.predict(np.array(X_images).reshape(-1, 299, 299, 3))
    except AttributeError:
        y_predict = model.predict(X_test)

    print(model_filename)
    if type(y_predict[0]) == np.str_ or type(y_predict[0]) == np.int64:
        # y_predict = label_binarize(y_predict, range(n_classes))
        y_predict = np.array(pd.get_dummies(y_predict).as_matrix())
    if type(y_test[0]) == np.str_ or type(y_test[0]) == np.int64:
        # y_test= label_binarize(y_test, range(n_classes))
        y_test = np.array(pd.get_dummies(y_test).as_matrix())
    p, r, ap  = calculate_p_r(y_test, y_predict)
    precision.append(p)
    recall.append(r)
    average_precision.append(ap)



# plot figure
# set up nice f1 curves
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


fig = plt.gcf()
#fig.subplots_adjust(right=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title("Precision-Recall curves for the fine-tuned InceptionNet \nand some other approaches based on (non-fine-tuned) InceptionNet representations")
lgd = plt.legend(lines, labels, loc='upper left', prop=dict(size=14), bbox_to_anchor=(1.02, 1))

plt.savefig('algos.png', bbox_extra_artists=(lgd,), bbox_inches="tight")
