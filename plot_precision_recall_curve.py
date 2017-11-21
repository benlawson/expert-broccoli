import matplotlib
matplotlib.use("Agg")


# extensive re-use of code from the following resource:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import numpy as np
import joblib
from sklearn.preprocessing import label_binarize

from itertools import cycle
# setup plot details
colors = cycle(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan'])

algos = ["Nearest Neighbors", "SVM", "Decision Tree", "Random Forest", "Single Layer Perceptron", "AdaBoost", "Naive Bayes", "LDA", "XGBoost"]

filenames = ["nearestneighbors", "svm", "decisiontree", "randomforest", "neuralnet", "adaboost", "naivebayes", "lda", "xgb"]

n_classes = 11
X_train, X_test, y_train, y_test = joblib.load("testdata.joblib")


# load classifier
# For each class
def calculate_p_r(y_test, y_predict):
    # A "micro-average": quantifying score on all classes jointly
    precision, recall, _ = precision_recall_curve(y_test.ravel(), y_predict.ravel())
    average_precision = average_precision_score(y_test, y_predict, average="micro")
    return precision, recall, average_precision

precision, recall, average_precision = [], [], []

for model_filename in filenames:
    model = joblib.load("{0}.joblib".format(model_filename))
    y_predict = model.predict(X_test)
    if type(y_predict[0]) == np.str_ or type(y_predict[0]) == np.int64:
        y_predict = label_binarize(y_predict, range(n_classes))
    if type(y_test[0]) == np.str_ or type(y_test[0]) == np.int64:
        y_test= label_binarize(y_test, range(n_classes))
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
plt.title("Precision-Recall curves for some approaches based on PCA'd raw image data  representations")
lgd = plt.legend(lines, labels, loc='upper left', prop=dict(size=14), bbox_to_anchor=(1.02, 1))

plt.savefig('algos_pca.png', bbox_extra_artists=(lgd,), bbox_inches="tight")
