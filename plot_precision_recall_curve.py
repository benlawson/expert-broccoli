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
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
algos = ["XGBoost", "Decision Tree"]

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

for model_filename in ["xgb.joblib", 'decisiontree.joblib']:
    model = joblib.load(model_filename)
    y_predict = model.predict(X_test)
    if type(y_predict[0]) == int or type(y_predict[0]) == np.int64:
        y_predict = label_binarize(y_predict, range(n_classes))
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
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')


# plot each algorithm
for i, (color, algo) in enumerate(zip(colors, algos)):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for {0} (area = {1:0.2f})'
              ''.format(algo, average_precision[i]))


fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curves for some approaches')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))


plt.savefig('algos.svg')
