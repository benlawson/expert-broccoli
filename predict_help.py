
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_palette(sns.color_palette("hls", 9))
colors = sns.color_palette()

import time
import os

import joblib
# from joblib import Parallel, delayed

import numpy as np

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


class Stopwatch(object):
    def __enter__(self):
        self.start = time.time()
    def __exit__(self, exception_type, exception_value, traceback):
        print(time.time() - self.start)

def calculate_p_r(y_test, y_predict):
    # A "micro-average": quantifying score on all classes jointly
    precision, recall, _ = precision_recall_curve(y_test.ravel(), y_predict.ravel())
    average_precision = average_precision_score(y_test, y_predict, average="micro")
    return precision, recall, average_precision

def filter_labels(X, y, threshold=0.01):
    print('{} images w/ {} possible labels'.format(y.shape[0], y.shape[1]))
    image_count = len(y)
    label_counts = y.sum(axis=0)
    label_percent = np.divide(label_counts, image_count)
    remove_labels = [idx for idx, val in enumerate(label_percent) if val < threshold]
    print('discarding {} labels; {} labels remaining'.format(len(remove_labels), y.shape[1] - len(remove_labels)))
    y = np.delete(y, remove_labels, axis=1)

    labels_per_image = y.sum(axis=1)
    remove_imgs = [idx for idx, val in enumerate(labels_per_image) if val == 0 ]
    if len(remove_imgs) > 0:
        y = np.delete(y, remove_imgs, axis=0)
        X = np.delete(X, remove_imgs, axis=0)
        print('removing {} images w/ 0 labels after discarding {} labels'.format(len(remove_imgs), len(remove_labels)))
    return np.array(X), np.array(y)

def train_save(model, filename, X_train, y_train, y_train_array, representation_name, prefix):
    if not(os.path.exists(representation_name)):
        os.makedirs(representation_name)
    try:
        model.fit(X_train, y_train)
    except ValueError:
        model.fit(X_train, y_train_array)
    joblib.dump(model, os.path.join(prefix, representation_name, "{}.joblib".format(filename)))
    # try:
    #     print("{0}: {1} acc".format(filename, model.score(X_test, y_test)))
    # except ValueError:
    #     print("{0}: {1} acc".format(filename, model.score(X_test, y_test_array)))


def plot_curve(precision, recall, average_precision, representation_name, algos, prefix):
    print(len(precision), len(recall), len(average_precision), len(representation_name), len(algos), prefix)

    ## plotting code (once per representation)
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

    plt.savefig(os.path.join(prefix, '{0}algos.png'.format(representation_name)), bbox_extra_artists=(lgd,), bbox_inches="tight")

