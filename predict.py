#standard libraries
import glob

#external libraries
import joblib
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


tags = pd.read_csv("/backupdrive/datasets/tags.csv", index_col=0, header=None).T.to_dict()
filenames = sorted(glob.glob("/backupdrive/datasets/BUSampleDataSet/*.jpg"))
filenames = [filename.split("/")[-1] for filename in filenames]

ten_tags = ["day", "night", "aurora", "sunrise sunset", "moon", "inside iss", "iss structure",  "stars", "dock undock", "spacewalk", "deployed satellite", "cupola"]

y = []
for filename in filenames:
    try:
        file_tags = set([str(t).lower().strip() for t in tags[filename].values()])
        y.append(list(file_tags.intersection(ten_tags)))
    except KeyError:
        continue
N = -1
X =  joblib.load("representations.joblib")[:N]
encoder =  MultiLabelBinarizer()
y = encoder.fit_transform(y)
#y = y.toarray()
y = np.nan_to_num(y)[:N]

X_train, X_test, y_train, y_test =  train_test_split(X, y, random_state=42)

print(X_train.shape)
print(y_train.shape)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
joblib.dump(model, "decisiontree.joblib")
joblib.dump((X_train, X_test, y_train, y_test), "testdata.joblib")


