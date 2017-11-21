#standard libraries
import os
import glob
from collections import defaultdict

#external libraries
import numpy as np
import joblib
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img, img_to_array

import theano
theano.config.openmp = True

model = InceptionV3(include_top=False, pooling='avg')

cat_dict = defaultdict(list)

 # filenames = sorted(glob.glob("/backupdrive/datasets/BUSampleDataSet/*.jpg"))
for folder in sorted(glob.glob("/ssdrive/WinEarthPhotosByKeyword/*")):
    print(folder)
    filenames = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    for filename in filenames:
        image = img_to_array(load_img(filename, target_size=(299,299)))
        cat_dict[folder].append(image)

#print("loaded all images")
#joblib.dump(cat_dict, "images.joblib")
cat_dict = joblib.load("images.joblib")
# images = joblib.load("images.joblib")


reps = []
for cat in sorted(cat_dict.keys()):
    print("representing {}".format(cat))
    representations = model.predict(np.array(cat_dict[cat]))
    reps.extend(representations)

joblib.dump(reps, "representations.joblib")

