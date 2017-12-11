import joblib
import numpy as np
from keras.models import load_model, Model


# do inception
images = np.array(joblib.load("inception_preprocessed.joblib"))
model = load_model("inception_single_model-final.hdf5")
model3 = Model(inputs=model.input, outputs=model.layers[-3].output)
inc_fine = model3.predict(images)
joblib.dump(inc_fine, "fine_tuned_inception_xxx.joblib")

# do resnet
images = np.array(joblib.load("resnet_preprocessed.joblib"))
model = load_model("resnet_single_model-final.hdf5")
model4 = Model(inputs=model.input, outputs=model.layers[-3].output)
res_fine = model4.predict(images)
joblib.dump(res_fine, "fine_tuned_resnet_xxx.joblib")

