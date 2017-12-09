from keras.applications.resnet50 import ResNet50, preprocess_input
# from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.metrics import categorical_accuracy
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
import numpy as np
import joblib

def preprocess(imgs):
    # shape: (# samples, flattened 1 x 224 x 224 x 3 dimensions)
    X = []
    for img in imgs:
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        X.extend(img)
    return np.array(X)

def filter_labels(X, y, threshold=0.1):
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
    return X, y

class_label = 'multi' # SET THIS FIRST: "single" or "multi"
print('Running ResNet on {} label'.format(class_label))

'''
Reference: https://keras.io/applications/#usage-examples-for-image-classification-models
'''
X = joblib.load('resnet_preprocessed.joblib')
X = preprocess(X)
if class_label == 'single':
    one_hot, _, _ = joblib.load('labels.joblib')
    y_array = np.array(one_hot)
else:
    one_hot, _, _ = joblib.load('labels_multiclass.joblib')
    y_array = np.array(one_hot)
    X, y_array = filter_labels(X, y_array)
X_train, X_test, y_train_array, y_test_array = train_test_split(X, y_array, random_state=42)


# create base pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False)

# add global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# add fully conected layer
x = Dense(1024, activation='relu')(x)
# add logistic layer to make prediction on 11 classes
predictions = Dense(y_array.shape[1], activation='softmax')(x)

# model to train
model = Model(inputs=base_model.input, outputs=predictions)

# train only top layer ie. freeze convolutional layers
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# initial training (only top layer)
model.fit(X_train, y_train_array, epochs = 10, verbose=1)

# train only the top 2 inception block
# ie. freeze first 249 and unfreeze the rest
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

# save (checkpoint) model every 3 epochs
epoch_file = "resnet_" + class_label + "_model-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(epoch_file, period=3)
callbacks_list = [checkpoint]

# train to fine-tune convolutional layers
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=[categorical_accuracy])
hist = model.fit(X_train, y_train_array, epochs=10, validation_data=(X_test, y_test_array), verbose=1, callbacks=callbacks_list)
print('fine tune scores: {}'.format(hist.history))
model.save('resnet_' + class_label + '_model.hdf5')

scores = model.evaluate(X_test, y_test_array)
print('Validation score: {} loss {} accuracy'.format(scores[0], scores[1]))
