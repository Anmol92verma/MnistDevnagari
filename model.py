import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import os
import cv2

NUMBER_OF_PREDICTIONS = 0
IMG_SIZE = 32

DATADIR = "/Users/anmolverma/ml/Images/"
training_data = []
keymap = []


def load_data():
    global NUMBER_OF_PREDICTIONS
    for category in os.listdir(DATADIR):
        bdirpath = os.path.join(DATADIR, category)
        if os.path.isdir(bdirpath):
            path = os.path.join(DATADIR, category)
            class_num = NUMBER_OF_PREDICTIONS
            keymap.append([NUMBER_OF_PREDICTIONS, path])
            NUMBER_OF_PREDICTIONS += 1
            for subdir in os.listdir(path):
                img = os.path.join(path, subdir)
                img_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])


load_data()
X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

x_train = tf.keras.utils.normalize(X, axis=1)  # scales data between 0 and 1
x_test = tf.keras.utils.normalize(x_test, axis=1)  # scales data between 0 and 1


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE)))  # Flattening the 2D arrays for fully connected layers
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(NUMBER_OF_PREDICTIONS, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)

keras_file = "keras_model.h5"
tf.keras.models.save_model(model, keras_file)
predictions = model.predict(x_test)
print("predictions " + str(predictions))
print(predictions[0])
print("predictions label " + str(np.argmax(predictions[0])))
print(keymap)
plt.imshow(x_test[0],cmap="gray")
# Convert to TensorFlow Lite model.
converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(keras_file)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
