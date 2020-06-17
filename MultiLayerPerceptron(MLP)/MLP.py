# TensorFlow y tf.keras
import tensorflow as tf
from tensorflow import keras

# Librerias de ayuda
import numpy as np
import matplotlib.pyplot as plt

##Obtencion de datos
fashion_mnist = keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

w, h = len(x_train), 28*28;

x_trainF = np.zeros((w,h),dtype=np.int32)
it = 0
for i in x_trainF:
    np.copyto(i,x_train[it].flatten())
    it = it +1

w2, h2 = len(x_test), 28*28;

x_testF = np.zeros((w2,h2),dtype=np.int32)
it=0
for i in x_testF:
    np.copyto(i,x_test[it].flatten())
    it = it+1

print("MPL")

from sklearn.neural_network import MLPClassifier
# all parameters not specified are set to their defaults
mlperceptron = MLPClassifier(solver='sgd', hidden_layer_sizes=(25), activation='logistic',max_iter=500, early_stopping=True,random_state=0)
mlperceptron.fit(x_trainF, y_train)

predictions = mlperceptron.predict(x_testF)


# Use score method to get accuracy of model
score = mlperceptron.score(x_testF, y_test)
print(score)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, predictions, normalize=False)
print("Number of Correct predicted: " + str(acc) +" of "+ str(len(y_test)))
print(acc / len(y_test))

from sklearn.metrics import f1_score
f1_score(y_test, predictions, average='macro')

