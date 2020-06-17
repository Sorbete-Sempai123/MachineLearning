# TensorFlow y tf.keras
import tensorflow as tf
from tensorflow import keras

# Librerias de ayuda
import numpy as np
import matplotlib.pyplot as plt

##Obtencion de datos
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

##Verificando el formato de los datos
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)


w, h = len(train_images), 28*28;

x_train = np.zeros((w,h),dtype=np.int32)
it = 0
for i in x_train:
    np.copyto(i,train_images[it].flatten())
    it = it +1

w2, h2 = len(test_images), 28*28;

x_test = np.zeros((w2,h2),dtype=np.int32)
it=0
for i in x_test:
    np.copyto(i,test_images[it].flatten())
    it = it+1

from sklearn.linear_model import LogisticRegression
LogisRegr = LogisticRegression()

##Entrenamineot
LogisRegr.fit(x_train,train_labels)



predictions = LogisRegr.predict(x_test)

score = LogisRegr.score(x_test,test_labels)
print(score)


from sklearn import metrics

cm = metrics.confusion_matrix(test_labels, predictions)
print(cm)
