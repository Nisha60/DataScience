#import libararies
import tensorflow as tf
import matplotlib.pyplot as plt

#Load the MNIDT dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

#Display the first image in the training dataset
plt.imshow(x_train[0], cmap='gray')
plt.title(f'label: {y_train[0]}')
plt.show()

