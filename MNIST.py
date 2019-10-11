from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras.utils import to_categorical
import cv2
import numpy as np

#define the data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#define the network architecture
network = models.Sequential()
network.add(layers.Dense(512, activation="relu", input_shape=(28*28,)))
network.add(layers.Dense(10, activation="softmax"))
#choose the optimizer,loss function and metrics
network.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
test_images_original = test_images.copy()
train_images = train_images.reshape(60000, 28*28)
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape(10000, 28*28)
test_images = test_images.astype("float32") / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#train, iterate over data
network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print(test_accuracy)
print(test_images.shape)

result = np.array(network.predict(test_images[5][np.newaxis,...]))
idx = np.argmax(result)
print("prediction", idx)

cv2.imshow("digit", test_images_original[5])
cv2.waitKey(0)
