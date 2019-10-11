from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras.utils import to_categorical
from PIL import Image


#define the data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#define the network architecture
input_tensor = layers.Input(shape=(784,))
x = layers.Dense(32, activation="relu",)(input_tensor)
output_tensor = layers.Dense(10, activation="softmax")(x)
network = models.Model(inputs=input_tensor, outputs=output_tensor)
#choose the optimizer,loss function and metrics
network.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

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