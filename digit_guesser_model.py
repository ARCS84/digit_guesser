import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from matplotlib import pyplot
import numpy as np
from tensorflow.keras.preprocessing import image

mnist_data = tfds.load(name="mnist")

for item in mnist_data:
    print(item)

class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.99):
            print("\nreached 99% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallBack()

(training_images, training_labels),(test_images, test_labels) = tfds.as_numpy(tfds.load(name='mnist', split=['train', 'test'], batch_size=-1, as_supervised=True))
print(type(test_images))
training_images = training_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    #optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])

model.evaluate(test_images, test_labels)

model.save('digit_guesser.model')