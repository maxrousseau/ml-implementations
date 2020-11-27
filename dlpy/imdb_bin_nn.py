import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import models
from tensorflow.keras import layers

# import data
from tensorflow.keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
        num_words=10000)

print(np.shape(train_data) + np.shape(train_labels))
print(train_data[10])
print(train_labels[10])
print(np.shape(test_data) + np.shape(test_labels))

word_index = imdb.get_word_index()
reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join(
        [reverse_word_index.get(i - 3, '?') for i in train_data[10]])
print(decoded_review)


# proprocess and view
# one hot encoding
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


print(str(np.shape(x_train)) + ' || ' +  str(np.shape(x_test)))


# make nn
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# NOTE: use the validation set to iterate over model and hyperparameter tuning
# in oder to address overfitting and avoid biasing model to optimize for test
# set
# short explanation: https://stats.stackexchange.com/questions/19048/what-is-the-difference-between-test-set-and-validation-set
# run and view performance
# here you actually check how your model performs

history = model.fit(
        partial_x_train,
        partial_y_train,
        epochs=20,
        batch_size=512,
        validation_data=(x_val,y_val)
)

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='training loss')
plt.plot(epochs, val_loss_values, 'b', label='validation loss')
plt.title('losses')
plt.xlabel('epocs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.clf()
print(history_dict)

acc = history_dict['accuracy']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='training acc')
plt.plot(epochs, val_acc, 'b', label='validation acc')
plt.title('training accuracy')
plt.xlabel('epocs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# here we train our final model on the whole dataset
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
print(results)
