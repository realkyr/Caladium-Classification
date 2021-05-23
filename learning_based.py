#import library
from handcraft_based import ROOT_PATH
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# funtion onehot example have 3 classes 1 = [1,0,0] , 2 = [0,1,0] , 3 = [0,0,1]
def onehot(Y, nclass=12):
  Y_ = np.zeros((Y.shape[0], nclass))
  for i, y in enumerate(Y):
    Y_[i, Y[i]] = 1
  return Y_

# import data
train = pd.read_csv(ROOT_PATH + 'train_handcraft_based.csv')
test = pd.read_csv(ROOT_PATH + 'test.csv')
train_x = train.iloc[:,3:-2]
test_x = test.iloc[:,3:-2]
train_y = train['labels']
train_y = onehot(train_y)
test_y = test['labels']
test_y = onehot(test_y)

# create model
import tensorflow as tf
d_in = (train_x.shape[1],)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(260, input_shape=d_in,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(520, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(520, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(260, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(130, activation='relu'))
model.add(tf.keras.layers.Dense(12, activation='softmax'))
model.compile(loss=tf.keras.losses.MeanSquaredError(),
                        optimizer=tf.keras.optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,
    name="RMSprop"),metrics=['accuracy'])

# train model and save best model
from keras.callbacks import ModelCheckpoint
train_x = np.array(train_x)
test_x = np.array(test_x)
train_y = np.array(train_y)
test_y = np.array(test_y)
checkpoint = ModelCheckpoint(ROOT_PATH + 'best_model.h5', verbose=1, monitor='val_accuracy',save_best_only=True, mode='auto') 
history = model.fit(train_x, train_y, epochs=1000, validation_data=(test_x,test_y), callbacks=[checkpoint])

# evaluate the model
train_acc = model.evaluate(train_x, train_y, verbose=0)
test_acc = model.evaluate(test_x, test_y, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc[1]*100, test_acc[1]*100))

# plot loss during training
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()

# plot accuracy during training
plt.subplot(212)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()