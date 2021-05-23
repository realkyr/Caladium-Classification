#import library
from Path import ROOT_PATH
from tensorflow import keras
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# import best model
model = keras.models.load_model(ROOT_PATH + 'best_model.h5')

# funtion onehot example have 3 classes 1 = [1,0,0] , 2 = [0,1,0] , 3 = [0,0,1]
def onehot(Y, nclass=12):
  Y_ = np.zeros((Y.shape[0], nclass))
  for i, y in enumerate(Y):
    Y_[i, Y[i]] = 1
  return Y_

#import test data
test = pd.read_csv('test.csv')
test_x = test.iloc[:,3:-2]
test_y = test['labels']
test_y = onehot(test_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

# test model
Z = model.predict(test_x)
result = Z
for i in range(len(result)):
  for j in range(len(result[i])):
    if result[i][j] == max(result[i]):
      result[i][j] = 1
    else:
      result[i][j] = 0
accuracy = model.evaluate(test_x, test_y)
print(accuracy[1]*100)

#confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import sklearn
tranform = []
for i in range(len(result)):
  for j in range(len(result[i])):
    if result[i][j] == 1:
      tranform.append(j)
y_true = test['labels'].tolist()
y_pred = tranform
CM = confusion_matrix(y_true, y_pred)

# precision , recall , f1 score and accuracy score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_true, y_pred)))

print('Micro Precision: {:.2f}'.format(precision_score(y_true, y_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_true, y_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_true, y_pred, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(y_true, y_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_true, y_pred, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_true, y_pred, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(y_true, y_pred, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_true, y_pred, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_true, y_pred, average='weighted')))

from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(y_true, y_pred, target_names=['Class 1', 'Class 2', 'Class 3','Class 4',
                                                          'Class 5','Class 6','Class 7','Class 8',
                                                          'Class 9','Class 10','Class 11','Class 12']))