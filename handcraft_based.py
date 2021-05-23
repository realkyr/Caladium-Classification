from FeatureExtractors import *
from Utils import *
from Album import *
import sklearn.neighbors as sn
from sklearn import svm

ROOT_PATH = '' # if use absolute path. path goes here like C:/document/

# ===============================
# training session
# ===============================
class_number = 12 # number of classes
for label in range(class_number):
  # loop for each classes
  dynamic_input_dir = ROOT_PATH + 'Train/' + str(label+1)

  # create train images classes
  train_images = Album()
  train_images.init_album(dynamic_input_dir)
  # create label's column
  train_images.set_labels([label] * train_images.get_length())

  # extract feature
  extract_feature(train_images)

  # export as csv file
  df = train_images.get_dataframe()
  df.to_csv(ROOT_PATH + 'train_handcraft_based.csv', mode=("w" if label == 0 else "a"), header=(label == 0))

# ===============================
# testing session
# ===============================
# read test set
data_set = pd.read_csv(ROOT_PATH + 'train_handcraft_based.csv')
# select all column exept 
train_set = data_set.values[:, 1:-2]

# select last column as a label
test_label = data_set.values[:, -1]
dynamic_input_dir = 'Test/'

# create an Test Image's album
test_images = Album()
for i in range(1, 13):
  test_images.init_album(dynamic_input_dir + str(i))

# extracting feature
extract_feature(test_images)
# get the feature vector of test images
featureTs = test_images.get_feature_vectors()

# normalize
# all data concate between train set's features and test set's features
all_dataset = np.concatenate((train_set, featureTs), axis=0)
# find max and min attribute
min_attribute = [ min(all_dataset[:, i]) for i in range(test_images.get_n_features())]
max_attribute = [ max(all_dataset[:, i]) for i in range(test_images.get_n_features())]

def normalize(x, x_min, x_max):
  # normalize equation
  if x_max-x_min == 0:
    # prevent devided by zero
    return 0
  return (x-x_min)/(x_max-x_min)

# normalize train set
train_set = list(map(
  lambda x: [
    normalize(x[i], min_attribute[i], max_attribute[i])
    for i in range(test_images.get_n_features())
  ], train_set
))

# normalize test set feature
featureTs = list(map(
  lambda x: [
    normalize(x[i], min_attribute[i], max_attribute[i])
    for i in range(test_images.get_n_features())
  ], featureTs
))


# classify
# knn neighbors for testing
# classifier = sn.KNeighborsClassifier(n_neighbors=1, metric='euclidean')
# classifier.fit(train_set, label)
# out = classifier.predict(featureTs)


# create SVM Object; Its type is one versus one; ‘ovr’ for one versus rest
# using linear kernel for linear data
clf = svm.SVC(kernel='linear', decision_function_shape='ovr')
test_label = test_label.astype('int')
clf.fit(train_set, test_label)
out = clf.predict(featureTs)

test_set_name = test_images.get_names()
new_label = []
print(len(featureTs))

# print(max_attribute[0])
# print(min_attribute[0])
for i in range(len(featureTs)):
  print(test_set_name[i])
  # print(featureTs[i][0])
  new_label.append(int(out[i]))
  print(out[i])
test_images.set_labels(new_label)
df = test_images.get_dataframe()
df.to_csv(ROOT_PATH + 'test.csv')