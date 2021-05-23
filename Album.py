from glob import *
from Utils import *
import pandas as pd

class Album:
  def __init__(self):
    # class parameters
    self.length = 0
    self.color_images = []
    self.binary_images = []
    self.grayscale_images = []
    self.contour_images = []
    self.names = []
    self.features_vectors = []
    self.features_names = []
    self.labels = []

  def init_album(self, path):
    files = glob(path+'/*.jpg')
    images_gray = []
    images_color = []
    contour_images = []
    ids = []

    for f in files:
      # read all image in different format
      images_gray.append(read_image_grayscale(f))
      images_color.append(read_image_color(f))
      ids.append(f)

    for img in images_gray:
      # convert grayscale into binary contour image
      contour_images.append(get_binary_image_contours(img))

    self.length += len(ids)
    self.grayscale_images += images_gray
    self.color_images += images_color
    self.names += ids
    self.contour_images += contour_images
  
  def get_names(self):
    return self.names

  def get_contour_images(self):
    return self.contour_images

  def get_grayscale_images(self):
    return self.grayscale_images

  def get_images(self):
    return self.color_images
  
  def set_feature_vectors(self, vectors):
    self.features_vectors = vectors
  
  def set_feature_names(self, names):
    self.features_names = names

  def get_feature_names(self):
    return self.features_names

  def get_feature_vectors(self):
    return self.features_vectors

  def get_length(self):
    return self.length

  def set_labels(self, labels):
    self.labels = labels
  
  def get_dataframe(self):
    # save extracting feature into csv file
    if self.length == 0:
      print("no data in album")
      return
    n_feature = len(self.features_vectors[0])
    data = {
      self.features_names[i] : [self.features_vectors[j][i] for j in range(self.length)] for i in range(n_feature)
    }
    data['id'] = self.names
    if len(self.labels):
      data['labels'] = self.labels
    df = pd.DataFrame(data)
    return df

  def get_n_features(self):
    if self.length == 0:
      print("no data in album")
      return
    n_feature = len(self.features_vectors[0])
    return n_feature