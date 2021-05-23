import cv2
from FeatureExtractors import *
from math import *
from numpy import *

def read_image_grayscale(image_path):
  # read and return image as grayscale
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  return image

def read_image_color(image_path):
  """ use this function color image """
  # read and return image as color image
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  return image

def get_binary_image_contours(imgray):
  # turn grayscale image into binary image using treshold
  ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY_INV)
  # create black image
  mask = zeros(imgray.shape[:2], uint8)
  # find contours drawing high frequency or draw line from image
  contours, hierarchy = cv2.findContours(thresh,1,2)

  # draw line by draw contours on mask image
  cv2.drawContours(mask,contours,-1,(200,200,0),3)
  return mask

def extract_feature(album):
    b_imgs = album.get_contour_images()
    c_imgs = album.get_images()
    g_imgs = album.get_grayscale_images()
    name = album.get_names()
    feature_vecs = []

    # create feature extractor object
    f_e = FeatureExtractors()
    feature_names = None

    for i in range(album.get_length()):
      # extring each image in album
      print('extringing ' + name[i], end="")
      # get features
      (feature_names, features) = f_e.feature_extractor_with_corner_count(b_imgs[i], c_imgs[i], g_imgs[i])
      feature_vecs.append(features)

    # store features vector in album
    album.set_feature_vectors(array(feature_vecs))
    album.set_feature_names(array(feature_names))
    return feature_vecs, feature_names
