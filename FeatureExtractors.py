# ========================
# Import needed library
# ========================
import cv2
import numpy as np
from numpy import *
from sys import *

# ========================
# this session adapted from ruipoliveira's leaves-classification-opencv project
# ========================

# ========================
# This section is function constructure
# ========================
def get_corner_points(img, maxFeat):
  
  feature_params = dict( maxCorners = maxFeat, qualityLevel = 0.6, minDistance = 7, blockSize = 7 )
  # finding corner using  goodFeaturesToTrack
  corners = cv2.goodFeaturesToTrack(img, mask = None, **feature_params)
  return corners

def get_edge_points(img):
  edges = []
  # using canny edge detection to find edge
  canny_image = cv2.Canny(img, 100, 200)

  for i in range(len(canny_image)):
    for j in range(len(canny_image[0])):
      if canny_image[i][j] == 255:
        # selecting strong edge
        edges.append((i,j))
  # return edge
  return edges

def max_x_diff(img):
  # finding the point of the most width or height in x
  edge_points = get_edge_points(img)

  min_x = maxsize
  max_x = -1

  for point in edge_points:
    x = point[0]
    
    if x > max_x:
      max_x = x
    if x < min_x:
      min_x = x

  x_diff = 1.0*max_x - min_x
  # return result x_diff 
  return x_diff

def max_y_diff(img):
  # finding the point of the most width  or height in y
  edge_points = get_edge_points(img)

  min_y = maxsize
  max_y = -1

  for point in edge_points:
    y = point[1]
    
    if y > max_y:
      max_y = y
    if y < min_y:
      min_y = y

  y_diff = 1.0*max_y - min_y
  #return result y_diff
  return y_diff


# ========================
# this section is extractor setion, write in Class
# ========================
class FeatureExtractors:

  # ========================
  # this is a series of feature extraction method
  # ========================
  def length_width_ratio_feature_extractor(self, image):
    # finding ratio between width and height
    ratio = 0
    
    # finding the end of each axis
    x_diff = max_x_diff(image)
    y_diff = max_y_diff(image)

    # calculate ratio
    if y_diff > x_diff:
      ratio = y_diff/x_diff
    else:
      ratio = x_diff/y_diff

    features = array([ratio])
    feature_names = array(['length_width_ratio'])
    print('.', end="")
    return (feature_names, features)

  def corner_count_feature_extractor(self, image):
    # count strong corner as a feature
    corners = get_corner_points(image, 100)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    features = array([len(corners)])
    feature_names = array(['number_corners'])

    # corners = np.int0(corners)
    # for i in corners:
    #     x, y = i.ravel()
    #     cv2.circle(image, (x,y),3,255,-1)
    # plt.imshow(image)
    # plt.show()
    print('.', end="")
    return (feature_names, features)

  def hsv_color_extractor(self, image):
    # TODO: extract color hsv
    out = image[:,:,0].reshape(1,-1)
    hist, bins = np.histogram(out, bins=np.arange(-0.5,256,1))
    feature = hist/np.sum(hist)
    feature_names = array(['hsv_bin' + str(i+1) for i in range(256)])
    print('.', end="")
    return (feature_names, feature)

  def hog_texture_extractor(self, image):
    # cv2.imshow('hello', image)
    # cv2.waitKey(0)
    
    # defind needed variable
    winSize = (20,20)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradients = True
    
    #Construct HOG with HOG descriptor
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,
      cellSize,nbins,derivAperture,
      winSigma,histogramNormType,L2HysThreshold,
      gammaCorrection,nlevels, signedGradients)
    image = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
    hist = hog.compute(image)
    features = np.array(hist).flatten()
    feature_names = array(['hog_' + str(i+1) for i in range(len(features))])
    print('.', end="")
    return (feature_names, features)

  def feature_extractor_with_corner_count(self, image, c_image, g_imgs):
    #get feature info from selected extractor
    (feature_names1, features1) = self.corner_count_feature_extractor(image)
    (feature_names2, features2) = self.length_width_ratio_feature_extractor(image)
    (feature_names6, features6) = self.hsv_color_extractor(c_image)
    
    features = array(list(features1) + list(features2) + list(features6))
    feature_names = array(list(feature_names1) + list(feature_names2) + list(feature_names6))
    print()
    return (feature_names, features)

  def all_feature_extractor(self, image, c_image, g_imgs):
    #get feature info from selected extractor
    (feature_names1, features1) = self.corner_count_feature_extractor(image)
    (feature_names2, features2) = self.length_width_ratio_feature_extractor(image)
    (feature_names3, features3) = self.hog_texture_extractor(image)
    (feature_names6, features6) = self.hsv_color_extractor(c_image)
    
    features = array(list(features1) + list(features2) + list(features3) + list(features6))
    feature_names = array(list(feature_names1) + list(feature_names3) + list(feature_names2) + list(feature_names6))
    print()
    return (feature_names, features)
  
  def feature_extractor_only_hue(self, image, c_image, g_imgs):
    #get feature info from selected extractor
    (feature_names6, features6) = self.hsv_color_extractor(c_image)
    
    features = array(list(features6))
    feature_names = array(list(feature_names6))
    print()
    return (feature_names, features)
