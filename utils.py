"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""
import cv2
import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np

import tensorflow as tf

try:
  range
except:
  xrange = range
  
FLAGS = tf.app.flags.FLAGS


def checkpoint_dir(config):
  if config.is_train:
    return os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
  else:
    return os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")

def read_data(path):
  """
  Read h5 format data file
  
  Args:
    path: file path of desired file
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values
  """
  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    return data, label

def imread(path):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  img = cv2.imread(path)
  return img

def preprocess(path, path2 , scale):
  """
  Preprocess single image file 
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation

  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
  """
  image = imread(path)
  label_ = imread(path2)

  #label_ = modcrop(label, scale)

  # Must be normalized
  input_ = image / 255.
  label_ = label_ / 255.

  #input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
  #input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)

  return input_, label_

def prepare_data(dataset):
  """
  Args:
    dataset: choose train dataset or test dataset
    
    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
  """
  print(dataset)
  if dataset != "Test":
    filenames = os.listdir(dataset)
    data_dir = os.path.join(os.getcwd(), dataset)
    image = glob.glob(os.path.join(data_dir, "*.jpg"))
    data=[]
    label=[]
    for item in image:
        if item.find("_grey")is -1:
          if item.find("_dof")is -1:
            data.append(item)
          else:
            label.append(item)

    print("data: ", len(data))
    print("label: ", len(label))

  else:
      data_dir = os.path.join(os.path.join(os.getcwd(), dataset), "focus_image")
      data = glob.glob(os.path.join(data_dir, "*.jpg"))
      data_dir = os.path.join(os.path.join(os.getcwd(), dataset), "focus_label")
      label = glob.glob(os.path.join(data_dir, "*.jpg"))
      print("data: ", len(data))
      print("label: ", len(label))
  return data,label


def make_data(config, data, label):
  """
  Make input data as h5 file format
  Depending on 'is_train' (flag value), savepath would be changed.
  """
  if not os.path.isdir(os.path.join(os.getcwd(), config.checkpoint_dir)):
    os.makedirs(os.path.join(os.getcwd(), config.checkpoint_dir))

  if config.is_train:
    savepath = os.path.join(os.getcwd(), config.checkpoint_dir +'/train.h5')
  else:
    savepath = os.path.join(os.getcwd(), config.checkpoint_dir +'/test.h5')

  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=label)



def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def input_setup(config):
  """
  Read image files and make their sub-images and saved them as a h5 file format.
  """
  print(config.is_train)
  # Load data path
  if config.is_train:
    data, label = prepare_data(dataset="Train/DoF_Images (2)")
  else:
    data, label = prepare_data(dataset="Test")

  sub_input_sequence = []
  sub_label_sequence = []
  padding = abs(config.image_size - config.label_size) / 2 # 6
  nx = ny = 0

  if config.is_train:
    for i in range(len(data)):
      input_, label_ = preprocess(data[i], label[i], config.scale)

      if len(input_.shape) == 3:
        h, w, c = input_.shape
      else:
        h, w = input_.shape

      for x in range(0, h-config.image_size+1, config.stride):
        if i == 0:
          nx += 1; ny = 0
        for y in range(0, w-config.image_size+1, config.stride):
          if i == 0:
            ny += 1

          sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [33 x 33]
          sub_label = label_[x+int(padding):x+int(padding)+config.label_size
          , y+int(padding):y+int(padding)+config.label_size] # [21 x 21]
          # print(sub_input.shape)
          # print(sub_label.shape)
          # Make channel value
          sub_input = sub_input.reshape([config.image_size, config.image_size, 3])
          sub_label = sub_label.reshape([config.label_size, config.label_size, 3])

          sub_input_sequence.append(sub_input)
          sub_label_sequence.append(sub_label)

  else:
    nx_l = []
    ny_l = []
    for i in range(len(data)):
      input_, label_ = preprocess(data[i], label[i], config.scale)

      if len(input_.shape) == 3:
        h, w, c = input_.shape
      else:
        h, w = input_.shape

      if w >= 4000 or h > 4000:
        input_ = cv2.resize(input_, dsize=(int(w/2), int(h/2)), interpolation=cv2.INTER_AREA)
        label_ = cv2.resize(label_, dsize=(int(w/2), int(h/2)), interpolation=cv2.INTER_AREA)
        w = int(w/2)
        h = int(h/2)

      # Numbers of sub-images in height and width of image are needed to compute merge operation.

      nx = ny = 0
      for x in range(0, h-config.image_size+1, config.stride):
        nx += 1; ny = 0
        for y in range(0, w-config.image_size+1, config.stride):
          ny += 1
          sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [33 x 33]
          sub_label = label_[x+int(padding):x+int(padding)+config.label_size, y+int(padding):y+int(padding)+config.label_size] # [21 x 21]

          sub_input = sub_input.reshape([config.image_size, config.image_size, 3])
          sub_label = sub_label.reshape([config.label_size, config.label_size, 3])

          sub_input_sequence.append(sub_input)
          sub_label_sequence.append(sub_label)
      #print("nx: %d  ny: %d" % (nx, ny))
      nx_l.append(nx)
      ny_l.append(ny)
  """
  len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
  (sub_input_sequence[0]).shape : (33, 33, 1)
  """
  # Make list to numpy array. With this transform
  arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 3]
  arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 3]

  make_data(config, arrdata, arrlabel)
  print("make_data success")
  if config.is_train:
    return nx, ny
  else:
    return nx_l, ny_l, len(data)
    
def imsave(image, path,config):
  cv2.imwrite(os.path.join(os.getcwd(), path), image * 255.)


def merge(images, size,c_dim):
  h, w = images.shape[1], images.shape[2]

  img = np.zeros((h * size[0], w * size[1], c_dim))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j * h: j * h + h, i * w: i * w + w, :] = image
    # cv2.imshow("srimg",img)
    # cv2.waitKey(0)

  return img


def ycbcr2rgb(im):
  xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
  rgb = im.astype(np.float)
  rgb[:, :, [1, 2]] -= 128
  rgb = rgb.dot(xform.T)
  np.putmask(rgb, rgb > 255.0, 255.0)
  np.putmask(rgb, rgb < 0.0, 0.0)
  return np.float32(rgb)

