from utils import (
  read_data, 
  input_setup, 
  imsave,
  ycbcr2rgb,
  merge,
  checkpoint_dir
)

import time
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf

try:
  range
except:
  xrange = range

class SRCNN(object):

  def __init__(self, 
               sess, 
               image_size=33,
               label_size=21,
               c_dim = 3,
               ):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size = image_size
    self.label_size = label_size


    self.c_dim = c_dim


    self.build_model()

  def build_model(self):
    self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
    self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')
    
    self.weights = {
      'w1': tf.Variable(tf.random_normal([9, 9, self.c_dim, 64], stddev=1e-3), name='w1'),
      'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
      'w3': tf.Variable(tf.random_normal([5, 5, 32, self.c_dim], stddev=1e-3), name='w3')
    }
    self.biases = {
      'b1': tf.Variable(tf.zeros([64]), name='b1'),
      'b2': tf.Variable(tf.zeros([32]), name='b2'),
      'b3': tf.Variable(tf.zeros([self.c_dim]), name='b3')
    }

    self.pred = self.model()

    # Loss function (MSE)
    self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))

    self.saver = tf.train.Saver()

  def train(self, config):
    if config.is_train:
      nx, ny = input_setup(config)
    else:
      nx, ny , num_data = input_setup(config)

    data_dir = checkpoint_dir(config)
    train_data, train_label = read_data(data_dir)
    print("take data success!\n")
    # Stochastic gradient descent with the standard backpropagation
    self.train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)

    tf.initialize_all_variables().run()
    
    counter = 0

    start_time = time.time()

    self.load(config.checkpoint_dir)


    if config.is_train:
      print("Training...")
      batch_size = 128
      test_batch_size = nx * ny
      for ep in range(config.epoch):
        # Run by batch images
        batch_idxs = len(train_data) // batch_size
        for idx in range(0, batch_idxs):
          batch_images = train_data[idx*batch_size : (idx+1)*batch_size]
          batch_labels = train_label[idx*batch_size : (idx+1)*batch_size]

          counter += 1
          _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})

          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err))

          if counter % 500 == 0:
            self.save(config.checkpoint_dir, counter)

          if ep % 1000 == 0:
            result = self.pred.eval({self.images: train_data[0:test_batch_size],
                                     self.labels: train_label[0:test_batch_size]})
            result = merge(result, [nx, ny], self.c_dim)
            #result = result.squeeze()
            imsave(result,"train_result/"+"result"+str(ep)+".png",config)


    else:
      print("Testing...")
      old_batch = 0
      for j in range(num_data):
        nx_x = nx[j]
        ny_y = ny[j]
        test_batch_size = nx_x * ny_y
        #result = self.pred.eval({self.images: train_data, self.labels: train_label})
        result = self.pred.eval({self.images: train_data[old_batch: old_batch + test_batch_size]})
        #print("old: %d , batch_size: %d" % (old_batch , test_batch_size))
        #result = merge(result, [nx, ny])
        image = merge(result, [nx_x, ny_y], self.c_dim)
        # result = result.squeeze()
        # image_path = os.path.join(os.getcwd(), config.sample_dir)
        # image_path = os.path.join(image_path, "test_image.png")
        # imsave(result, image_path)
        print("timg: [%4.4f]" %  (time.time()-start_time))
        imsave(image, "test_result" + '/result'+str(j)+'.png', config)
        old_batch += test_batch_size

  def model(self):
    conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='VALID') + self.biases['b1'])
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='VALID') + self.biases['b2'])
    conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='VALID') + self.biases['b3']
    return conv3

  def save(self, checkpoint_dir, step):
    model_name = "SRCNN.model"
    model_dir = "%s_%s" % ("srcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print("\nReading Checkpoints.....\n\n")
    model_dir = "%s_%s" % ("srcnn", self.label_size)  # give the model name by label_size
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    # Check the checkpoint is exist
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_path = str(ckpt.model_checkpoint_path)  # convert the unicode to string
      self.saver.restore(self.sess, os.path.join(os.getcwd(), ckpt_path))
      print("\n Checkpoint Loading Success! %s\n\n" % ckpt_path)
    else:
      print("\n! Checkpoint Loading Failed \n\n")
