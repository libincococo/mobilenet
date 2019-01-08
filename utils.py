import tensorflow as tf
import os,io
from PIL import Image, ImageDraw, ImageFont
import PIL
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
from preprocess import *
import tensorflow.contrib.eager as tfe

#tf.enable_eager_execution()
NUMS_BATCH = 10


IMAGE_C = 3

def resize_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image

def crop_image(image,hight,width,radio=1.2):
    r = random.randint(1,3)
    if r == 1:
        radio = random.uniform(1,radio)
        image = tf.image.resize_images(image, [int(hight*radio), int(width*radio)], method=1)
        image = tf.image.resize_image_with_crop_or_pad(image,hight,width)

    return image

def distort_color(image, color_ordering = 0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta = 32. / 255.)
        image = tf.image.random_saturation(image, lower = 0.5, upper = 1.5)
        image = tf.image.random_hue(image, max_delta = 0.2)
        image = tf.image.random_contrast(image, lower = 0.5, upper = 1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower = 0.5, upper = 1.5)
        image = tf.image.random_brightness(image, max_delta = 32. / 255.)
        image = tf.image.random_contrast(image, lower = 0.5, upper = 1.5)
        image = tf.image.random_hue(image, max_delta = 0.2)
    elif color_ordering == 2:
        image = tf.image.random_hue(image, max_delta = 0.2)
        image = tf.image.random_saturation(image, lower = 0.5, upper = 1.5)
        image = tf.image.random_brightness(image, max_delta = 32. / 255.)
        image = tf.image.random_contrast(image, lower = 0.5, upper = 1.5)
    #image = tf.clip_by_value(image, 0.0, 1.0)
    return  image



def read_tfrecord(filename,height,width,num_classes=10,is_train=True):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image/encoded': tf.FixedLenFeature([],tf.string),
                                           'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
                                           'image/height': tf.FixedLenFeature([], tf.int64),
                                           'image/width': tf.FixedLenFeature([], tf.int64),
                                           'image/class/label': tf.FixedLenFeature(
                                               [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
                                       })

    imagef,labelf,heightf,widthf = features['image/encoded'],features['image/class/label'],features['image/height'],features['image/width']

    image_png = tf.image.decode_png(imagef)
    image_png = tf.reshape(image_png,[heightf,widthf,3])
    image_png = tf.image.resize_images(image_png,[height,width],method=3)
    if is_train:
        image_png = distort_color(image_png,np.random.randint(2))
        image_png = resize_image(image_png)
        image_png = crop_image(image_png,hight=height,width=width)
    #image_png = tf.clip_by_value(image_png, 0.0, 1.0)
    #image_png = tf.cast(image_png, tf.uint8)
    label = tf.cast(labelf,tf.int32)
    label = tf.one_hot(label,num_classes,1,0)
    return image_png,label

def get_batch(filename,batch_size=10,num_threads=3,shuffle=False,min_after_dequeue=None,height=224,width=224,num_classes=10,is_train=True):
    image,label = read_tfrecord(filename,height=height,width=width,num_classes=num_classes,is_train=is_train)

    if min_after_dequeue is None:
        min_after_dequeue = batch_size * 5
    capacity = min_after_dequeue + 3 * batch_size
    if shuffle:
        img_batch,label_batch = tf.train.shuffle_batch([image,label],
                                                       batch_size=batch_size,
                                                       capacity=capacity,
                                                       num_threads=num_threads,
                                                       min_after_dequeue=min_after_dequeue)
    else:
        img_batch,label_batch = tf.train.batch([image,label],
                                               batch_size=batch_size,
                                               capacity=capacity,
                                               num_threads=num_threads,
                                               allow_smaller_final_batch=True)
    print("get the image...")
    return img_batch,label_batch

def test_read_tfrecord(image,label):
    #test
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        for i in range(10):
            image_d,label_r = sess.run([image,label])
            imagess = tf.cast(image_d[0],tf.uint8)
            imagess = image_d[0]
            plt.imshow(imagess)
            plt.show()


            #print(label_r)
            '''
            image_r = tf.reshape(image_r,[IMAGE_H,IMAGE_W,IMAGE_C])
            plt.imshow(image_r)
            plt.show()
            '''
    #print(image.shape)
        coord.request_stop()
        coord.join()

if __name__ == "__main__":
    #image,label = read_tfrecord("./guesture/guesture_train.tfrecord",height=224,width=224,num_classes=2,is_train=False)
    #image, label = read_tfrecord("./cifar10/cifar10_train.tfrecord", height=224, width=224, num_classes=10,is_train=False)
    image, label = get_batch("./guesture/guesture_train.tfrecord",shuffle=True,batch_size=1,num_classes=2)
    test_read_tfrecord(image,label)