import tensorflow as tf
import os,io
from PIL import Image, ImageDraw, ImageFont
import PIL
import matplotlib.pyplot as plt
import cv2
import numpy as np

NUMS_BATCH = 10

IMAGE_W = 32
IMAGE_H = 32
IMAGE_C = 3


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
    #return tf.clip_by_value(image, 0.0, 1.0)
    return  image



def read_tfrecord(filename):
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
    heightf = tf.cast(heightf,tf.int32)
    image_png = tf.image.decode_png(imagef)
    image_png = tf.reshape(image_png,[IMAGE_H,IMAGE_W,3])
    image_png = tf.image.resize_images(image_png,[224,224],method=3)
    image_png = distort_color(image_png,np.random.randint(2))
    image_png = tf.clip_by_value(image_png, 0.0, 1.0)
    label = tf.cast(labelf,tf.int32)
    label = tf.one_hot(label,20,1,0)
    return image_png,label

def get_batch(filename,batch_size=10,num_threads=3,shuffle=False,min_after_dequeue=None):
    image,label = read_tfrecord(filename)

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
                                               allow_smaller_final_batch=False)
    print("get the image...")
    return img_batch,label_batch

def test_read_tfrecord(image,label):
    #test
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        for i in range(10):
            image_d,label_r = sess.run([image,label])

            plt.imshow(image_d[0])
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
    #image,label = read_tfrecord("cifar10/cifar10_train.tfrecord")
    image, label = get_batch("cifar10/cifar10_train.tfrecord",shuffle=False)
    test_read_tfrecord(image,label)