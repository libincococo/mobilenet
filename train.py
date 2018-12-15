import tensorflow as tf
import os,io
from PIL import Image, ImageDraw, ImageFont
import PIL
import matplotlib.pyplot as plt
import cv2

NUMS_BATCH = 10

IMAGE_W = 32
IMAGE_H = 32
IMAGE_C = 3


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

    image_png = tf.image.decode_png(imagef)
    label = tf.cast(labelf,tf.int32)
    return image_png,label



def test_read_tfrecord(image,label):
    #test
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        for i in range(10):
            image_d,label_r = sess.run([image,label])

            cv2.imshow("this",image_d)
            cv2.waitKey(2000)
            print(label_r.shape)
            '''
            image_r = tf.reshape(image_r,[IMAGE_H,IMAGE_W,IMAGE_C])
            plt.imshow(image_r)
            plt.show()
            '''
    #print(image.shape)

if __name__ == "__main__":
    image,label = read_tfrecord("cifar10/cifar10_train.tfrecord")
    test_read_tfrecord(image,label)
    pass