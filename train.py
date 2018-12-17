import tensorflow as tf
from mobilenet import MobileNet
#import MobileNet
from utils import *
from config import *
from utils import *

import time
import glob
import os


def train():
    height = args.height
    width = args.width



    if True:
        #glob_pattern = os.path.join(args.dataset_dir,"*_train.tfrecord")
        #tfrecords_list = glob.glob(glob_pattern)
        #filename_queue = tf.train.string_input_producer(tfrecords_list, num_epochs=None)
        img_batch, label_batch = get_batch("cifar10/cifar10_train.tfrecord", args.batch_size)

        mobilenet = MobileNet(img_batch,num_classes=args.num_classes)


        logits = mobilenet.logits
        pred = mobilenet.predictions
        testnet = mobilenet.tests

        cross = tf.nn.softmax_cross_entropy_with_logits(labels=label_batch,logits=logits)
        loss = tf.reduce_mean(cross)
        # L2 regularization
        #l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = loss # + l2_loss


        # evaluate model, for classification
        preds = tf.argmax(pred,1)
        labels = tf.argmax(label_batch, 1)
        #correct_pred = tf.equal(tf.argmax(pred, 1), tf.cast(label_batch, tf.int64))
        correct_pred = tf.equal(preds,labels)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


        # optimizer
        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=args.beta1).minimize(loss)

        with tf.Session() as sess:

            print("the start run init")
            sess.run(tf.global_variables_initializer())
            print("the end  run init")

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for step in range(10000):
                start_time = time.time()

                _,lr = sess.run([train_op,total_loss])
                #print("the loss is %f"%lr)
                if step % 100 == 0:
                    _loss, _acc = sess.run([total_loss, acc])

                    print('global_step:{0}, time:{1:.3f}, lr:{2:.8f}, acc:{3:.6f}, loss:{4:.6f}'.format
                        (step, time.time() - start_time, lr, _acc, _loss))
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    train()
    pass