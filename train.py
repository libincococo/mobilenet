import tensorflow as tf
from mobilenet import MobileNet
#import MobileNet
from utils import *
from config import *
from utils import *

import time
import glob
import os

def load(sess, saver, checkpoint_dir):
    import re
    print("[*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print("[*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print("[*] Failed to find a checkpoint")
        return False, 0



def train():
    height = args.height
    width = args.width
    _step = 0


    if True:
        #glob_pattern = os.path.join(args.dataset_dir,"*_train.tfrecord")
        #tfrecords_list = glob.glob(glob_pattern)
        #filename_queue = tf.train.string_input_producer(tfrecords_list, num_epochs=None)
        img_batch, label_batch = get_batch("cifar10/cifar10_train.tfrecord", args.batch_size)

        mobilenet = MobileNet(img_batch,num_classes=args.num_classes)


        logits = mobilenet.logits
        pred = mobilenet.predictions

        cross = tf.nn.softmax_cross_entropy_with_logits(labels=label_batch,logits=logits)
        loss = tf.reduce_mean(cross)

        # L2 regularization
        list_reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(list_reg)>0:
            l2_loss = tf.add_n(list_reg)
            total_loss = loss + l2_loss
        else:
            total_loss = loss

        # evaluate model, for classification
        preds = tf.argmax(pred,1)
        labels = tf.argmax(label_batch, 1)
        #correct_pred = tf.equal(tf.argmax(pred, 1), tf.cast(label_batch, tf.int64))
        correct_pred = tf.equal(preds,labels)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # learning rate decay
        base_lr = tf.constant(args.learning_rate)
        global_step = tf.Variable(0)
        lr = tf.train.exponential_decay(args.learning_rate,
                                        global_step=global_step,
                                        decay_steps=args.lr_decay_step,
                                        decay_rate=args.lr_decay,
                                        staircase=True)

        # optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=args.beta1).minimize(loss,global_step=global_step)

        max_steps = int(args.num_samples/int(args.batch_size)*int(args.epoch))

        # summary
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('accuracy', acc)
        tf.summary.scalar('learning_rate', lr)
        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver()
        _step = 0

        with tf.Session() as sess:

            # summary writer
            writer = tf.summary.FileWriter(args.logs_dir, sess.graph)

            _,_step = load(sess,saver,args.checkpoint_dir)
            print("the start run init")
            sess.run(tf.global_variables_initializer())
            print("the end  run init")

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for step in range(_step+1,max_steps+1):
                start_time = time.time()

                _,_lr = sess.run([train_op,lr])
                #print("the loss is %f"%lr)
                if step % args.num_log == 0:
                    summ,_loss, _acc = sess.run([summary_op,total_loss, acc])
                    writer.add_summary(summ, step)
                    print('global_step:{0}, time:{1:.3f}, lr:{2:.8f}, acc:{3:.6f}, loss:{4:.6f}'.format
                        (step, time.time() - start_time, _lr, _acc, _loss))

                if step % args.num_log == 0:
                    save_path = saver.save(sess, os.path.join(args.checkpoint_dir, args.model_name), global_step=step)

            tf.train.write_graph(sess.graph_def, args.checkpoint_dir, args.model_name + '.pb')
            save_path = saver.save(sess, os.path.join(args.checkpoint_dir, args.model_name), global_step=max_steps)

            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    train()
    pass