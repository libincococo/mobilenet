import tensorflow as tf


def create_variable(name,shape,initializer,dtype=tf.float32,trainable=True):
    return tf.get_variable(name,shape=shape,dtype=dtype,initializer=initializer,trainable=trainable)

def conv2d():
    pass


class MobileNet(object):
    def __init__(self,inputs,num_classes=100,is_training=true,width_multipliter=1,scope='MobileNet'):
        self.inputs = inputs
        self.num_classes=num_classes
        self.is_training = is_training
        self.width_multipliter = width_multipliter
        self.scope = scope


    def build_model(self):
        with tf.variable_scope(self.scope):
            net =