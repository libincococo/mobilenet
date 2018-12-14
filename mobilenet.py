import tensorflow as tf


def create_variable(name,shape,initializer,dtype=tf.float32,trainable=True):
    return tf.get_variable(name,shape=shape,dtype=dtype,initializer=initializer,trainable=trainable)

def conv2d(inputs,scope,num_filters,filter_size=1,strides=1):
    inputs_shape = inputs.get_shape()
    in_channels = inputs_shape[-1]

    with tf.variable_scope(scope):
        filters = create_variable('filter',shape=[filter_size,filter_size,in_channels,num_filters],
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        return tf.nn.conv2d(inputs,filters,strides=[1,strides,strides,1],padding="SAME")


class MobileNet(object):
    def __init__(self,inputs,num_classes=100,is_training=true,width_multipliter=1,scope='MobileNet'):
        self.inputs = inputs
        self.num_classes=num_classes
        self.is_training = is_training
        self.width_multipliter = width_multipliter
        self.scope = scope

    def build_model(self):
        with tf.variable_scope(self.scope):
            net = conv2d(self.inputs,self.scope,)