import tensorflow as tf
from tensorflow.python.training import moving_averages

UPDATE_OPS_COLLECTION = "_update_ops_"

def create_variable(name,shape,initializer,dtype=tf.float32,trainable=True):
    return tf.get_variable(name,shape=shape,dtype=dtype,initializer=initializer,trainable=trainable)

def conv2d(inputs,scope,num_filters,filter_size=1,strides=1,weight_decay=0):
    inputs_shape = inputs.get_shape().as_list()
    in_channels = inputs_shape[-1]
    #print("conv2d shape:")
    #print(inputs_shape)

    with tf.variable_scope(scope):
        filters = create_variable('filter',shape=[filter_size,filter_size,in_channels,num_filters],
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        if weight_decay != 0:
            l2_reg = tf.contrib.layers.l2_regularizer(weight_decay)(filters)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,l2_reg)
        return tf.nn.conv2d(inputs,filters,strides=[1,strides,strides,1],padding="SAME")

def bacthnorm(inputs, scope, epsilon=1e-05, momentum=0.99, is_training=True):

    outputs = tf.layers.batch_normalization(inputs,momentum=momentum,epsilon=epsilon,training=is_training)
    return outputs


# 批规范化 归一化层 BN层 减均值除方差 batchnorm layer
# s1 = W*x + b
# s2 = (s1 - s1均值)/s1方差
# s3 = beta * s2 + gamma
def bacthnorm_old(inputs, scope, epsilon=1e-05, momentum=0.99, is_training=True):


    inputs_shape = inputs.get_shape().as_list() # 输出 形状尺寸
    params_shape = inputs_shape[-1:] # 输入参数的长度
    axis = list(range(len(inputs_shape) - 1))

    with tf.variable_scope(scope):
        beta = create_variable("beta", params_shape,
                               initializer=tf.zeros_initializer())
        gamma = create_variable("gamma", params_shape,
                                initializer=tf.ones_initializer())
        # 均值 常量 不需要训练 for inference
        moving_mean = create_variable("moving_mean", params_shape,
                                      initializer=tf.zeros_initializer(), trainable=False)
        # 方差 常量 不需要训练
        moving_variance = create_variable("moving_variance", params_shape,
                                          initializer=tf.ones_initializer(), trainable=False)
    if is_training:
        mean, variance = tf.nn.moments(inputs, axes=axis)   # 计算均值和方差
        # 移动平均求 均值和 方差  考虑上一次的量 xt = a * x_t-1 +(1-a)*x_now
        update_move_mean = moving_averages.assign_moving_average(moving_mean,
                                                                 mean, decay=momentum)
        update_move_variance = moving_averages.assign_moving_average(moving_variance,
                                                                     variance, decay=momentum)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_move_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_move_variance)
    else:
        mean, variance = moving_mean, moving_variance
    return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)


def depthwise_conv2d(inputs,scope,filter_size=3,channel_multiplier=1,strides=1,weight_decay=0):
    inputs_shape = inputs.get_shape().as_list()  # 输入通道 形状尺寸 64*64* 512
    in_channels = inputs_shape[-1]  # 输入通道数量 最后一个参数 512

    with tf.variable_scope(scope):
        filters = create_variable("filter",shape=[filter_size,filter_size,in_channels,channel_multiplier],
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        if weight_decay != 0:
            l2_reg = tf.contrib.layers.l2_regularizer(weight_decay)(filters)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,l2_reg)

        return tf.nn.depthwise_conv2d(inputs,filters,strides=[1,strides,strides,1],padding='SAME',rate=[1,1])

def avg_pool(inputs,pool_size,scope):
    with tf.variable_scope(scope):
        return tf.nn.avg_pool(inputs,[1,pool_size,pool_size,1],
                              strides=[1,pool_size,pool_size,1],padding='VALID')


def fc(inputs,n_out,scope,use_bias=True,weight_decay=0):
    inputs_shape = inputs.get_shape().as_list()  # 输入通道 形状尺寸 64*64* 512
    in_channels = inputs_shape[-1]  # 输入通道数量 最后一个参数 512

    with tf.variable_scope(scope):
        weight = create_variable("weight",shape=[in_channels,n_out],
                                 initializer=tf.random_normal_initializer(stddev=0.01))
        if weight_decay != 0:
            l2_reg = tf.contrib.layers.l2_regularizer(weight_decay)(weight)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,l2_reg)

        bias = create_variable("bias", shape=[n_out, ],
                               initializer=tf.zeros_initializer())
        if use_bias:
            return tf.nn.xw_plus_b(inputs,weights=weight,biases=bias)
        return tf.matmul(inputs,weight)

def relu(inputs,is_relu6=True,name="relu"):
    if is_relu6:
        return tf.nn.relu6(inputs, name=name)
    else:
        return tf.nn.relu(inputs, name=name)

def con2d_1x1(inputs,num_filters,is_training=True,scope="con2d_1x1"):
    with tf.variable_scope(scope):
        linear_conv = conv2d(inputs, "linear_conv", num_filters=num_filters)
        linear_bn = bacthnorm(linear_conv, "linear_bn", is_training=is_training)
        return linear_bn


def res_block(inputs,out_channels,width_multiplier,num_strides=1,is_relu6=False,is_training=True, scope=None, shortcut=True):
    inputs_shape = inputs.get_shape().as_list()  # 输入通道 形状尺寸 64*64* 512
    in_channels = inputs_shape[-1]  # 输入通道数量 最后一个参数 512

    num_filters = round(in_channels * width_multiplier)  # 输出通道数量
    with tf.variable_scope(scope):
        pw_conv = conv2d(inputs, "dw_conv", num_filters=num_filters)
        bn = bacthnorm(pw_conv, "dw_bn", is_training=is_training)
        relus = relu(bn,name="dw_relu6")

        dc = depthwise_conv2d(relus,scope,strides=num_strides,weight_decay=1e-5)

        net = con2d_1x1(dc,out_channels,is_training=is_training)

        if num_strides == 1 and shortcut:
            if in_channels != out_channels:
                ins = conv2d(inputs, "linear_conv", num_filters=out_channels)
                net = tf.add(ins,net)
            else:
                net = tf.add(inputs,net)
        return net

