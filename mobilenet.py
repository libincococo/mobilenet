import tensorflow as tf
from tensorflow.python.training import moving_averages
UPDATE_OPS_COLLECTION = "_update_ops_"

def create_variable(name,shape,initializer,dtype=tf.float32,trainable=True):
    return tf.get_variable(name,shape=shape,dtype=dtype,initializer=initializer,trainable=trainable)

def conv2d(inputs,scope,num_filters,filter_size=1,strides=1):
    inputs_shape = inputs.get_shape()
    in_channels = inputs_shape[-1]

    with tf.variable_scope(scope):
        filters = create_variable('filter',shape=[filter_size,filter_size,in_channels,num_filters],
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        return tf.nn.conv2d(inputs,filters,strides=[1,strides,strides,1],padding="SAME")

# 批规范化 归一化层 BN层 减均值除方差 batchnorm layer
# s1 = W*x + b
# s2 = (s1 - s1均值)/s1方差
# s3 = beta * s2 + gamma
def bacthnorm(inputs, scope, epsilon=1e-05, momentum=0.99, is_training=True):
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

def depthwise_conv2d(inputs,scope,filter_size=3,channel_multiplier=1,strides=1):
    inputs_shape = inputs.get_shape().as_list()  # 输入通道 形状尺寸 64*64* 512
    in_channels = inputs_shape[-1]  # 输入通道数量 最后一个参数 512

    with tf.variable_scope(scope):
        filters = create_variable("filter",shape=[filter_size,filter_size,in_channels,channel_multiplier],
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        return tf.nn.depthwise_conv2d(inputs,filters,strides=[1,strides,strides,1],padding='SAME',rate=[1,1])

def avg_pool(inputs,pool_size,scope):
    with tf.variable_scope(scope):
        return tf.nn.avg_pool(inputs,[1,pool_size,pool_size,1],
                              strides=[1,pool_size,pool_size,1],padding='VALID')


def fc(inputs,n_out,scope,use_bias=True):
    inputs_shape = inputs.get_shape().as_list()  # 输入通道 形状尺寸 64*64* 512
    in_channels = inputs_shape[-1]  # 输入通道数量 最后一个参数 512

    with tf.variable_scope(scope):
        weight = create_variable("weight",shape=[in_channels,n_out],
                                 initializer=tf.random_normal_initializer(stddev=0.01))
        bias = create_variable("bias", shape=[n_out, ],
                               initializer=tf.zeros_initializer())
        if use_bias:
            return tf.nn.xw_plus_b(inputs,weights=weight,biases=bias)
        return tf.matmul(inputs,weight)

class MobileNet(object):
    def __init__(self,inputs,num_classes=100,is_training=True,width_multipliter=1,scope='MobileNet'):
        self.inputs = inputs
        self.num_classes=num_classes
        self.is_training = is_training
        self.width_multiplier = width_multipliter
        self.scope = scope
        self.build_model()

    def build_model(self):
        with tf.variable_scope(self.scope):
            net = conv2d(self.inputs,"conv_1",round(32*self.width_multiplier),
                         filter_size=3,strides=2)
            self.tests = net
            net = bacthnorm(net,"conv_1/bn",is_training=self.is_training)
            net = tf.nn.relu(net,name="conv_1/relu")

            net = self._depthwise_separable_conv2d(net, 64, self.width_multiplier,
                                                   "ds_conv_2")  # ->[N, 112, 112, 64]
            ###################### b.  MobileNet 核心模块 128输出 卷积步长2 尺寸减半
            net = self._depthwise_separable_conv2d(net, 128, self.width_multiplier,
                                                   "ds_conv_3", downsample=True)  # ->[N, 56, 56, 128]
            ###################### c.  MobileNet 核心模块 128输出 卷积步长1 尺寸不变
            net = self._depthwise_separable_conv2d(net, 128, self.width_multiplier,
                                                   "ds_conv_4")  # ->[N, 56, 56, 128]
            ###################### d.  MobileNet 核心模块 256 输出 卷积步长2 尺寸减半
            net = self._depthwise_separable_conv2d(net, 256, self.width_multiplier,
                                                   "ds_conv_5", downsample=True)  # ->[N, 28, 28, 256]
            ###################### e.  MobileNet 核心模块 256输出 卷积步长1 尺寸不变
            net = self._depthwise_separable_conv2d(net, 256, self.width_multiplier,
                                                   "ds_conv_6")  # ->[N, 28, 28, 256]
            ###################### f.  MobileNet 核心模块 512 输出 卷积步长2 尺寸减半
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                                   "ds_conv_7", downsample=True)  # ->[N, 14, 14, 512]
            ###################### g.  MobileNet 核心模块 512输出 卷积步长1 尺寸不变
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                                   "ds_conv_8")  # ->[N, 14, 14, 512]
            ###################### h.  MobileNet 核心模块 512输出 卷积步长1 尺寸不变
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                                   "ds_conv_9")  # ->[N, 14, 14, 512]
            ###################### i.  MobileNet 核心模块 512输出 卷积步长1 尺寸不变
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                                   "ds_conv_10")  # ->[N, 14, 14, 512]
            ###################### j.  MobileNet 核心模块 512输出 卷积步长1 尺寸不变
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                                   "ds_conv_11")  # ->[N, 14, 14, 512]
            ###################### k.  MobileNet 核心模块 512输出 卷积步长1 尺寸不变
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier,
                                                   "ds_conv_12")  # ->[N, 14, 14, 512]
            ###################### l.  MobileNet 核心模块 1024输出 卷积步长2 尺寸减半
            net = self._depthwise_separable_conv2d(net, 1024, self.width_multiplier,
                                                   "ds_conv_13", downsample=True)  # ->[N, 7, 7, 1024]
            ###################### m.  MobileNet 核心模块 1024输出 卷积步长1 尺寸不变
            net = self._depthwise_separable_conv2d(net, 1024, self.width_multiplier,
                                                   "ds_conv_14")  # ->[N, 7, 7, 1024]
            net = avg_pool(net,7,"avg_pool_15")
            net = tf.squeeze(net, [1, 2], name="SpatialSqueeze")  # 去掉维度为1的维[N, 1, 1, 1024] => [N,1024]
            self.logits = fc(net,self.num_classes,"fc_16",use_bias=True)
            self.predictions = tf.nn.softmax(self.logits,name="softmax_17")


    def _depthwise_separable_conv2d(self,inputs,num_filters,width_multiplier,scope,downsample=False):
        num_filters = round(num_filters * width_multiplier)  # 输出通道数量
        strides = 2 if downsample else 1  # 下采样 确定卷积步长

        with tf.variable_scope(scope):
            dw_conv=depthwise_conv2d(inputs,'depthwise_conv',filter_size=3,
                                     channel_multiplier=num_filters,strides=strides)
            bn = bacthnorm(dw_conv,'dw_bn',is_training=self.is_training)
            relu = tf.nn.relu(bn,name='relu1')
            pw_conv=conv2d(relu,"pw",num_filters=num_filters)
            bn = bacthnorm(pw_conv,"pw_bn",is_training=self.is_training)
            return tf.nn.relu(bn,name="relu2")

if __name__ == "__main__":
    # test data
    inputs = tf.random_normal(shape=[1, 224, 224, 3],name="input")# 4张图片 224*224 大小 3通道
    mobileNet = MobileNet(inputs)# 网络模型输出
    graph = tf.get_default_graph()
    writer = tf.summary.FileWriter("./logs", graph=graph)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        pred = sess.run(mobileNet.predictions)#预测输出
        print(pred.shape)#打印
