import tensorflow as tf
from tensorflow.python.training import moving_averages
from ops import *

class MobileNetV2(object):
    def __init__(self,inputs,num_classes=10,is_training=True,width_multipliter=1,is_relu6=False,scope='MobileNet'):
        self.inputs = inputs
        self.num_classes=num_classes
        self.is_training = is_training
        self.width_multiplier = width_multipliter
        self.scope = scope
        self.is_relu6=is_relu6
        self.build_model()


    def build_model(self):
        with tf.variable_scope(self.scope):
            net = conv2d(self.inputs, "conv_1", round(32 * self.width_multiplier),
                         filter_size=3, strides=2, weight_decay=1e-4)

            net = bacthnorm(net, "conv_1/bn", is_training=self.is_training)
            self.net = relu(net,name="conv_1/relu")

            net = res_block(net,out_channels=16,width_multiplier=1,num_strides=1,scope="bottleneck_2")

            net = res_block(net, out_channels=24, width_multiplier=6, num_strides=2,scope="bottleneck_3_1")
            net = res_block(net, out_channels=24, width_multiplier=6, num_strides=1,scope="bottleneck_3_2")

            net = res_block(net, out_channels=32, width_multiplier=6, num_strides=2,scope="bottleneck_4_1")
            net = res_block(net, out_channels=32, width_multiplier=6, num_strides=1,scope="bottleneck_4_2")
            net = res_block(net, out_channels=32, width_multiplier=6, num_strides=1,scope="bottleneck_4_3")

            net = res_block(net, out_channels=64, width_multiplier=6, num_strides=2,scope="bottleneck_5_1")
            net = res_block(net, out_channels=64, width_multiplier=6, num_strides=1,scope="bottleneck_5_2")
            net = res_block(net, out_channels=64, width_multiplier=6, num_strides=1,scope="bottleneck_5_3")
            net = res_block(net, out_channels=64, width_multiplier=6, num_strides=1,scope="bottleneck_5_4")

            net = res_block(net, out_channels=96, width_multiplier=6, num_strides=1,scope="bottleneck_6_1")
            net = res_block(net, out_channels=96, width_multiplier=6, num_strides=1,scope="bottleneck_6_2")
            net = res_block(net, out_channels=96, width_multiplier=6, num_strides=1,scope="bottleneck_6_3")

            net = res_block(net, out_channels=160, width_multiplier=6, num_strides=2,scope="bottleneck_7_1")
            net = res_block(net, out_channels=160, width_multiplier=6, num_strides=1,scope="bottleneck_7_2")
            net = res_block(net, out_channels=160, width_multiplier=6, num_strides=1,scope="bottleneck_7_3")


            net = res_block(net, out_channels=320, width_multiplier=6, num_strides=1, scope="bottleneck_8",shortcut=False)

            net = con2d_1x1(net,num_filters=1280,is_training=self.is_training,scope="con2d_1x1_9_1")
            #net = bacthnorm(net,scope="bn_9_2")
            net = relu(net,name="relu_9_2")
            
            net = avg_pool(net, 7, "avg_pool_10")

            net = con2d_1x1(net, num_filters=self.num_classes, is_training=self.is_training, scope="con2d_1x1_12")

            self.logits = tf.layers.flatten(net,name="output")
            self.predictions = tf.nn.softmax(self.logits,name="softmax_13")


if __name__ == "__main__":
    # test data
    inputs = tf.random_normal(shape=[1, 224, 224, 3],name="input")# 4张图片 224*224 大小 3通道
    mobileNet = MobileNetV2(inputs) # 网络模型输出
    graph = tf.get_default_graph()
    #writer = tf.summary.FileWriter("./logs", graph=graph)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        pred = sess.run(mobileNet.predictions)#预测输出
        print(pred.shape)#打印
        print(pred)