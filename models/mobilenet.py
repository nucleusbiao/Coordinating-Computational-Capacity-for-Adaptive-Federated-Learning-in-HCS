import tensorflow as tf
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_abstract import ModelCNNAbstract


class ModelMobileNet(ModelCNNAbstract):
    """
    MobileNet-based model for apple leaf disease classification.
    Optimized for resource-constrained devices like Raspberry Pi.
    """
    def __init__(self):
        super().__init__()
        pass

    def create_graph(self, learning_rate=None, is_training=True):
        def depthwise_conv2d(x, W, strides=[1, 1, 1, 1]):
            return tf.nn.depthwise_conv2d(x, W, strides, padding='SAME')

        def conv2d(x, W, strides=[1, 1, 1, 1]):
            return tf.nn.conv2d(x, W, strides, padding='SAME')

        def weight_variable(shape):
            initial = tf.random.truncated_normal(shape, stddev=0.1)
            weight = tf.Variable(initial)
            tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(weight))
            return weight

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def batch_norm(x, is_training, decay=0.99):
            scale = tf.Variable(tf.ones([x.get_shape()[-1]]))
            beta = tf.Variable(tf.zeros([x.get_shape()[-1]]))
            pop_mean = tf.Variable(tf.zeros([x.get_shape()[-1]]), trainable=False)
            pop_var = tf.Variable(tf.ones([x.get_shape()[-1]]), trainable=False)

            if is_training:
                batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
                train_mean = tf.compat.v1.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
                train_var = tf.compat.v1.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
                with tf.control_dependencies([train_mean, train_var]):
                    return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, 0.001)
            else:
                return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, 0.001)

        # Input: 224x224x3 images, 4 classes
        self.x = tf.compat.v1.placeholder(tf.float32, shape=[None, 150528])  # 224*224*3
        self.y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
        self.keep_prob = tf.compat.v1.placeholder(tf.float32)
        self.x_image = tf.reshape(self.x, [-1, 224, 224, 3])

        # Layer 1: Depthwise separable convolution (stride 2)
        self.w_depth1 = weight_variable([3, 3, 3, 1])
        self.b_depth1 = bias_variable([3])
        self.w_point1 = weight_variable([1, 1, 3, 32])
        self.b_point1 = bias_variable([32])
        self.h_depth1 = tf.nn.relu(batch_norm(depthwise_conv2d(self.x_image, self.w_depth1, [1, 2, 2, 1]), is_training) + self.b_depth1)
        self.h_point1 = tf.nn.relu(batch_norm(conv2d(self.h_depth1, self.w_point1), is_training) + self.b_point1)

        # Layer 2
        self.w_depth2 = weight_variable([3, 3, 32, 1])
        self.b_depth2 = bias_variable([32])
        self.w_point2 = weight_variable([1, 1, 32, 64])
        self.b_point2 = bias_variable([64])
        self.h_depth2 = tf.nn.relu(batch_norm(depthwise_conv2d(self.h_point1, self.w_depth2, [1, 2, 2, 1]), is_training) + self.b_depth2)
        self.h_point2 = tf.nn.relu(batch_norm(conv2d(self.h_depth2, self.w_point2), is_training) + self.b_point2)

        # Layer 3
        self.w_depth3 = weight_variable([3, 3, 64, 1])
        self.b_depth3 = bias_variable([64])
        self.w_point3 = weight_variable([1, 1, 64, 128])
        self.b_point3 = bias_variable([128])
        self.h_depth3 = tf.nn.relu(batch_norm(depthwise_conv2d(self.h_point2, self.w_depth3, [1, 2, 2, 1]), is_training) + self.b_depth3)
        self.h_point3 = tf.nn.relu(batch_norm(conv2d(self.h_depth3, self.w_point3), is_training) + self.b_point3)

        # Layer 5
        self.w_depth5 = weight_variable([3, 3, 128, 1])
        self.b_depth5 = bias_variable([128])
        self.w_point5 = weight_variable([1, 1, 128, 256])
        self.b_point5 = bias_variable([256])
        self.h_depth5 = tf.nn.relu(batch_norm(depthwise_conv2d(self.h_point3, self.w_depth5, [1, 2, 2, 1]), is_training) + self.b_depth5)
        self.h_point5 = tf.nn.relu(batch_norm(conv2d(self.h_depth5, self.w_point5), is_training) + self.b_point5)

        # Layer 7
        self.w_depth7 = weight_variable([3, 3, 256, 1])
        self.b_depth7 = bias_variable([256])
        self.w_point7 = weight_variable([1, 1, 256, 512])
        self.b_point7 = bias_variable([512])
        self.h_depth7 = tf.nn.relu(batch_norm(depthwise_conv2d(self.h_point5, self.w_depth7, [1, 2, 2, 1]), is_training) + self.b_depth7)
        self.h_point7 = tf.nn.relu(batch_norm(conv2d(self.h_depth7, self.w_point7), is_training) + self.b_point7)

        # Layer 8
        self.w_depth8 = weight_variable([3, 3, 512, 1])
        self.b_depth8 = bias_variable([512])
        self.w_point8 = weight_variable([1, 1, 512, 512])
        self.b_point8 = bias_variable([512])
        self.h_depth8 = tf.nn.relu(batch_norm(depthwise_conv2d(self.h_point7, self.w_depth8, [1, 1, 1, 1]), is_training) + self.b_depth8)
        self.h_point8 = tf.nn.relu(batch_norm(conv2d(self.h_depth8, self.w_point8), is_training) + self.b_point8)

        # Layer 8.1
        self.w_depth81 = weight_variable([3, 3, 512, 1])
        self.b_depth81 = bias_variable([512])
        self.w_point81 = weight_variable([1, 1, 512, 512])
        self.b_point81 = bias_variable([512])
        self.h_depth81 = tf.nn.relu(batch_norm(depthwise_conv2d(self.h_point8, self.w_depth81, [1, 1, 1, 1]), is_training) + self.b_depth81)
        self.h_point81 = tf.nn.relu(batch_norm(conv2d(self.h_depth81, self.w_point81), is_training) + self.b_point81)

        # Average pooling
        self.avg_pool = tf.nn.avg_pool2d(self.h_point81, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Dropout
        self.h_fc_drop = tf.nn.dropout(tf.nn.lrn(self.avg_pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75), rate=1 - self.keep_prob)

        # Fully connected layer
        shape = self.h_fc_drop.get_shape().as_list()
        dim = shape[1] * shape[2] * shape[3]
        self.flat = tf.reshape(self.h_fc_drop, [-1, dim])
        self.W_fc = weight_variable([dim, 4])
        self.b_fc = bias_variable([4])
        self.y = tf.nn.softmax(tf.matmul(self.flat, self.W_fc) + self.b_fc)

        # Cross entropy loss
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.math.log(self.y + 1e-10), reduction_indices=[1]))

        # Collect all weights
        self.all_weights = [
            self.w_depth1, self.b_depth1, self.w_point1, self.b_point1,
            self.w_depth2, self.b_depth2, self.w_point2, self.b_point2,
            self.w_depth3, self.b_depth3, self.w_point3, self.b_point3,
            self.w_depth5, self.b_depth5, self.w_point5, self.b_point5,
            self.w_depth7, self.b_depth7, self.w_point7, self.b_point7,
            self.w_depth8, self.b_depth8, self.w_point8, self.b_point8,
            self.w_depth81, self.b_depth81, self.w_point81, self.b_point81,
            self.W_fc, self.b_fc
        ]

        self._assignment_init()
        self._optimizer_init(learning_rate=learning_rate)
        self.grad = self.optimizer.compute_gradients(self.cross_entropy, var_list=self.all_weights)
        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.acc = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self._session_init()
        self.graph_created = True
