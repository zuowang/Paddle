# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddle.trainer_config_helpers import *
import numpy as np

is_predict = get_config_arg("is_predict", bool, False)

####################Data Configuration ##################

if not is_predict:
    data_dir = './data/'
    define_py_data_sources2(
        train_list=data_dir + 'train.list',
        test_list=data_dir + 'test.list',
        module='mnist_provider',
        obj='process')

######################Algorithm Configuration #############
settings(
    batch_size=100,
    learning_rate=0.01,
    learning_method=AdamOptimizer())

#######################Network Configuration #############

# 样本集X
n_X = 784 # 28 * 28
n_z = 20 # latent variable count

X = data_layer(name='pixel', size=data_size)

# Encoder

## \mu(X) 采用二层网络
ENCODER_HIDDEN_COUNT = 400
with mixed_layer(bias_attr=True) as L1:
    L1 += full_matrix_projection(input=X, size=ENCODER_HIDDEN_COUNT)

with mixed_layer(bias_attr=True) as mu:
    mu += full_matrix_projection(input=L1, size=ENCODER_HIDDEN_COUNT)


## \Sigma(X) 采用二层网络
with mixed_layer(bias_attr=True) as L2:
    L2 += full_matrix_projection(input=X, size=ENCODER_HIDDEN_COUNT)

with mixed_layer(bias_attr=True) as log_sigma:
    log_sigma += full_matrix_projection(input=L2, size=ENCODER_HIDDEN_COUNT)

with mixed_layer(act=ExpActivation()) as sigma:
    sigma += log_sigma


## KLD = D[N(mu(X), sigma(X))||N(0, I)] = 1/2 * sum(sigma_i + mu_i^2 - log(sigma_i) - 1)
with mixed_layer() as kld:
    kld += identity_projection(input=sigma)
    kld += dotmul_operator(a=mu, b=mu)
    kld += slope_intercept_layer(input=log_sigma, slope=-1.0, intercept=-1.0)

kld = slope_intercept_layer(input=kld, slope=0.5, intercept=0.0)

#KLD = 0.5 * tf.reduce_sum(sigma + tf.pow(mu, 2) - log_sigma - 1, reduction_indices = 1) # reduction_indices = 1代表按照每个样本计算一条KLD

# epsilon = N(0, I) 采样模块
#epsilon = tf.random_normal(tf.shape(sigma), name = 'epsilon')

epsilon = data_layer(name='epsilon', size=data_size)


# z = mu + sigma^ 0.5 * epsilon
#z = mu + tf.exp(0.5 * log_sigma) * epsilon

with mixed_layer(act=ExpActivation()) as L3:
    L3 += slope_intercept_layer(input=log_sigma, slope=0.5, intercept=0.0)

with mixed_layer(act=ExpActivation()) as z:
    z += identity_projection(input=mu)
    z += dotmul_operator(a=L3, b=epsilon)




# Decoder ||f(z) - X|| ^ 2 重建的X与X的欧式距离，更加成熟的做法是使用crossentropy
def buildDecoderNetwork(z):
    # 构建一个二层神经网络，因为二层神经网络可以逼近任何函数
    DECODER_HIDDEN_COUNT = 400
    with mixed_layer(bias_attr=True) as layer1:
        layer1 += full_matrix_projection(input=X, size=ENCODER_HIDDEN_COUNT)

    with mixed_layer(bias_attr=True) as layer2:
        layer2 += full_matrix_projection(input=layer1, size=ENCODER_HIDDEN_COUNT)

    return layer2
    #layer1 = Layer(z, DECODER_HIDDEN_COUNT)
    #layer2 = Layer(layer1.output, n_X)
    #return layer2.raw_output

reconstructed_X = buildDecoderNetwork(z)

reconstruction_loss = cross_entropy(input=reconstructed_X, label=X)

#cost = sum_cost(input=cross_entropy)
#reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(reconstructed_X, X), reduction_indices = 1)

with mixed_layer() as loss:
    loss += identity_projection(input=reconstruction_loss)
    loss += identity_projection(input=kld)
#loss = tf.reduce_mean(reconstruction_loss + kld)

#data_size = 1 * 28 * 28
#label_size = 10
#img = data_layer(name='pixel', size=data_size)



# small_vgg is predined in trainer_config_helpers.network
#predict = small_vgg(input_image=img, num_channels=1, num_classes=label_size)

if not is_predict:
    #lbl = data_layer(name="label", size=label_size)
    #inputs(img, lbl)
    #outputs(classification_cost(input=predict, label=lbl))
    output(cost)
#else:
    #outputs(predict)
