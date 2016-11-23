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

n_X = 784 # 28 * 28
n_z = 20 # latent variable count

X = data_layer(name='pixel', size=n_X)

# Encoder

ENCODER_HIDDEN_COUNT = 400
with mixed_layer(bias_attr=True) as L1:
    L1 += full_matrix_projection(input=X, size=ENCODER_HIDDEN_COUNT)

with mixed_layer(bias_attr=True) as mu:
    mu += full_matrix_projection(input=L1, size=n_z)


with mixed_layer(bias_attr=True) as L2:
    L2 += full_matrix_projection(input=X, size=ENCODER_HIDDEN_COUNT)

with mixed_layer(bias_attr=True) as log_sigma:
    log_sigma += full_matrix_projection(input=L2, size=n_z)

with mixed_layer(act=ExpActivation()) as sigma:
    sigma += identity_projection(input=log_sigma)


## KLD = D[N(mu(X), sigma(X))||N(0, I)] = 1/2 * sum(sigma_i + mu_i^2 - log(sigma_i) - 1)

kld0 = slope_intercept_layer(input=log_sigma, slope=-1.0, intercept=-1.0)

with mixed_layer() as kld1:
    kld1 += identity_projection(input=sigma)
    kld1 += dotmul_operator(a=mu, b=mu)
    kld1 += identity_projection(kld0)

    kld1 += dotmul_operator(a=mu, b=mu)
    kld1 += identity_projection(kld0)

kld2 = sum_cost(input=kld1)

kld = slope_intercept_layer(input=kld2, slope=0.5, intercept=0.0)

#epsilon = tf.random_normal(tf.shape(sigma), name = 'epsilon')

epsilon = data_layer(name='epsilon', size=n_z)


# z = mu + sigma^ 0.5 * epsilon
#z = mu + tf.exp(0.5 * log_sigma) * epsilon

log_sigma1 = slope_intercept_layer(input=log_sigma, slope=0.5, intercept=0.0)

with mixed_layer(act=ExpActivation()) as L3:
    L3 += identity_projection(input=log_sigma1)

with mixed_layer(act=ExpActivation()) as z:
    z += identity_projection(input=mu)
    #z += dotmul_operator(a=L3, b=epsilon)

def buildDecoderNetwork(z):
    DECODER_HIDDEN_COUNT = 400
    with mixed_layer(bias_attr=True) as layer1:
        layer1 += full_matrix_projection(input=z, size=ENCODER_HIDDEN_COUNT)

    with mixed_layer(bias_attr=True) as layer2:
        layer2 += full_matrix_projection(input=layer1, size=n_X)

    return layer2
    #layer1 = Layer(z, DECODER_HIDDEN_COUNT)
    #layer2 = Layer(layer1.output, n_X)
    #return layer2.raw_output

    #layer2 = Layer(layer1.output, n_X)
    #return layer2.raw_output

reconstructed_X = buildDecoderNetwork(z)

#reconstruction_loss = cross_entropy(input=reconstructed_X, label=X)
reconstruction_loss = cross_entropy_with_selfnorm(input=reconstructed_X, label=X)

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
    outputs(loss)
#else:
    #outputs(predict)



