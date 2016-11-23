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

kld2 = sum_cost(input=kld1)

kld = slope_intercept_layer(input=kld2, slope=0.5, intercept=0.0)

similarity = cos_sim(a=mu, b=sigma)

if not is_predict:
    #inputs(X)
    #outputs(kld)
    outputs(similarity)


