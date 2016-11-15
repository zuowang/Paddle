# edit-mode: -*- python -*-

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
import math
dict_file = "./data/dict.txt"
word_dict = dict()
word_count = len(open(dict_file, 'r').readlines())
word_count_sqrt = math.ceil(math.sqrt(word_count))
print word_count_sqrt
with open(dict_file, 'r') as f:
    row = 0
    col = 0
    for i, line in enumerate(f):
        w = line.strip().split()[0]
        word_dict[w] = (row, col)
        col += 1
        if col == word_count_sqrt:
            row += 1
            col = 0


is_predict = get_config_arg('is_predict', bool, False)
trn = 'data/train.list' if not is_predict else None
tst = 'data/test.list' if not is_predict else 'data/pred.list'
process = 'process' if not is_predict else 'process_predict'
define_py_data_sources2(train_list=trn,
                        test_list=tst,
                        module="dataprovider_2cs",
                        obj=process,
                        args={"dictionary": word_dict,
                              "word_count_sqrt": word_count_sqrt})

batch_size = 128 if not is_predict else 1
settings(
    batch_size=batch_size,
    learning_rate=2e-3,
    learning_method=AdamOptimizer(),
    regularization=L2Regularization(8e-4),
    gradient_clipping_threshold=25
)

bias_attr = ParamAttr(initial_std=0.,l2_rate=0.)

row_vector = data_layer(name="row_vector", size=word_count_sqrt)
emb1 = embedding_layer(input=row_vector, size=128)
fc1 = fc_layer(input=emb1, size=512,
               act=LinearActivation(),
               bias_attr=bias_attr,
               layer_attr=ExtraAttr(drop_rate=0.1))
lstm1 = lstmemory(input=fc1, act=TanhActivation(),
                   bias_attr=bias_attr,
                  layer_attr=ExtraAttr(drop_rate=0.25))

col_vector = data_layer(name="col_vector", size=word_count_sqrt)
emb2 = embedding_layer(input=row_vector, size=128)

input_tmp = [emb2, lstm1]
fc2 = fc_layer(input=input_tmp, size=512,
               act=LinearActivation(),
               bias_attr=bias_attr,
               layer_attr=ExtraAttr(drop_rate=0.1))
lstm2 = lstmemory(input=fc2, act=TanhActivation(),
                  bias_attr=bias_attr,
                  layer_attr=ExtraAttr(drop_rate=0.25))

input_tmp = [lstm1, lstm2]
lstm_last = pooling_layer(input=input_tmp, pooling_type=MaxPooling())
output = fc_layer(input=lstm_last, size=2,
                  bias_attr=bias_attr,
                  act=SoftmaxActivation())
if is_predict:
    maxid = maxid_layer(output)
    outputs([maxid, output])
else:
    label = data_layer(name="label", size=2)
    cls = classification_cost(input=output, label=label)
    outputs(cls)
