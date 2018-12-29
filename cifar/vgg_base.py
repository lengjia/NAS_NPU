# Copyright chenweiwei@ict.ac.cn.
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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 

class ConvNet(object):
    """construct ConvNet"""
    def __init__(self, is_training,data_format,batch_norm_decay,batch_norm_epsilon):
        super(ConvNet, self).__init__()
        self._batch_norm_decay = batch_norm_decay
        self._batch_norm_epsilon = batch_norm_epsilon
        self._is_training = is_training
        assert data_format in ('channels_first','channels_last')
        self._data_format = data_format

    def forward_pass(self,x):
        raise NotImplementedError('forward pass function is not implemented!')

    def _relu(self,x):
        return tf.nn.relu(x)

    def _relu_layer(self,x):
        with tf.name_scope('relu') as name_scope:
            x = self._relu(x)
        tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
        return x

    def _conv(self,x,kernel_size,filters,strides,is_atrous=False):
        padding = 'SAME'
        if not is_atrous and strides >1:
            pad = kernel_size -1
            pad_beg = pad // 2
            pad_end = pad - pad_beg
            if self.data_format == 'channels_first':
                x = tf.pad(x,[[0,0],[0,0],[pad_beg,pad_end],[pad_beg,pad_end]])
            else:
                x = tf.pad(x,[[0,0],[pad_beg,pad_end],[pad_beg,pad_end],[0,0]])
            padding = 'VALID'
        return tf.layers.conv2d(
            inputs=x,
            kernel_size=kernel_size,
            filters=filters,
            strides=strides,
            padding=padding,
            use_bias=False,
            data_format=self._data_format)

    def _conv_layer(self, x,kernel_size,filters,strides,is_atrous=False):
        with tf.name_scope('conv') as name_scope:
            x = self._conv(x,kernel_size,filters,strides,is_atrous)
            x = self._batch_norm(x)
            x = self._relu(x)
        tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
        return x

    def _batch_norm(self,x):
        if self._data_format == 'channels_first':
            data_format = 'NCHW'
        else:
            data_format = 'NHWC'
        return tf.contrib.layers.batch_norm(x,
            decay=self._batch_norm_decay,
            center=True,
            scale=True,
            epsilon=self._batch_norm_epsilon,
            is_training=self._is_training,
            fused=True,
            data_format=data_format)


    def _full_connected_layer(self,x,out_dim):
        with tf.name_scope('fully_connected') as name_scope:
            if x.get_shape().ndims == 4:
                x = self._global_avg_pool(x)
            x = tf.layers.dense(x,out_dim)
        tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
        return x

    def _avg_pool_layer(self,x,pool_size, stride):
        with tf.name_scope('avg_pool') as name_scope:
            x = tf.layers.average_pooling2d(
                x,pool_size,stride, 'SAME',data_format=self._data_format)
        tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
        return x

    def _global_avg_pool(self,x):
        with tf.name_scope('global_avg_pool') as name_scope:
            assert x.get_shape().ndims == 4
            if self._data_format == 'channels_first':
                x = tf.reduce_mean(x,[2,3])
            else:
                x = tf.reduce_mean(x,[1,2])
        tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
        return x

    def _max_pool_layer(self,x,pool_size,stride):
        with tf.name_scope('max_pool') as name_scope:
            x = tf.layers.max_pooling2d(
                x,pool_size,stride,'SAME',data_format=self._data_format)
        tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
        return x

    def _softmax_layer(self,x):
        with tf.name_scope('softmax') as name_scope:
            x = tf.nn.softmax(x)
        tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
        return x


