"""model of vgg-net 16"""
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import vgg_base

class VGG16(vgg_base.ConvNet):
    """docstring for VGG16""" 
    def __init__(self, is_training,batch_norm_decay,batch_norm_epsilon,data_format='channels_first'):
        super(VGG16,self).__init__(is_training,data_format,batch_norm_decay,batch_norm_epsilon)
        self.num_classes = 10+1 
        self.block_num=2
        self.block_filter_sizes = [256,512]

    def forward_pass(self,x,input_data_format='channels_last'):
        if self._data_format != input_data_format:
            if input_data_format == 'channels_last':
                x = tf.transpose(x,[0,3,1,2])
            else:
                x = tf.transpose(x,[0,2,3,1])
        x = x/128.0 -1 
        x = self._conv_layer(x,3,64,1)
        x = self._conv_layer(x,3,64,1)
        x = self._max_pool_layer(x,3,1)
        x = self._conv_layer(x,3,128,1)
        x = self._conv_layer(x,3,128,1)
        x = self._max_pool_layer(x,3,2)
        for bfs in self.block_filter_sizes:
            for i in range(self.block_num):
                x = self._conv_layer(x,3,bfs,1)
            x = self._max_pool_layer(x,3,2)
        #x = self._full_connected_layer(x,512)
        #x = self._full_connected_layer(x,256)
        x = self._full_connected_layer(x,self.num_classes)
        #x = self._softmax_layer(x)

        return [x]

        

        